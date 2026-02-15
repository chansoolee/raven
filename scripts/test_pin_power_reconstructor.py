"""
Unit tests for pin_power_reconstructor module.

Test strategy:
- Toy example: 1 assembly, 2x2 pin grid, 2 axial planes, 2 timesteps
- Verifies: CoreMap, PinPowerStore, PinPowerParser, PinPowerAnalyzer
- Also tests against the actual PARCS sample .pin file (if available)
"""

from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pin_power_reconstructor import (
    CoreMap,
    PinDeltaResult,
    PinPowerAnalyzer,
    PinPowerParser,
    PinPowerStore,
    ParserConfig,
    coremap_from_parcs_input,
    get_pin_series,
)


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def simple_coremap():
    """1-assembly core map: 1x1 grid, type 20 = fuel."""
    layout = np.array([[20]])
    return CoreMap(layout, fuel_ids=None, symmetry="full")


@pytest.fixture
def multi_coremap():
    """3x3 core map with empties and reflector.

    Layout::
        20  30  10
        30  20   0
        10   0   0

    Fuel (20, 30): positions (0,0), (0,1), (1,0), (1,1)
    Reflector (10): (0,2), (2,0)
    Empty (0): (1,2), (2,1), (2,2)
    """
    layout = np.array([
        [20, 30, 10],
        [30, 20,  0],
        [10,  0,  0],
    ])
    return CoreMap(layout, fuel_ids={20, 30}, symmetry="quarter")


@pytest.fixture
def toy_pin_file(tmp_path):
    """Create a toy .pin file with 1 assembly, 2x2 pins, 2 axial, 2 timesteps.

    Pin values for assembly (1,1):
      Timestep 0 (t=0.0):
        Plane 2 (az=0):  [[1.0, 2.0], [3.0, 4.0]]
        Plane 3 (az=1):  [[5.0, 6.0], [7.0, 8.0]]
      Timestep 1 (t=1.0):
        Plane 2 (az=0):  [[1.1, 2.2], [3.3, 4.4]]
        Plane 3 (az=1):  [[5.5, 6.6], [7.7, 8.8]]
    """
    content = textwrap.dedent("""\
        At Time:    0.0000
        Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    2   30.480 , Normalization Factor:  1.00000E+00
                     1      2
              1 1.0000 2.0000
              2 3.0000 4.0000
           Peak at (i,j):    2   1 , Value:     4.0000
        Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    3   30.480 , Normalization Factor:  1.00000E+00
                     1      2
              1 5.0000 6.0000
              2 7.0000 8.0000
           Peak at (i,j):    2   1 , Value:     8.0000
        At Time:    1.0000
        Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    2   30.480 , Normalization Factor:  1.00000E+00
                     1      2
              1 1.1000 2.2000
              2 3.3000 4.4000
           Peak at (i,j):    2   1 , Value:     4.4000
        Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    3   30.480 , Normalization Factor:  1.00000E+00
                     1      2
              1 5.5000 6.6000
              2 7.7000 8.8000
           Peak at (i,j):    2   1 , Value:     8.8000
    """)
    fpath = tmp_path / "toy.pin"
    fpath.write_text(content)
    return fpath


@pytest.fixture
def toy_parcs_input(tmp_path):
    """Create a minimal PARCS input file for the toy example."""
    content = textwrap.dedent("""\
        CASEID toy
        GEOM
              geo_dim 1 1 4 1 1
              Rad_Conf
                20
              Boun_cond   0 0 0 0 0 0
              pincal_loc
                20
        .
    """)
    fpath = tmp_path / "toy.inp"
    fpath.write_text(content)
    return fpath


# ==========================================================================
# CoreMap Tests
# ==========================================================================

class TestCoreMap:
    def test_single_assembly(self, simple_coremap):
        cm = simple_coremap
        assert cm.n_assemblies == 1
        assert cm.n_rows == 1
        assert cm.n_cols == 1
        assert cm.asm_id(0, 0) == 0
        assert cm.pos_of(0) == (0, 0)
        assert cm.is_valid(0, 0) is True

    def test_multi_assembly_mapping(self, multi_coremap):
        cm = multi_coremap
        # Fuel positions: (0,0)=20, (0,1)=30, (1,0)=30, (1,1)=20
        assert cm.n_assemblies == 4
        assert cm.asm_id(0, 0) == 0
        assert cm.asm_id(0, 1) == 1
        assert cm.asm_id(1, 0) == 2
        assert cm.asm_id(1, 1) == 3

    def test_invalid_position_raises(self, multi_coremap):
        cm = multi_coremap
        # (0,2) is reflector (10), not in fuel_ids
        with pytest.raises(KeyError, match="not a valid fuel assembly"):
            cm.asm_id(0, 2)
        # (2,2) is empty
        with pytest.raises(KeyError, match="not a valid fuel assembly"):
            cm.asm_id(2, 2)

    def test_is_valid_false_for_empty(self, multi_coremap):
        assert multi_coremap.is_valid(2, 2) is False
        assert multi_coremap.is_valid(1, 2) is False

    def test_is_valid_false_for_reflector(self, multi_coremap):
        assert multi_coremap.is_valid(0, 2) is False
        assert multi_coremap.is_valid(2, 0) is False

    def test_valid_positions_order(self, multi_coremap):
        positions = multi_coremap.valid_positions()
        assert positions == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_symmetry_metadata(self, multi_coremap):
        assert multi_coremap.symmetry == "quarter"

    def test_pos_of_roundtrip(self, multi_coremap):
        cm = multi_coremap
        for pos in cm.valid_positions():
            aid = cm.asm_id(*pos)
            assert cm.pos_of(aid) == pos


# ==========================================================================
# PinPowerStore Tests
# ==========================================================================

class TestPinPowerStore:
    def test_ndarray_backend(self):
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=2, backend="ndarray"
        )
        assert store.data.shape == (1 * 4 * 2 * 2,)  # 16
        assert store.data.dtype == np.float64

    def test_memmap_backend(self, tmp_path):
        mpath = tmp_path / "test.dat"
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=2, backend="memmap",
            memmap_path=mpath,
        )
        assert isinstance(store.data, np.memmap)
        assert mpath.exists()

    def test_set_and_get_pin(self):
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=2, backend="ndarray"
        )
        store.set_pin(asm_id=0, pin_r=0, pin_c=1, t=0, az=0, value=42.0)
        series = store.get_pin_series(asm_id=0, pin_r=0, pin_c=1)
        assert series.shape == (2, 2)
        assert series[0, 0] == 42.0
        assert series[0, 1] == 0.0  # unset

    def test_set_plane(self):
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=2, backend="ndarray"
        )
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        store.set_plane(asm_id=0, t=0, az=0, pin_matrix=mat)
        plane = store.get_plane(asm_id=0, t=0, az=0)
        np.testing.assert_array_equal(plane, mat)

    def test_get_plane(self):
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=1, backend="ndarray"
        )
        mat = np.array([[10.0, 20.0], [30.0, 40.0]])
        store.set_plane(asm_id=0, t=0, az=1, pin_matrix=mat)
        result = store.get_plane(asm_id=0, t=0, az=1)
        np.testing.assert_array_equal(result, mat)

    def test_pin_series_contiguous_layout(self):
        """Verify that a pin's (Nt, Naz) data is contiguous in flat storage."""
        store = PinPowerStore(
            n_assemblies=2, n_pin_r=2, n_pin_c=2,
            n_axial=3, n_time=4, backend="ndarray"
        )
        # Fill specific pin with known pattern
        for t in range(4):
            for az in range(3):
                store.set_pin(asm_id=1, pin_r=1, pin_c=0, t=t, az=az,
                              value=t * 100 + az)
        series = store.get_pin_series(asm_id=1, pin_r=1, pin_c=0)
        expected = np.array([
            [0, 1, 2],
            [100, 101, 102],
            [200, 201, 202],
            [300, 301, 302],
        ], dtype=np.float64)
        np.testing.assert_array_equal(series, expected)

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="Store size is zero"):
            PinPowerStore(
                n_assemblies=0, n_pin_r=2, n_pin_c=2,
                n_axial=2, n_time=2, backend="ndarray"
            )

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            PinPowerStore(
                n_assemblies=1, n_pin_r=2, n_pin_c=2,
                n_axial=2, n_time=2, backend="hdf5"
            )


# ==========================================================================
# PinPowerParser Tests (Toy Example)
# ==========================================================================

class TestPinPowerParser:
    def test_toy_parse(self, simple_coremap, toy_pin_file):
        """Parse the toy .pin file and verify all values."""
        config = ParserConfig(
            axial_bottom_to_top=True,
            fuel_plane_range=(2, 3),
            file_index_base=1,
        )
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)

        assert store.n_asm == 1
        assert store.n_pin_r == 2
        assert store.n_pin_c == 2
        assert store.n_axial == 2
        assert store.n_time == 2

        # Timestep 0, Plane 2 (az=0): [[1,2],[3,4]]
        plane_t0_az0 = store.get_plane(asm_id=0, t=0, az=0)
        np.testing.assert_allclose(plane_t0_az0, [[1.0, 2.0], [3.0, 4.0]])

        # Timestep 0, Plane 3 (az=1): [[5,6],[7,8]]
        plane_t0_az1 = store.get_plane(asm_id=0, t=0, az=1)
        np.testing.assert_allclose(plane_t0_az1, [[5.0, 6.0], [7.0, 8.0]])

        # Timestep 1, Plane 2 (az=0): [[1.1,2.2],[3.3,4.4]]
        plane_t1_az0 = store.get_plane(asm_id=0, t=1, az=0)
        np.testing.assert_allclose(plane_t1_az0, [[1.1, 2.2], [3.3, 4.4]])

        # Timestep 1, Plane 3 (az=1): [[5.5,6.6],[7.7,8.8]]
        plane_t1_az1 = store.get_plane(asm_id=0, t=1, az=1)
        np.testing.assert_allclose(plane_t1_az1, [[5.5, 6.6], [7.7, 8.8]])

    def test_get_pin_series_from_parser(self, simple_coremap, toy_pin_file):
        """Verify get_pin_series returns correct (Nt, Naz) shape and values."""
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)

        # Pin (0,0): t0_az0=1.0, t0_az1=5.0, t1_az0=1.1, t1_az1=5.5
        series_00 = get_pin_series(store, simple_coremap, 0, 0, 0, 0)
        assert series_00.shape == (2, 2)
        np.testing.assert_allclose(series_00, [[1.0, 5.0], [1.1, 5.5]])

        # Pin (1,1): t0_az0=4.0, t0_az1=8.0, t1_az0=4.4, t1_az1=8.8
        series_11 = get_pin_series(store, simple_coremap, 0, 0, 1, 1)
        np.testing.assert_allclose(series_11, [[4.0, 8.0], [4.4, 8.8]])

    def test_time_values_metadata(self, simple_coremap, toy_pin_file):
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)
        np.testing.assert_allclose(store.time_values, [0.0, 1.0])

    def test_fuel_planes_metadata(self, simple_coremap, toy_pin_file):
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)
        assert store.fuel_plane_indices == [2, 3]

    def test_axial_reversal(self, simple_coremap, toy_pin_file):
        """With axial_bottom_to_top=False, plane ordering should reverse."""
        config = ParserConfig(
            axial_bottom_to_top=False,
            fuel_plane_range=(2, 3),
        )
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)

        # Now plane 3 (higher in file) maps to az=0, plane 2 to az=1
        # Plane 3 data: [[5,6],[7,8]] at t=0
        plane_az0 = store.get_plane(asm_id=0, t=0, az=0)
        np.testing.assert_allclose(plane_az0, [[5.0, 6.0], [7.0, 8.0]])

        # Plane 2 data: [[1,2],[3,4]] at t=0
        plane_az1 = store.get_plane(asm_id=0, t=0, az=1)
        np.testing.assert_allclose(plane_az1, [[1.0, 2.0], [3.0, 4.0]])


# ==========================================================================
# PinPowerAnalyzer Tests
# ==========================================================================

class TestPinPowerAnalyzer:
    def test_compute_delta(self, simple_coremap, toy_pin_file):
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)
        analyzer = PinPowerAnalyzer(store, simple_coremap, fill_t0="zero")

        # Pin (0,0): series = [[1.0, 5.0], [1.1, 5.5]]
        # delta[0,:] = [0, 0],  delta[1,:] = [0.1, 0.5]
        delta = analyzer.compute_delta(0, 0, 0, 0)
        assert delta.shape == (2, 2)
        np.testing.assert_allclose(delta[0, :], [0.0, 0.0])
        np.testing.assert_allclose(delta[1, :], [0.1, 0.5], atol=1e-10)

    def test_compute_delta_nan_fill(self, simple_coremap, toy_pin_file):
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)
        analyzer = PinPowerAnalyzer(store, simple_coremap, fill_t0="nan")

        delta = analyzer.compute_delta(0, 0, 0, 0)
        assert np.all(np.isnan(delta[0, :]))
        np.testing.assert_allclose(delta[1, :], [0.1, 0.5], atol=1e-10)

    def test_rank_max_delta(self, simple_coremap, toy_pin_file):
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)
        analyzer = PinPowerAnalyzer(store, simple_coremap)

        results = analyzer.rank_max_delta()

        # 4 pins total (2x2), each has 2 timesteps so delta is valid
        assert len(results) == 4

        # Pin (1,1) has max delta: max(|4.4-4.0|, |8.8-8.0|) = 0.8
        top = results[0]
        assert top.asm_row == 0
        assert top.asm_col == 0
        assert top.pin_row == 1
        assert top.pin_col == 1
        np.testing.assert_allclose(top.max_abs_delta, 0.8, atol=1e-10)
        assert top.axial_of_max == 1  # az=1 has larger delta (8.8-8.0=0.8 > 4.4-4.0=0.4)

        # Results should be descending
        deltas = [r.max_abs_delta for r in results]
        assert deltas == sorted(deltas, reverse=True)

    def test_rank_top_n(self, simple_coremap, toy_pin_file):
        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse(toy_pin_file)
        analyzer = PinPowerAnalyzer(store, simple_coremap)

        results = analyzer.rank_max_delta(top_n=2)
        assert len(results) == 2

    def test_rank_single_timestep_empty(self, simple_coremap):
        """With only 1 timestep, rank_max_delta returns empty list."""
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=1, backend="ndarray"
        )
        analyzer = PinPowerAnalyzer(store, simple_coremap)
        results = analyzer.rank_max_delta()
        assert len(results) == 0


# ==========================================================================
# coremap_from_parcs_input Tests
# ==========================================================================

class TestCoremapFromParcsInput:
    def test_toy_input(self, toy_parcs_input):
        cm = coremap_from_parcs_input(toy_parcs_input)
        assert cm.n_assemblies == 1
        assert cm.asm_id(0, 0) == 0

    def test_real_parcs_input(self):
        """Test with actual PARCS input file if available."""
        real_input = Path(
            r"C:\codes\raven\tests\framework\CodeInterfaceTests"
            r"\PARCS\PARCS-RAVEN-GA\sampleGA-PARCS\b2_r1\input.inp"
        )
        if not real_input.exists():
            pytest.skip("Real PARCS input not found")

        cm = coremap_from_parcs_input(real_input, pincal_section=True)

        # pincal_loc has 47 nonzero entries (fuel assemblies with pin calc)
        assert cm.n_assemblies == 47
        assert cm.symmetry == "quarter"

        # Check a few known positions from pincal_loc:
        # Row 0: 60 40 60 60 60 40 60 50  0  -> 8 valid
        assert cm.is_valid(0, 0) is True   # type 60
        assert cm.is_valid(0, 7) is True   # type 50
        assert cm.is_valid(0, 8) is False  # type 0 (zero in pincal_loc)

        # Row 8:  0  0  0  0  0  0  0  0  0 -> all invalid
        for c in range(9):
            assert cm.is_valid(8, c) is False


# ==========================================================================
# Integration Test: Real PARCS .pin File
# ==========================================================================

class TestRealPARCSFile:
    """Integration tests against the actual PARCS sample file."""

    PARCS_DIR = Path(
        r"C:\codes\raven\tests\framework\CodeInterfaceTests"
        r"\PARCS\PARCS-RAVEN-GA\sampleGA-PARCS\b2_r1"
    )

    @pytest.fixture
    def real_store(self):
        pin_file = self.PARCS_DIR / "input.inp.pin"
        inp_file = self.PARCS_DIR / "input.inp"
        if not pin_file.exists() or not inp_file.exists():
            pytest.skip("Real PARCS files not found")

        cm = coremap_from_parcs_input(inp_file, pincal_section=True)
        config = ParserConfig(
            axial_bottom_to_top=True,
            fuel_plane_range=(2, 13),
            file_index_base=1,
        )
        parser = PinPowerParser(cm, config, backend="ndarray")
        store = parser.parse(pin_file)
        return store, cm

    def test_dimensions(self, real_store):
        store, cm = real_store
        assert store.n_asm == 47
        assert store.n_pin_r == 17
        assert store.n_pin_c == 17
        assert store.n_axial == 12  # planes 2-13
        assert store.n_time == 1    # single timestep

    def test_known_value(self, real_store):
        """Check a known value from the file.

        Assembly (1,1) i.e. (0,0) in 0-based, Plane 2, pin (1,1):
        File says: 1.0886 (row=1, col=1 of the pin matrix)
        That's pin_row=0, pin_col=0 in 0-based.
        """
        store, cm = real_store
        aid = cm.asm_id(0, 0)
        plane = store.get_plane(aid, t=0, az=0)  # plane 2 -> az=0
        np.testing.assert_allclose(plane[0, 0], 1.0886, atol=1e-4)

    def test_corner_pin(self, real_store):
        """Check bottom-right pin of assembly (1,1), plane 2.

        File: row 17, col 17 -> pin (16,16) 0-based -> value 1.0936
        """
        store, cm = real_store
        aid = cm.asm_id(0, 0)
        plane = store.get_plane(aid, t=0, az=0)
        np.testing.assert_allclose(plane[16, 16], 1.0936, atol=1e-4)

    def test_last_assembly(self, real_store):
        """Check last assembly in pincal_loc: (2,8) i.e. (1,7) 0-based.

        Assembly (2,8) Plane 13 (az=11), pin (1,1) -> should match file value.
        From file line 12184: row 1, col 1 = 1.9816
        """
        store, cm = real_store
        aid = cm.asm_id(1, 7)  # Assembly (2,8) -> 0-based (1,7)
        plane = store.get_plane(aid, t=0, az=11)  # plane 13 -> az=11
        np.testing.assert_allclose(plane[0, 0], 1.9816, atol=1e-4)

    def test_get_pin_series_shape(self, real_store):
        store, cm = real_store
        series = get_pin_series(store, cm, 0, 0, 0, 0)
        assert series.shape == (1, 12)  # 1 timestep, 12 axial planes

    def test_all_values_positive(self, real_store):
        """All pin powers should be positive (physical constraint)."""
        store, cm = real_store
        # Check a random selection of assemblies
        for aid in [0, 10, 20, 30, 46]:
            for az in range(store.n_axial):
                plane = store.get_plane(aid, t=0, az=az)
                assert np.all(plane > 0), (
                    f"Non-positive value found: asm_id={aid}, az={az}"
                )


# ==========================================================================
# Edge Cases
# ==========================================================================

class TestEdgeCases:
    def test_coremap_all_empty(self):
        layout = np.array([[0, 0], [0, 0]])
        cm = CoreMap(layout)
        assert cm.n_assemblies == 0

    def test_coremap_all_fuel(self):
        layout = np.array([[10, 20], [30, 40]])
        cm = CoreMap(layout)
        assert cm.n_assemblies == 4

    def test_store_repr(self):
        store = PinPowerStore(
            n_assemblies=1, n_pin_r=2, n_pin_c=2,
            n_axial=2, n_time=1, backend="ndarray"
        )
        r = repr(store)
        assert "asm=1" in r
        assert "ndarray" in r

    def test_multi_file_timesteps(self, simple_coremap, tmp_path):
        """Test parsing multiple files, one timestep each."""
        content_t0 = textwrap.dedent("""\
            At Time:    0.0000
            Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    2   30.480 , Normalization Factor:  1.00000E+00
                         1      2
                  1 1.0000 2.0000
                  2 3.0000 4.0000
               Peak at (i,j):    2   1 , Value:     4.0000
            Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    3   30.480 , Normalization Factor:  1.00000E+00
                         1      2
                  1 5.0000 6.0000
                  2 7.0000 8.0000
               Peak at (i,j):    2   1 , Value:     8.0000
        """)
        content_t1 = textwrap.dedent("""\
            At Time:    5.0000
            Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    2   30.480 , Normalization Factor:  1.00000E+00
                         1      2
                  1 1.5000 2.5000
                  2 3.5000 4.5000
               Peak at (i,j):    2   1 , Value:     4.5000
            Assembly Coordinate (i,j):    1   1 , Plane Index & Height:    3   30.480 , Normalization Factor:  1.00000E+00
                         1      2
                  1 5.5000 6.5000
                  2 7.5000 8.5000
               Peak at (i,j):    2   1 , Value:     8.5000
        """)
        f0 = tmp_path / "t0.pin"
        f1 = tmp_path / "t1.pin"
        f0.write_text(content_t0)
        f1.write_text(content_t1)

        config = ParserConfig(fuel_plane_range=(2, 3))
        parser = PinPowerParser(simple_coremap, config, backend="ndarray")
        store = parser.parse([f0, f1])

        assert store.n_time == 2
        np.testing.assert_allclose(store.time_values, [0.0, 5.0])

        series = store.get_pin_series(0, 0, 0)  # pin (0,0)
        np.testing.assert_allclose(series, [[1.0, 5.0], [1.5, 5.5]])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
