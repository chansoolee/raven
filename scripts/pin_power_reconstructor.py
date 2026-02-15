"""
PARCS Pin-Power Reconstruction Module
======================================

Parses PARCS reactor core pin-by-pin power output files (.pin format),
stores data in a memory-efficient 1D flattened numpy.memmap, and provides
analysis utilities (temporal deltas, ranking).

Addressable by: (assembly_row, assembly_col, pin_row, pin_col, axial, timestep)

Memory layout (1D flat):
    data[global_pin_id * (Nt * Naz) + t * Naz + az]
where:
    global_pin_id = asm_id * (Npin_r * Npin_c) + pin_row * Npin_c + pin_col
    asm_id = CoreMap.asm_id(row, col)  # contiguous 0..Nasm-1, empties excluded

Author: auto-generated for RAVEN/PARCS analysis pipeline
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# CoreMap: 2D (row, col) <-> 1D asm_id mapping
# ---------------------------------------------------------------------------

class CoreMap:
    """Maps 2D core-map positions to contiguous 1D assembly IDs.

    Only *fuel* positions (nonzero entries that are NOT reflector-only) receive
    an ``asm_id``.  Empty (``0``) and reflector-only positions are excluded.

    Parameters
    ----------
    layout_2d : array-like, shape (Nr, Nc)
        Integer core layout.  ``0`` = empty, positive = assembly type.
    fuel_ids : set[int] | None
        Assembly-type IDs that are considered fuel (will receive pin-power
        data).  If *None*, every nonzero entry is treated as fuel.
    symmetry : {'full', 'quarter', 'octant'}
        Symmetry domain.  Currently stored as metadata; the caller is
        responsible for expanding quarter -> full if desired.
    origin : str
        ``'top-left'`` means ``(row=0, col=0)`` is the top-left corner of
        the printed core map (default PARCS convention).
    """

    def __init__(
        self,
        layout_2d: np.ndarray,
        fuel_ids: Optional[set] = None,
        symmetry: str = "full",
        origin: str = "top-left",
    ) -> None:
        self._layout = np.asarray(layout_2d, dtype=int)
        self._symmetry = symmetry
        self._origin = origin

        # Build valid-position mask
        if fuel_ids is None:
            self._valid_mask = self._layout > 0
        else:
            self._valid_mask = np.isin(self._layout, list(fuel_ids))

        # 2D -> asm_id  (row-major scan of valid positions)
        self._pos_to_id: Dict[Tuple[int, int], int] = {}
        self._id_to_pos: List[Tuple[int, int]] = []
        idx = 0
        for r in range(self._layout.shape[0]):
            for c in range(self._layout.shape[1]):
                if self._valid_mask[r, c]:
                    self._pos_to_id[(r, c)] = idx
                    self._id_to_pos.append((r, c))
                    idx += 1

    # -- public API ----------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return self._layout.shape[0]

    @property
    def n_cols(self) -> int:
        return self._layout.shape[1]

    @property
    def n_assemblies(self) -> int:
        return len(self._id_to_pos)

    @property
    def symmetry(self) -> str:
        return self._symmetry

    def is_valid(self, row: int, col: int) -> bool:
        """Return True if (row, col) is a valid fuel assembly position."""
        return (row, col) in self._pos_to_id

    def asm_id(self, row: int, col: int) -> int:
        """Return contiguous assembly ID for a 2D position.

        Raises ``KeyError`` if the position is empty or reflector-only.
        """
        try:
            return self._pos_to_id[(row, col)]
        except KeyError:
            raise KeyError(
                f"Position ({row}, {col}) is not a valid fuel assembly "
                f"(empty or reflector-only in core map)."
            )

    def pos_of(self, asm_id: int) -> Tuple[int, int]:
        """Return ``(row, col)`` for a given assembly ID."""
        return self._id_to_pos[asm_id]

    def valid_positions(self) -> List[Tuple[int, int]]:
        """Return all valid (row, col) in asm_id order."""
        return list(self._id_to_pos)

    def __repr__(self) -> str:
        return (
            f"CoreMap({self.n_rows}x{self.n_cols}, "
            f"n_asm={self.n_assemblies}, sym={self._symmetry})"
        )


# ---------------------------------------------------------------------------
# PinPowerStore: memmap-backed 1D storage
# ---------------------------------------------------------------------------

class PinPowerStore:
    """Memory-efficient, memmap-backed pin-power storage.

    Internal 1D layout::

        index = global_pin_id * (Nt * Naz) + t * Naz + az

    where ``global_pin_id = asm_id * n_pin + pin_row * n_pin_c + pin_col``.

    Parameters
    ----------
    n_assemblies : int
        Number of valid fuel assemblies (from ``CoreMap.n_assemblies``).
    n_pin_r, n_pin_c : int
        Pin-grid dimensions within each assembly.
    n_axial : int
        Number of axial fuel planes stored (reflector planes excluded).
    n_time : int
        Number of time steps.
    backend : {'memmap', 'ndarray'}
        ``'memmap'`` (default) uses ``numpy.memmap`` on a temporary file.
        ``'ndarray'`` keeps data in RAM (useful for small test cases).
    memmap_path : Path | str | None
        Explicit file path for the memmap.  If *None*, a temporary file is
        created (deleted when the store is garbage-collected).
    dtype : numpy dtype
        Floating-point precision (default ``float64``).
    """

    def __init__(
        self,
        n_assemblies: int,
        n_pin_r: int,
        n_pin_c: int,
        n_axial: int,
        n_time: int,
        backend: str = "memmap",
        memmap_path: Optional[Union[str, Path]] = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.n_asm = n_assemblies
        self.n_pin_r = n_pin_r
        self.n_pin_c = n_pin_c
        self.n_pin = n_pin_r * n_pin_c
        self.n_axial = n_axial
        self.n_time = n_time
        self.dtype = np.dtype(dtype)

        total = self.n_asm * self.n_pin * self.n_time * self.n_axial
        if total == 0:
            raise ValueError("Store size is zero — check dimensions.")

        if backend == "memmap":
            if memmap_path is None:
                self._tmpfile = tempfile.NamedTemporaryFile(
                    suffix=".dat", delete=False
                )
                memmap_path = self._tmpfile.name
            else:
                self._tmpfile = None
                memmap_path = str(memmap_path)
            self._data: np.ndarray = np.memmap(
                memmap_path, dtype=self.dtype, mode="w+", shape=(total,)
            )
            self._memmap_path = Path(memmap_path)
        elif backend == "ndarray":
            self._data = np.zeros(total, dtype=self.dtype)
            self._tmpfile = None
            self._memmap_path = None
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'memmap' or 'ndarray'.")

    # -- indexing helpers -----------------------------------------------------

    def _global_pin_id(self, asm_id: int, pin_r: int, pin_c: int) -> int:
        return asm_id * self.n_pin + pin_r * self.n_pin_c + pin_c

    def _flat_index(self, asm_id: int, pin_r: int, pin_c: int, t: int, az: int) -> int:
        gid = self._global_pin_id(asm_id, pin_r, pin_c)
        return gid * (self.n_time * self.n_axial) + t * self.n_axial + az

    # -- write ---------------------------------------------------------------

    def set_pin(
        self, asm_id: int, pin_r: int, pin_c: int, t: int, az: int, value: float
    ) -> None:
        """Set a single pin-power value."""
        self._data[self._flat_index(asm_id, pin_r, pin_c, t, az)] = value

    def set_plane(
        self, asm_id: int, t: int, az: int, pin_matrix: np.ndarray
    ) -> None:
        """Write an entire axial-plane's pin matrix for one assembly/time.

        Parameters
        ----------
        pin_matrix : ndarray, shape ``(n_pin_r, n_pin_c)``
        """
        assert pin_matrix.shape == (self.n_pin_r, self.n_pin_c), (
            f"Expected shape ({self.n_pin_r}, {self.n_pin_c}), "
            f"got {pin_matrix.shape}"
        )
        for pr in range(self.n_pin_r):
            for pc in range(self.n_pin_c):
                idx = self._flat_index(asm_id, pr, pc, t, az)
                self._data[idx] = pin_matrix[pr, pc]

    # -- read ----------------------------------------------------------------

    def get_pin_series(
        self, asm_id: int, pin_r: int, pin_c: int
    ) -> np.ndarray:
        """Return all (time, axial) data for one pin.

        Returns
        -------
        ndarray, shape ``(n_time, n_axial)``
            Contiguous view (or copy) of the pin's full history.
        """
        gid = self._global_pin_id(asm_id, pin_r, pin_c)
        start = gid * (self.n_time * self.n_axial)
        end = start + self.n_time * self.n_axial
        return self._data[start:end].reshape(self.n_time, self.n_axial).copy()

    def get_plane(self, asm_id: int, t: int, az: int) -> np.ndarray:
        """Return pin matrix for one assembly/time/axial.

        Returns
        -------
        ndarray, shape ``(n_pin_r, n_pin_c)``
        """
        out = np.empty((self.n_pin_r, self.n_pin_c), dtype=self.dtype)
        for pr in range(self.n_pin_r):
            for pc in range(self.n_pin_c):
                out[pr, pc] = self._data[self._flat_index(asm_id, pr, pc, t, az)]
        return out

    @property
    def data(self) -> np.ndarray:
        """Raw 1D backing array (memmap or ndarray)."""
        return self._data

    def __repr__(self) -> str:
        backend = "memmap" if self._memmap_path else "ndarray"
        return (
            f"PinPowerStore(asm={self.n_asm}, pin={self.n_pin_r}x{self.n_pin_c}, "
            f"ax={self.n_axial}, t={self.n_time}, backend={backend})"
        )


# ---------------------------------------------------------------------------
# PinPowerParser: parse PARCS .pin files
# ---------------------------------------------------------------------------

# Header patterns
_RE_TIME = re.compile(r"At\s+Time:\s+([\d.Ee+\-]+)")
_RE_ASSEMBLY = re.compile(
    r"Assembly\s+Coordinate\s+\(i,j\):\s+(\d+)\s+(\d+)\s*,"
    r"\s*Plane\s+Index\s+&\s+Height:\s+(\d+)\s+([\d.Ee+\-]+)\s*,"
    r"\s*Normalization\s+Factor:\s+([\d.Ee+\-]+)"
)
_RE_PEAK = re.compile(r"Peak\s+at")
_RE_COLHDR = re.compile(r"^\s+(\d+(?:\s+\d+)+)\s*$")


@dataclass
class ParserConfig:
    """Configuration for the .pin parser.

    Attributes
    ----------
    axial_bottom_to_top : bool
        If True (default), the file lists planes from bottom to top
        (plane 2 = physical bottom).  Internal storage always stores
        axial index 0 = physical bottom.  If False, the parser reverses
        the axial order during ingestion.
    skip_plane_zero : bool
        If True (default), plane index 0 (assembly-averaged) is skipped.
    fuel_plane_range : tuple[int, int]
        Inclusive range of PARCS plane indices considered fuel.
        Default ``(2, 13)`` for the standard 12-plane PWR layout.
    file_index_base : int
        PARCS assembly coordinates are 1-indexed in the file.
        This value (default 1) is subtracted to get 0-based row/col.
    """

    axial_bottom_to_top: bool = True
    skip_plane_zero: bool = True
    fuel_plane_range: Tuple[int, int] = (2, 13)
    file_index_base: int = 1


class PinPowerParser:
    """Parses PARCS ``.pin`` output files into a ``PinPowerStore``.

    Workflow::

        parser = PinPowerParser(core_map, config)
        store = parser.parse(pin_file_path)
        # or for multiple timestep files:
        store = parser.parse([file_t0, file_t1, ...])

    The parser performs two passes:
    1. **Discovery pass**: scan headers to determine n_time, n_axial,
       n_pin_r, n_pin_c, and validate assembly coordinates against the
       core map.
    2. **Data pass**: read pin matrices and populate the store.

    Parameters
    ----------
    core_map : CoreMap
        Defines valid assembly positions and 2D->1D mapping.
    config : ParserConfig | None
        Parser options.  Defaults to ``ParserConfig()``.
    backend : str
        Storage backend forwarded to ``PinPowerStore``.
    memmap_path : Path | str | None
        Explicit memmap file path (forwarded to ``PinPowerStore``).
    """

    def __init__(
        self,
        core_map: CoreMap,
        config: Optional[ParserConfig] = None,
        backend: str = "memmap",
        memmap_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.core_map = core_map
        self.config = config or ParserConfig()
        self._backend = backend
        self._memmap_path = memmap_path

    # -- public API ----------------------------------------------------------

    def parse(
        self,
        pin_files: Union[str, Path, Sequence[Union[str, Path]]],
    ) -> PinPowerStore:
        """Parse one or more ``.pin`` files and return a populated store.

        Parameters
        ----------
        pin_files : path or list of paths
            Single file with one or more ``At Time:`` blocks, **or** a
            list of files each containing one timestep (in chronological
            order).

        Returns
        -------
        PinPowerStore
        """
        if isinstance(pin_files, (str, Path)):
            pin_files = [Path(pin_files)]
        else:
            pin_files = [Path(f) for f in pin_files]

        # -- Discovery pass --------------------------------------------------
        time_values: List[float] = []
        plane_indices_seen: set = set()
        n_pin_r: Optional[int] = None
        n_pin_c: Optional[int] = None

        for fpath in pin_files:
            lines = fpath.read_text().splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                tm = _RE_TIME.search(line)
                if tm:
                    t_val = float(tm.group(1))
                    if t_val not in time_values:
                        time_values.append(t_val)
                    i += 1
                    continue

                am = _RE_ASSEMBLY.search(line)
                if am:
                    plane_idx = int(am.group(3))
                    plane_indices_seen.add(plane_idx)
                    # Read column header to get n_pin_c
                    i += 1
                    col_line = lines[i].strip()
                    cols = col_line.split()
                    if n_pin_c is None:
                        n_pin_c = len(cols)
                    # Count data rows to get n_pin_r
                    i += 1
                    row_count = 0
                    while i < len(lines) and not _RE_PEAK.search(lines[i]):
                        stripped = lines[i].strip()
                        if stripped == "":
                            i += 1
                            continue
                        row_count += 1
                        i += 1
                    if n_pin_r is None:
                        n_pin_r = row_count
                    # skip Peak line
                    i += 1
                    continue
                i += 1

        if n_pin_r is None or n_pin_c is None:
            raise ValueError("Could not determine pin dimensions from file(s).")
        if not time_values:
            raise ValueError("No 'At Time:' headers found in file(s).")

        # Determine fuel planes
        pmin, pmax = self.config.fuel_plane_range
        fuel_planes_in_file = sorted(
            p for p in plane_indices_seen if pmin <= p <= pmax
        )
        n_axial = len(fuel_planes_in_file)
        if n_axial == 0:
            raise ValueError(
                f"No fuel planes found in range [{pmin}, {pmax}]. "
                f"Planes seen: {sorted(plane_indices_seen)}"
            )

        # Map PARCS plane index -> internal axial index
        # If bottom-to-top (default), plane 2 -> az=0, plane 3 -> az=1, ...
        # If top-to-bottom, reverse.
        if self.config.axial_bottom_to_top:
            plane_to_az = {p: idx for idx, p in enumerate(fuel_planes_in_file)}
        else:
            plane_to_az = {
                p: idx
                for idx, p in enumerate(reversed(fuel_planes_in_file))
            }

        n_time = len(time_values)
        time_to_t = {tv: idx for idx, tv in enumerate(time_values)}

        # -- Create store ----------------------------------------------------
        store = PinPowerStore(
            n_assemblies=self.core_map.n_assemblies,
            n_pin_r=n_pin_r,
            n_pin_c=n_pin_c,
            n_axial=n_axial,
            n_time=n_time,
            backend=self._backend,
            memmap_path=self._memmap_path,
            dtype=np.float64,
        )

        # -- Data pass -------------------------------------------------------
        current_t: Optional[int] = None
        base = self.config.file_index_base

        for fpath in pin_files:
            lines = fpath.read_text().splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                tm = _RE_TIME.search(line)
                if tm:
                    current_t = time_to_t[float(tm.group(1))]
                    i += 1
                    continue

                am = _RE_ASSEMBLY.search(line)
                if am:
                    asm_row_file = int(am.group(1))
                    asm_col_file = int(am.group(2))
                    plane_idx = int(am.group(3))

                    # Convert to 0-based
                    asm_row = asm_row_file - base
                    asm_col = asm_col_file - base

                    # Skip column header
                    i += 1
                    i += 1  # move to first data row

                    # Read pin matrix rows
                    pin_rows: List[List[float]] = []
                    while i < len(lines) and not _RE_PEAK.search(lines[i]):
                        stripped = lines[i].strip()
                        if stripped == "":
                            i += 1
                            continue
                        # row format: "     NN val val val ..."
                        parts = stripped.split()
                        # parts[0] = row number (1-indexed), rest = values
                        values = [float(v) for v in parts[1:]]
                        pin_rows.append(values)
                        i += 1

                    # Skip peak line
                    i += 1

                    # Store if this is a fuel plane and valid assembly
                    if plane_idx not in plane_to_az:
                        continue
                    if not self.core_map.is_valid(asm_row, asm_col):
                        continue

                    az = plane_to_az[plane_idx]
                    aid = self.core_map.asm_id(asm_row, asm_col)
                    pin_matrix = np.array(pin_rows, dtype=np.float64)
                    store.set_plane(aid, current_t, az, pin_matrix)
                    continue

                i += 1

        # Store metadata
        store.time_values = np.array(time_values)
        store.fuel_plane_indices = fuel_planes_in_file
        store.plane_to_az = plane_to_az

        return store

    def __repr__(self) -> str:
        return f"PinPowerParser(core_map={self.core_map!r})"


# ---------------------------------------------------------------------------
# PinPowerAnalyzer: temporal deltas & ranking
# ---------------------------------------------------------------------------

@dataclass
class PinDeltaResult:
    """Result for a single pin's temporal-delta analysis."""

    asm_row: int
    asm_col: int
    pin_row: int
    pin_col: int
    max_abs_delta: float
    time_of_max: int  # time index where max delta occurs
    axial_of_max: int  # axial index where max delta occurs


class PinPowerAnalyzer:
    """Temporal-delta computation and pin ranking.

    Parameters
    ----------
    store : PinPowerStore
        Populated pin-power data store.
    core_map : CoreMap
        Core map for coordinate translation.
    fill_t0 : {'zero', 'nan'}
        How to fill ``dP[t=0, :]``.  Default ``'zero'``.
    """

    def __init__(
        self,
        store: PinPowerStore,
        core_map: CoreMap,
        fill_t0: str = "zero",
    ) -> None:
        self.store = store
        self.core_map = core_map
        self.fill_t0 = fill_t0

    def compute_delta(
        self,
        asm_row: int,
        asm_col: int,
        pin_row: int,
        pin_col: int,
    ) -> np.ndarray:
        """Compute temporal power delta for a single pin.

        ``dP[t, z] = P[t, z] - P[t-1, z]``   (t >= 1)
        ``dP[0, z] = 0`` or ``NaN`` depending on ``fill_t0``.

        Returns
        -------
        ndarray, shape ``(Nt, Naz)``
        """
        aid = self.core_map.asm_id(asm_row, asm_col)
        series = self.store.get_pin_series(aid, pin_row, pin_col)  # (Nt, Naz)
        delta = np.empty_like(series)
        if self.fill_t0 == "nan":
            delta[0, :] = np.nan
        else:
            delta[0, :] = 0.0
        delta[1:, :] = series[1:, :] - series[:-1, :]
        return delta

    def rank_max_delta(
        self, top_n: Optional[int] = None
    ) -> List[PinDeltaResult]:
        """Rank all pins by maximum absolute axial power change.

        Iterates over every valid pin in the core, computes the temporal
        delta, and returns a sorted list (descending by ``max_abs_delta``).

        Parameters
        ----------
        top_n : int | None
            If given, return only the top *N* pins.

        Returns
        -------
        list[PinDeltaResult]
        """
        results: List[PinDeltaResult] = []

        for aid, (r, c) in enumerate(self.core_map.valid_positions()):
            for pr in range(self.store.n_pin_r):
                for pc in range(self.store.n_pin_c):
                    series = self.store.get_pin_series(aid, pr, pc)
                    # delta for t>=1
                    if self.store.n_time < 2:
                        # Only one timestep — no meaningful delta
                        continue
                    delta = series[1:, :] - series[:-1, :]
                    abs_delta = np.abs(delta)
                    max_idx = np.unravel_index(
                        np.argmax(abs_delta), abs_delta.shape
                    )
                    max_val = abs_delta[max_idx]
                    results.append(
                        PinDeltaResult(
                            asm_row=r,
                            asm_col=c,
                            pin_row=pr,
                            pin_col=pc,
                            max_abs_delta=float(max_val),
                            time_of_max=int(max_idx[0]) + 1,  # +1 because delta starts at t=1
                            axial_of_max=int(max_idx[1]),
                        )
                    )

        results.sort(key=lambda x: x.max_abs_delta, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    def __repr__(self) -> str:
        return f"PinPowerAnalyzer(store={self.store!r})"


# ---------------------------------------------------------------------------
# Convenience: get_pin_series with (row, col, pin_r, pin_c) interface
# ---------------------------------------------------------------------------

def get_pin_series(
    store: PinPowerStore,
    core_map: CoreMap,
    assembly_row: int,
    assembly_col: int,
    pin_row: int,
    pin_col: int,
) -> np.ndarray:
    """Convenience function: get pin power series by core-map coordinates.

    Parameters
    ----------
    assembly_row, assembly_col : int
        Core-map 2D indices (0-based, origin = top-left).
    pin_row, pin_col : int
        Pin indices within the assembly (0-based, origin = top-left).

    Returns
    -------
    ndarray, shape ``(Nt, Naz)``
    """
    aid = core_map.asm_id(assembly_row, assembly_col)
    return store.get_pin_series(aid, pin_row, pin_col)


# ---------------------------------------------------------------------------
# Convenience: build CoreMap from PARCS input lines
# ---------------------------------------------------------------------------

def coremap_from_parcs_input(
    input_path: Union[str, Path],
    fuel_ids: Optional[set] = None,
    pincal_section: bool = True,
) -> CoreMap:
    """Build a ``CoreMap`` from a PARCS ``.inp`` file.

    Reads either the ``pincal_loc`` section (if present and *pincal_section*
    is True) or the ``Rad_Conf`` section.  Nonzero entries in ``pincal_loc``
    are treated as pin-calculation-enabled; in ``Rad_Conf`` mode, *fuel_ids*
    determines which types are fuel.

    The boundary conditions are inspected to infer symmetry.

    Parameters
    ----------
    input_path : path
        PARCS ``.inp`` file.
    fuel_ids : set[int] | None
        Fuel assembly-type IDs.  Ignored when using ``pincal_loc`` (which
        already encodes valid positions).
    pincal_section : bool
        If True (default) and a ``pincal_loc`` section exists, use it.

    Returns
    -------
    CoreMap
    """
    text = Path(input_path).read_text()
    lines = text.splitlines()

    # Parse geo_dim for grid size
    geo_match = re.search(r"geo_dim\s+(\d+)\s+(\d+)", text)
    if not geo_match:
        raise ValueError("Cannot find 'geo_dim' in PARCS input.")
    nr, nc = int(geo_match.group(1)), int(geo_match.group(2))

    # Parse symmetry from Boun_cond
    boun_match = re.search(r"Boun_cond\s+([\d\s]+)", text)
    symmetry = "full"
    if boun_match:
        bvals = [int(x) for x in boun_match.group(1).split()]
        # 2 = reflective.  If both x-boundaries OR both y-boundaries are
        # reflective, it's quarter symmetry.
        n_reflective = sum(1 for b in bvals if b == 2)
        if n_reflective >= 4:
            symmetry = "quarter"
        if n_reflective >= 5:
            symmetry = "octant"

    # Try to find pincal_loc section
    layout = None
    if pincal_section:
        for idx, ln in enumerate(lines):
            if "pincal_loc" in ln.lower():
                layout = _read_integer_block(lines, idx + 1, nr)
                break

    # Fallback to Rad_Conf
    if layout is None:
        for idx, ln in enumerate(lines):
            if "Rad_Conf" in ln:
                layout = _read_integer_block(lines, idx + 1, nr)
                break

    if layout is None:
        raise ValueError("Cannot find 'pincal_loc' or 'Rad_Conf' in PARCS input.")

    # For pincal_loc, nonzero = valid; fuel_ids not needed
    if pincal_section:
        # Every nonzero entry in pincal_loc is a valid pin-power location
        return CoreMap(layout, fuel_ids=None, symmetry=symmetry)
    else:
        return CoreMap(layout, fuel_ids=fuel_ids, symmetry=symmetry)


def _read_integer_block(lines: List[str], start: int, n_rows: int) -> np.ndarray:
    """Read *n_rows* of whitespace-separated integers starting at *start*."""
    rows = []
    i = start
    while len(rows) < n_rows and i < len(lines):
        stripped = lines[i].strip()
        if stripped == "" or stripped.startswith("!"):
            i += 1
            continue
        # Stop if we hit a keyword (non-numeric first token)
        tokens = stripped.split()
        try:
            int(tokens[0])
        except ValueError:
            break
        row = [int(t) for t in tokens]
        rows.append(row)
        i += 1

    if len(rows) < n_rows:
        raise ValueError(
            f"Expected {n_rows} rows of integers, got {len(rows)}."
        )

    # Pad rows to equal length (trailing zeros for triangular layouts)
    max_c = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_c:
            r.append(0)

    return np.array(rows, dtype=int)
