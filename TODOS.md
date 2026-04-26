# TODOS

## 1. PARCS OpenMP 벤치마크 실행 및 최적 M 값 결정

**What:** benchmark_parcs_parallel.slurm 실행하여 OMP_NUM_THREADS=1,2,4,8,16,32,64의 PARCS wall-clock time 측정.

**Why:** --cpus-per-task=M 값이 PARCS 실행 속도를 결정. M이 너무 크면 OpenMP 스케일링 포화로 CPU 낭비, M이 너무 작으면 PARCS가 느림. 최적 M을 찾아야 debug_run.slurm의 PARCS_CPUS_PER_TASK를 확정할 수 있음.

**Depends on:** /scratch/leec3/PARCS_GIFT_teton/benchmark_parcs_parallel.slurm (이미 작성됨), prev_optimization의 PARCS 테스트 케이스.

**How to start:** `cd /scratch/leec3/PARCS_GIFT_teton && bash benchmark_parcs_parallel.slurm`

## 2. fuel_runloop.py의 miscOutParams.csv 쓰기 실패 에러 핸들링

**What:** fuel_runloop.py에서 fuel_integrity_metric을 miscOutParams.csv에 기록할 때 try/except로 보호. 실패 시 0.0 (중립값) fallback 기록하여 GA 진행이 중단되지 않도록 함.

**Why:** getMiscParams()에서 fuel_integrity_metric 키가 없으면 KeyError → realization 전체 실패 → GA 중단. 파일 쓰기는 디스크 full, NFS 장애 등으로 실패할 수 있음.

**Depends on:** eq_fuel 구현 (TODO 아님, 현재 세션에서 구현 예정).

**How to start:** fuel_runloop.py의 CSV 쓰기 섹션에 try/except 추가, except 시 fallback CSV 기록.
