"""
STEP 1 of the ML pipeline: synthetic dataset generation.

Runs many simulated missions through the twin-pair (expected vs. sensed)
physics simulator, some nominal, some with an injected fault ramping up
over the mission, and writes one flat labeled CSV. This is the training
data for every model in Phase 2 — there is no real telemetry involved;
ground truth is exact because we injected the fault ourselves.

Usage:
    python3 generate_dataset.py
Output:
    data/synthetic_dataset.csv
"""
import random
import sys
import time

sys.path.insert(0, ".")

import pandas as pd

from app.ml.mission_profile import generate_mission
from app.ml.fault_injection import FaultScenario, run_twin_pair, FAULT_TYPES, DRIFT_CHANNELS
from app.ml.dataset_builder import build_rows

MISSION_DURATION_S = 4800   # 80 min simulated sortie
DT_S = 2.0
N_NOMINAL_RUNS = 24
N_RUNS_PER_FAULT = 20       # fault types excluding "none"

OUT_PATH = "data/synthetic_dataset.csv"


def make_scenario(fault_type: str, run_idx: int) -> FaultScenario:
    # NOTE: previously seeded with Python's built-in hash(fault_type), which
    # is randomized per-process (PYTHONHASHSEED) unless disabled — that
    # silently made every run of this script non-reproducible (caught by
    # comparing per-fault-type failure counts across two otherwise-identical
    # runs and seeing them change). FAULT_TYPES.index() is stable.
    fault_salt = FAULT_TYPES.index(fault_type)
    rng = random.Random(run_idx * 7919 + fault_salt * 104729)
    if fault_type == "none":
        return FaultScenario(fault_type="none", rng_seed=run_idx)
    onset = rng.uniform(0.15, 0.45) * MISSION_DURATION_S
    ramp = rng.uniform(0.25, 0.65) * MISSION_DURATION_S
    max_sev = rng.choice([1.0, 1.0, 1.0, 0.7, 0.55])  # mostly run-to-failure, some near-miss
    return FaultScenario(
        fault_type=fault_type,
        onset_s=onset,
        ramp_s=ramp,
        max_severity=max_sev,
        rng_seed=run_idx,
        drift_channel=rng.choice(DRIFT_CHANNELS),
        drift_sign=rng.choice([-1.0, 1.0]),
    )


def main():
    t0 = time.time()
    all_rows = []
    run_id = 0
    summary = []

    for i in range(N_NOMINAL_RUNS):
        mission = generate_mission(MISSION_DURATION_S, dt_s=DT_S, seed=run_id)
        scenario = make_scenario("none", run_id)
        records = run_twin_pair(mission, scenario)
        rows, t_fail = build_rows(records, run_id=run_id)
        all_rows.extend(rows)
        summary.append(("none", run_id, t_fail))
        run_id += 1

    for fault_type in FAULT_TYPES:
        if fault_type == "none":
            continue
        for i in range(N_RUNS_PER_FAULT):
            mission = generate_mission(MISSION_DURATION_S, dt_s=DT_S, seed=run_id)
            scenario = make_scenario(fault_type, run_id)
            records = run_twin_pair(mission, scenario)
            rows, t_fail = build_rows(records, run_id=run_id)
            all_rows.extend(rows)
            summary.append((fault_type, run_id, t_fail))
            run_id += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - t0
    print(f"Generated {len(df):,} rows from {run_id} runs in {elapsed:.1f}s -> {OUT_PATH}")
    print(f"\nClass balance (fault_type):")
    print(df["fault_type"].value_counts())
    print(f"\nHealth status balance:")
    print(df["health_status"].value_counts())
    reached = sum(1 for _, _, t in summary if t is not None)
    print(f"\nRuns that reached a failure threshold: {reached}/{run_id}")
    print("\nPer-fault-type: runs reaching failure / total, mean time-to-failure (s):")
    by_fault = {}
    for ft, rid, t_fail in summary:
        by_fault.setdefault(ft, []).append(t_fail)
    for ft, vals in by_fault.items():
        reached_vals = [v for v in vals if v is not None]
        mean_t = sum(reached_vals) / len(reached_vals) if reached_vals else float("nan")
        print(f"  {ft:24s} {len(reached_vals):2d}/{len(vals):2d}   mean_t_failure={mean_t:8.1f}s")


if __name__ == "__main__":
    main()
