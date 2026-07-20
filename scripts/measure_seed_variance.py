"""Measure seed variance of the full training protocol.

Runs :func:`run_training.run_protocol` — the exact pipeline that produces the
shipped artefacts, candidate selection included — across N seeds and records
the spread of the headline metrics next to the naive baseline. Writes
``reports/seed_variance.json``, which the README/MODEL_CARD tables are gated
against.

    python scripts/measure_seed_variance.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_training import prepare_data, run_protocol  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "seed_variance.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    df_clean = prepare_data()
    per_seed = []
    for seed in range(args.seeds):
        result = run_protocol(df_clean, seed=seed)
        row = {
            "seed": seed,
            "selected_model": result["reg_record"]["selected_model"],
            "test_r2": round(result["reg_record"]["metrics"]["r2"], 4),
            "zones_macro_f1": round(result["clf_record"]["metrics"]["macro_f1"], 4),
            "baseline_test_r2": result["baseline"]["test_r2"],
            "baseline_zones_macro_f1": result["baseline"]["test_zones_macro_f1"],
        }
        per_seed.append(row)
        print(f"[{seed + 1:2}/{args.seeds}] {row}", flush=True)

    def agg(key: str) -> dict[str, float]:
        values = [r[key] for r in per_seed]
        return {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    record = {
        "n_seeds": args.seeds,
        "protocol": (
            "run_training.run_protocol per seed: split (price-qcut stratified), "
            "train-only cap bounds / zone bins / category vocabulary, candidate "
            "selection on val, single test read"
        ),
        "test_r2": agg("test_r2"),
        "zones_macro_f1": agg("zones_macro_f1"),
        "baseline_test_r2": agg("baseline_test_r2"),
        "baseline_zones_macro_f1": agg("baseline_zones_macro_f1"),
        "selected_model_counts": {
            name: sum(1 for r in per_seed if r["selected_model"] == name)
            for name in sorted({r["selected_model"] for r in per_seed})
        },
        "per_seed": per_seed,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"test R2      {record['test_r2']}")
    print(f"zones F1     {record['zones_macro_f1']}")
    print(f"selection    {record['selected_model_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
