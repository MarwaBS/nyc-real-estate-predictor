"""Measure the IQR cap-factor trade-off cited in fit_cap_bounds' docstring.

For each candidate factor: fit bounds on the TRAIN split, cap, train the
Random Forest candidate, and score val on a COMMON evaluation support
(rows under the tightest cap's ceiling) so every variant faces an identical
target distribution — scored on all rows, the uncapped variant "wins" purely
through variance inflation from a single $195M listing.

    python scripts/measure_cap_factor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_training import REFERENCE_POINTS, get_feature_df, prepare_data  # noqa: E402
from src.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE  # noqa: E402
from src.data.cleaner import apply_cap, fit_cap_bounds  # noqa: E402
from src.data.features import (  # noqa: E402
    add_numeric_features,
    apply_top_categories,
    fit_top_categories,
)
from src.models.pipelines import build_regression_pipeline  # noqa: E402
from src.utils.geo import add_distance_features  # noqa: E402

FACTORS: list[float | None] = [1.5, 3.0, 5.0, None]


def main() -> int:
    df_clean = prepare_data().reset_index(drop=True)
    strat_key = pd.qcut(df_clean["PRICE"], 4, labels=False, duplicates="drop")
    idx_trainval, _ = train_test_split(
        df_clean.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=strat_key,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=strat_key[idx_trainval],
    )

    # Common support: everything under the tightest candidate's PRICE ceiling.
    tight_hi = fit_cap_bounds(df_clean.loc[idx_train], factor=1.5)["PRICE"][1]
    rows = []
    for factor in FACTORS:
        df = df_clean.copy()
        if factor is not None:
            bounds = fit_cap_bounds(df.loc[idx_train], factor=factor)
            df = apply_cap(df, bounds)
            at_cap = int((df.loc[idx_train, "PRICE"] == bounds["PRICE"][1]).sum())
        else:
            at_cap = 0
        df = add_numeric_features(df)
        df = add_distance_features(df, REFERENCE_POINTS)
        df["LOG_PRICE"] = np.log1p(df["PRICE"])
        top = fit_top_categories(
            df.loc[idx_train], columns=["SUBLOCALITY", "TYPE", "ZIPCODE"]
        )
        df = apply_top_categories(df, top)
        for col in ("BOROUGH", "TYPE", "ZIPCODE", "SUBLOCALITY"):
            df[col] = df[col].astype(str).str.lower().str.strip()

        features = get_feature_df(df)
        model = build_regression_pipeline(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=10,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        )
        model.fit(features.loc[idx_train], df.loc[idx_train, "LOG_PRICE"])

        common = [i for i in idx_val if df_clean.loc[i, "PRICE"] < tight_hi]
        y_true = df.loc[common, "LOG_PRICE"]
        y_pred = model.predict(features.loc[common])
        rows.append(
            {
                "factor": factor if factor is not None else "none",
                "val_r2_common": round(float(r2_score(y_true, y_pred)), 4),
                "val_mae_common": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "train_rows_at_cap": at_cap,
                "pct_at_cap": round(100 * at_cap / len(idx_train), 2),
            }
        )
        print(rows[-1], flush=True)

    study = {
        "model": "random_forest",
        "shipped_factor": 3.0,
        "train_fit_price_bounds": [
            round(float(b), 2) for b in fit_cap_bounds(df_clean.loc[idx_train])["PRICE"]
        ],
        "common_support_ceiling": round(float(tight_hi), 2),
        "n_train": len(idx_train),
        "rows": rows,
    }
    out = Path(__file__).resolve().parents[1] / "reports" / "cap_factor_study.json"
    out.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    print("\nfactor   val R2 (common)   val MAE (common)   train rows at cap")
    for r in rows:
        print(
            f"{str(r['factor']):>6}   {r['val_r2_common']:^15}   "
            f"{r['val_mae_common']:^16}   {r['train_rows_at_cap']}  "
            f"({r['pct_at_cap']}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
