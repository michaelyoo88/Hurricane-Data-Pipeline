"""Challenger: LightGBM (gradient-boosted trees) on the same splits and metrics as the GLMs.

    python3 sandbox/lgbm_challenger.py dev     # 3 x 5-fold on the dev storms -> results/lgbm_experiments.csv
    python3 sandbox/lgbm_challenger.py test    # fit on all dev storms, predict the locked test storms
Needs lightgbm (pip install lightgbm). Settings are fixed in advance, not tuned:
Tweedie loss (power 1.5) on the damage ratio, rows weighted by building value, small trees
(7 leaves, at least 50 rows per leaf), 300 trees, learning rate 0.05, SEED 42.
"Monotone" = the model is forced to never predict LESS damage for more wind or wind-hours,
and never MORE damage for more distance.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from lib import BASE, STORM, TRACK, evaluate, load

HERE = Path(__file__).resolve().parent
SEED = 42
RAW = ["wind_ms", "log_wind_exposure", "log_dist_km", "coastal", "log_density"]       # trees need no hinge / interaction columns
DIRECTION = {"wind_ms": 1, "log_wind_exposure": 1, "log_dist_km": -1, "storm_peak_wind": 1}
MODELS = {"H1 LightGBM monotone": (RAW, True),
          "H2 LightGBM monotone + storm + track": (RAW + STORM + ["right_of_track", "log_fwd_speed", "dvmax_24h"], True),
          "H3 LightGBM free (no constraints)": (RAW, False)}


def fit_predict(df, fit, features, monotone):
    e = df.bldg_value.to_numpy()
    extra = {"monotone_constraints": [DIRECTION.get(f, 0) for f in features]} if monotone else {}
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.05, n_estimators=300,
                          num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=SEED, verbose=-1, n_jobs=2, **extra)
    m.fit(df.loc[fit, features], (df.prop_usd / df.bldg_value)[fit], sample_weight=e[fit] / e[fit].mean())
    return np.minimum(m.predict(df[features]), 1.0), m


if __name__ == "__main__":
    mode = sys.argv[1]
    df = load()
    dev, e, y = (df.set == "dev").to_numpy(), df.bldg_value.to_numpy(), df.prop_usd.to_numpy()
    if mode == "dev":
        rows, keep = [], df.loc[dev, ["storm_id", "fips"]].copy()
        for label, (features, monotone) in MODELS.items():
            runs = []
            for r in range(3):
                fold = df[f"fold_r{r}"].to_numpy()
                pred, null = np.zeros(len(df)), np.zeros(len(df))
                for k in range(5):
                    held, fit = dev & (fold == k), dev & (fold != k)
                    pred[held] = (fit_predict(df, fit, features, monotone)[0] * e)[held]
                    null[held] = y[fit].sum() / e[fit].sum() * e[held]
                runs.append(evaluate(df.loc[dev, ["storm_id", "storm_strength", "fips", "bldg_value"]]
                                     .assign(pred=pred[dev], rec=y[dev], null=null[dev])))
                keep[f"{label}|r{r}"] = pred[dev]
            runs = pd.DataFrame(runs)
            rows.append({"round": "lgbm", "model": label, **runs.mean().to_dict(), "d2_min": runs.d2.min(), "d2_max": runs.d2.max()})
        out = pd.DataFrame(rows)
        out.round(4).to_csv(HERE / "results" / "lgbm_experiments.csv", index=False)
        keep.to_csv(HERE / "results" / "lgbm_oof.csv", index=False)
        pd.set_option("display.width", 250)
        print(out.drop(columns="round").set_index("model").round(2).to_string())
    else:
        out = df.loc[~dev, ["storm_id", "fips"]].copy()
        for label, (features, monotone) in MODELS.items():
            ratio, model = fit_predict(df, dev, features, monotone)
            out[label] = (ratio * e)[~dev]
            if label.startswith("H1"):
                gain = model.booster_.feature_importance(importance_type="gain")
                print("H1 feature importance (share of gain):", dict(zip(features, (gain / gain.sum()).round(3))))
        out.to_csv(HERE / "results" / "lgbm_test_pred.csv", index=False)
        print(f"wrote results/lgbm_test_pred.csv ({len(out):,} locked-test rows)")
