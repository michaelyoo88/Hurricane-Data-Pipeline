"""Part 1 only ("any recorded property damage?"): logistic regression vs random forest vs XGBoost.

Same held-out design as the sandbox: 3 repeats of 5-fold over all 194 storms (fold_all_r*), whole storms held out.
Settings fixed in advance (not tuned). Two feature sets:
  J2     the 4 inputs of the chosen model: capped wind, coastal x wind, distance, coastal
  rich   10 raw inputs trees can combine on their own: wind, wind-hours, distance, coastal, density,
         right of track, forward speed, 24h intensity change, storm peak wind, storm footprint
Scores: ROC-AUC and PR-AUC (ranking), Brier score and log loss (are the probabilities right?),
calibration = mean predicted probability / actual damage rate (1 = right on average), and ROC-AUC among
strong-wind rows (25+ m/s), where the 4-input model was weakest.
Out: results/classifier_compare.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
SEED = 42
df = pd.read_csv(HERE / "data" / "table.csv", dtype={"fips": str})
df["wind_cap"] = df.wind_ms.clip(upper=50)
df["coastal_x_wind_cap"] = df.coastal * (df.wind_ms - 17.5).clip(0, 32.5)
J2 = ["wind_cap", "coastal_x_wind_cap", "log_dist_km", "coastal"]
RICH = ["wind_ms", "log_wind_exposure", "log_dist_km", "coastal", "log_density", "right_of_track",
        "log_fwd_speed", "dvmax_24h", "storm_peak_wind", "log_footprint"]
y = (df.prop_usd > 0).to_numpy().astype(int)
strong = (df.wind_ms >= 25).to_numpy()


def logistic(X, yy):
    m, s = X.mean(0), X.std(0)
    model = LogisticRegression(C=1.0, max_iter=2000).fit((X - m) / s, yy)
    return lambda Z: model.predict_proba((Z - m) / s)[:, 1]


def forest(X, yy):
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt", n_jobs=2, random_state=SEED).fit(X, yy)
    return lambda Z: model.predict_proba(Z)[:, 1]


def boost(X, yy):
    model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=10, n_jobs=2, random_state=SEED, eval_metric="logloss").fit(X, yy)
    return lambda Z: model.predict_proba(Z)[:, 1]


MODELS = {"Logistic regression, J2 (current)": (logistic, J2), "Logistic regression, rich": (logistic, RICH),
          "Random forest, J2": (forest, J2), "Random forest, rich": (forest, RICH),
          "XGBoost, J2": (boost, J2), "XGBoost, rich": (boost, RICH)}
rows = []
for name, (make, feats) in MODELS.items():
    X = df[feats].to_numpy(float)
    runs = []
    for r in range(3):
        fold, prob = df[f"fold_all_r{r}"].to_numpy(), np.zeros(len(df))
        for k in range(5):
            fit, held = fold != k, fold == k
            prob[held] = make(X[fit], y[fit])(X[held])
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        runs.append({"roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob),
                     "brier": brier_score_loss(y, prob), "log_loss": log_loss(y, prob),
                     "calibration": prob.mean() / y.mean(), "roc_auc_strong_wind": roc_auc_score(y[strong], prob[strong]),
                     "calib_strong_wind": prob[strong].mean() / y[strong].mean()})
    rows.append({"model": name, **pd.DataFrame(runs).mean().to_dict()})
    print(f"  done {name}", flush=True)
out = pd.DataFrame(rows)
out.round(4).to_csv(HERE / "results" / "classifier_compare.csv", index=False)
pd.set_option("display.width", 250)
print(f"\n{len(y):,} rows, damage rate {y.mean():.1%}; strong-wind rows {strong.sum():,}, damage rate {y[strong].mean():.1%}")
print(out.set_index("model").round(3).to_string())
