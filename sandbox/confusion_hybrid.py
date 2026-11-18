"""Confusion matrix of the step-1 classifier ("any property damage?"): current logistic (J2 + latitude) vs the
XGBoost classifier on the richer inputs used by the round-8 hybrid (Y4). Held-out: fold_all_r0 over all 194 storms.
Cutoff = the one that flags as many county-storms as really had damage (precision = recall at that point).
Out: results/confusion_hybrid.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from round8 import LAT, RICH, chance_glm, chance_tree, prep

HERE = Path(__file__).resolve().parent
df = prep()
y = (df.prop_usd > 0).to_numpy()
fold = df.fold_all_r0.to_numpy()
probs = {"Logistic, 4 inputs + latitude (current)": np.zeros(len(df)), "XGBoost, 11 inputs (hybrid)": np.zeros(len(df))}
for k in range(5):
    fit, held = fold != k, fold == k
    probs["Logistic, 4 inputs + latitude (current)"][held] = chance_glm(df, fit, LAT)[held]
    probs["XGBoost, 11 inputs (hybrid)"][held] = chance_tree(df, fit, "xgb", RICH)[held]
rows = []
for name, p in probs.items():
    for rule in ("default 0.5", "flag as many as really damaged"):
        cut = 0.5 if rule.startswith("default") else np.sort(p)[::-1][y.sum() - 1]
        f = p >= cut
        tp, fp, fn, tn = (f & y).sum(), (f & ~y).sum(), (~f & y).sum(), (~f & ~y).sum()
        rows.append({"model": name, "rule": rule, "cutoff": cut, "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
                     "precision": tp / (tp + fp), "recall": tp / (tp + fn), "accuracy": (tp + tn) / len(y),
                     "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)})
out = pd.DataFrame(rows)
out.round(4).to_csv(HERE / "results" / "confusion_hybrid.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
