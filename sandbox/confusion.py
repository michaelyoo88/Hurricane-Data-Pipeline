"""Confusion matrix of the logistic part ("any property damage?") of the 4-input model (J2).
Held-out: fresh 5-fold over all 194 storms (fold_all_r0), each storm predicted by a model that never saw it.
Shown at three cutoffs: 0.5 (the default), the cutoff that flags as many rows as really had damage, and the
cutoff with the best F1. Out: results/confusion_matrix.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lib import J2, fit_logistic, load, roc_auc

HERE = Path(__file__).resolve().parent
df = load()
y = (df.prop_usd > 0).to_numpy()
X = df[J2].to_numpy(float)
fold = df.fold_all_r0.to_numpy()
prob = np.zeros(len(df))
for k in range(5):
    fit, held = fold != k, fold == k
    m, s = X[fit].mean(0), X[fit].std(0)
    X1 = np.column_stack([np.ones(len(X)), (X - m) / np.where(s > 0, s, 1)])
    b = fit_logistic(X1[fit], y[fit].astype(float), np.r_[0.0, np.ones(len(J2))])
    prob[held] = 1 / (1 + np.exp(-np.clip(X1[held] @ b, -30, 30)))


def matrix(cut):
    p = prob >= cut
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    return {"cutoff": cut, "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
            "accuracy": (tp + tn) / len(y), "precision": tp / max(tp + fp, 1), "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


cuts = np.unique(np.quantile(prob, np.linspace(0.5, 0.999, 400)))
best_f1 = max(cuts, key=lambda c: matrix(c)["f1"])
same_count = np.sort(prob)[::-1][y.sum() - 1]
out = pd.DataFrame([matrix(0.5) | {"rule": "default 0.5"}, matrix(same_count) | {"rule": "flag as many as really damaged"},
                    matrix(best_f1) | {"rule": "best F1"}])
out.round(4).to_csv(HERE / "results" / "confusion_matrix.csv", index=False)
print(f"{len(y):,} storm x county rows held out; {y.sum():,} had recorded damage ({y.mean():.1%}); ROC-AUC of the probability {roc_auc(y, prob):.3f}")
for _, r in out.iterrows():
    print(f"\n{r.rule} (cutoff {r.cutoff:.3f}):")
    print(f"                    predicted damage   predicted none")
    print(f"  actual damage     {int(r.true_pos):>15,}   {int(r.false_neg):>14,}")
    print(f"  actual none       {int(r.false_pos):>15,}   {int(r.true_neg):>14,}")
    print(f"  accuracy {r.accuracy:.1%}  precision {r.precision:.1%}  recall {r.recall:.1%}  specificity {r.specificity:.1%}  F1 {r.f1:.2f}")
