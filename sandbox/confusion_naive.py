"""Confusion matrices: naive one-class guesses vs the best model's "any damage?" step.

Rows: storm x county, 2000-2024 (every row had a storm within 500 km); positive = recorded property damage.
Best model = the logistic step of the averaged model (J2 + latitude: capped wind, coastal x wind, distance,
coastal, latitude), held out: 5-fold over all 194 storms (fold_all_r0), whole storms held out.
Naive models: always "no damage" (the majority class) and always "damage".
Out: results/confusion_naive_vs_best.csv, results/confusion_naive_vs_best.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib import J2, fit_logistic, load, roc_auc

HERE = Path(__file__).resolve().parent
FEATS = J2 + ["pop_lat"]
df = load()
y = (df.prop_usd > 0).to_numpy()
X = df[FEATS].to_numpy(float)
fold = df.fold_all_r0.to_numpy()
prob = np.zeros(len(df))
for k in range(5):
    fit, held = fold != k, fold == k
    m, s = X[fit].mean(0), X[fit].std(0)                     # scale with the training storms only
    X1 = np.column_stack([np.ones(len(X)), (X - m) / np.where(s > 0, s, 1)])
    b = fit_logistic(X1[fit], y[fit].astype(float), np.r_[0.0, np.ones(len(FEATS))])
    prob[held] = 1 / (1 + np.exp(-np.clip(X1[held] @ b, -30, 30)))


def scores(name, pred):
    tp, fp, fn, tn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum(), (~pred & ~y).sum()
    return {"model": name, "true_pos": tp, "false_neg": fn, "false_pos": fp, "true_neg": tn,
            "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn), "specificity": tn / (tn + fp), "f1": 2 * tp / (2 * tp + fp + fn),
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2}


cuts = np.unique(np.quantile(prob, np.linspace(0.5, 0.999, 400)))
best_cut = max(cuts, key=lambda c: scores("", prob >= c)["f1"])
rows = [scores("Naive: always 'no damage'", np.zeros(len(y), bool)),
        scores("Naive: always 'damage'", np.ones(len(y), bool)),
        scores("Best model, cutoff 0.50", prob >= 0.5),
        scores(f"Best model, cutoff {best_cut:.2f} (best F1)", prob >= best_cut)]
out = pd.DataFrame(rows)
out.round(4).to_csv(HERE / "results" / "confusion_naive_vs_best.csv", index=False)

# ---- chart: majority-class naive vs best model at two cutoffs ----
show = [rows[0], rows[2], rows[3]]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
for ax, r in zip(axes, show):
    cm = np.array([[r["true_pos"], r["false_neg"]], [r["false_pos"], r["true_neg"]]])
    share = cm / cm.sum(axis=1, keepdims=True)               # colour = share of each actual class
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({share[i, j]:.0%} of row)", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1], ["Predicted damage", "Predicted none"], fontsize=8)
    ax.set_yticks([0, 1], ["Actual damage", "Actual none"], fontsize=8)
    ax.set_title(f"{r['model']}\naccuracy {r['accuracy']:.1%} · recall {r['recall']:.0%} · F1 {r['f1']:.2f}", fontsize=9)
fig.suptitle(f"Any recorded hurricane damage, storm x county 2000-2024 (held-out storms; {y.sum():,} of {len(y):,} rows damaged)", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "results" / "confusion_naive_vs_best.png", dpi=150)

pd.set_option("display.width", 250)
print(f"{len(y):,} rows, {y.sum():,} with damage ({y.mean():.1%}); best model ROC-AUC {roc_auc(y, prob):.3f}")
print(out.round(3).to_string(index=False))
