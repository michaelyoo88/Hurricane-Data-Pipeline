"""Confusion matrices: "hurricane damage" vs "no damage", naive guess vs the best model, at four views.

Model = the probability step of the best model (logistic regression on 4 inputs + latitude), held out:
fresh 5-fold over all 194 storms (fold_all_r0), every storm predicted by a model that never saw it.
Actual = recorded property damage (NY Fed / NOAA), 2000-2024.

Views
  1 storm x county, all rows within 500 km of a storm         (97,179 rows)
  2 storm x county, hurricane-relevant: tropical-storm wind (17.5+ m/s) at the county
  3 county, all 3,109 counties: any recorded damage in 2000-2024?
    predicted chance = 1 - product over storms of (1 - chance per storm); 0 if no storm came within 500 km
  4 county, hurricane-relevant: counties that felt tropical-storm wind at least once in 2000-2024
Naive = always guess the more common class in that view (the class it guesses is printed).
Cutoff 0.5 (default). The balanced cutoff (flags as many as really had damage) is in the CSV too.
Out: results/confusion_views.csv, results/confusion_views.png
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

# ---- held-out chance of damage per storm x county ----
df = load()
X = df[FEATS].to_numpy(float)
fold = df.fold_all_r0.to_numpy()
chance = np.zeros(len(df))
for k in range(5):
    fit, held = fold != k, fold == k
    m, s = X[fit].mean(0), X[fit].std(0)
    X1 = np.column_stack([np.ones(len(X)), (X - m) / np.where(s > 0, s, 1)])
    b = fit_logistic(X1[fit], (df.prop_usd[fit] > 0).to_numpy(float), np.r_[0.0, np.ones(len(FEATS))])
    chance[held] = 1 / (1 + np.exp(-np.clip(X1[held] @ b, -30, 30)))
df["chance"], df["damaged"], df["ts_wind"] = chance, df.prop_usd > 0, df.wind_ms >= 17.5

# ---- county level ----
all_fips = pd.read_csv(HERE.parent / "final" / "counties.csv", dtype={"fips": str}, usecols=["fips"]).fips
co = df.groupby("fips").agg(damaged=("damaged", "any"), ts_wind=("ts_wind", "any"),
                            log_no=("chance", lambda p: np.log1p(-np.minimum(p, 1 - 1e-12)).sum()))
co["chance"] = 1 - np.exp(co.log_no)
co = co.reindex(all_fips)
co["damaged"], co["ts_wind"] = co.damaged.eq(True), co.ts_wind.eq(True)   # counties no storm reached: False
co["chance"] = co.chance.fillna(0.0)

VIEWS = {"1 Storm x county, all": (df.damaged.to_numpy(), df.chance.to_numpy()),
         "2 Storm x county, tropical-storm wind": (df.damaged[df.ts_wind].to_numpy(), df.chance[df.ts_wind].to_numpy()),
         "3 County, all 3,109": (co.damaged.to_numpy(bool), co.chance.to_numpy()),
         "4 County, felt tropical-storm wind": (co.damaged[co.ts_wind].to_numpy(bool), co.chance[co.ts_wind].to_numpy())}


def cm(y, pred):
    tp, fn, fp, tn = (pred & y).sum(), (~pred & y).sum(), (pred & ~y).sum(), (~pred & ~y).sum()
    return {"true_pos": int(tp), "false_neg": int(fn), "false_pos": int(fp), "true_neg": int(tn),
            "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


rows = []
for view, (y, p) in VIEWS.items():
    majority = y.mean() >= 0.5
    rows.append({"view": view, "model": f"Naive: always '{'damage' if majority else 'no damage'}'", "cutoff": np.nan,
                 "units": len(y), "damage_rate": y.mean(), "roc_auc": 0.5, **cm(y, np.full(len(y), majority))})
    balanced = np.sort(p)[::-1][max(int(y.sum()) - 1, 0)]
    for label, cut in [("Best model, cutoff 0.5", 0.5), ("Best model, balanced cutoff", balanced)]:
        rows.append({"view": view, "model": label, "cutoff": cut, "units": len(y), "damage_rate": y.mean(),
                     "roc_auc": roc_auc(y, p), **cm(y, p >= cut)})
out = pd.DataFrame(rows)
out.round(4).to_csv(HERE / "results" / "confusion_views.csv", index=False)

# ---- chart: naive vs model (cutoff 0.5) for each view; cells show count and share of the actual row ----
fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
for j, view in enumerate(VIEWS):
    for i, which in enumerate(["Naive", "Best model, cutoff 0.5"]):
        r = out[(out.view == view) & out.model.str.startswith(which)].iloc[0]
        M = np.array([[r.true_pos, r.false_neg], [r.false_pos, r.true_neg]], float)
        share = M / M.sum(1, keepdims=True)
        ax = axes[i, j]
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for b_ in range(2):
                ax.text(b_, a, f"{int(M[a, b_]):,}\n{share[a, b_]:.0%} of row", ha="center", va="center", fontsize=9,
                        color="white" if share[a, b_] > 0.55 else "#1a1a1a")
        ax.set_xticks([0, 1], ["Predicted\ndamage", "Predicted\nnone"], fontsize=8)
        ax.set_yticks([0, 1], ["Actual damage", "Actual none"], fontsize=8)
        name = r.model if which == "Naive" else "Best model (cutoff 0.5)"
        ax.set_title(f"{view[2:]}\n{name}\naccuracy {r.accuracy:.0%} · recall {r.recall:.0%} · precision "
                     + ("n/a" if np.isnan(r.precision) else f"{r.precision:.0%}"), fontsize=8)
fig.suptitle("Recorded hurricane damage vs none: naive guess (top) vs best model (bottom), held-out storms 2000-2024", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "results" / "confusion_views.png", dpi=150)

pd.set_option("display.width", 250)
print(out[["view", "model", "units", "damage_rate", "true_pos", "false_neg", "false_pos", "true_neg",
           "accuracy", "precision", "recall", "f1", "roc_auc"]].round(3).to_string(index=False))

# ---- slide version: view 3 only (all 3,109 counties), naive vs best model ----
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, which in zip(axes, ["Naive", "Best model, cutoff 0.5"]):
    r = out[(out.view == "3 County, all 3,109") & out.model.str.startswith(which)].iloc[0]
    M = np.array([[r.true_pos, r.false_neg], [r.false_pos, r.true_neg]], float)
    share = M / M.sum(1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for a in range(2):
        for b_ in range(2):
            ax.text(b_, a, f"{int(M[a, b_]):,}\n{share[a, b_]:.0%} of row", ha="center", va="center", fontsize=10,
                    color="white" if share[a, b_] > 0.55 else "#1a1a1a")
    ax.set_xticks([0, 1], ["Predicted\ndamage", "Predicted\nnone"], fontsize=9)
    ax.set_yticks([0, 1], ["Actual damage", "Actual none"], fontsize=9)
    name = "Naive: always guess 'no damage'" if which == "Naive" else "Best model (cutoff 0.5)"
    ax.set_title(f"{name}\naccuracy {r.accuracy:.0%} · recall {r.recall:.0%} · precision "
                 + ("n/a" if np.isnan(r.precision) else f"{r.precision:.0%}"), fontsize=9)
fig.suptitle("Did a county record hurricane damage in 2000-2024? All 3,109 counties", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "results" / "confusion_county_all.png", dpi=150)
