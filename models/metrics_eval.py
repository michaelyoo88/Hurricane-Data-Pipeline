"""Which metrics fit our goal: a dollar EAL per county?

Compares every model on candidate metrics at two levels, using held-out predictions
priced on that year's exposure (same basis as the recorded damage):
  county x year   one row per county per year (the annual loss)
  county EAL      each county's 25-year average (2000-2024) = the product we deliver

Metrics
  roc_auc            did it flag the county-years that had damage? (0.5 = guessing)
  spearman           does it put counties in the right order? (every county counts the same)
  gini               normalized Gini: does it put the DOLLARS where the losses are? Counties
                     sorted by prediction; 1 = perfect order, 0 = random. Insurance standard.
  top10_capture      share of all recorded dollars found in the model's top 10% of counties
  typical_factor     middle county's predicted / recorded EAL, as "off by X times" (1 = right)
  within_3x          share of counties whose EAL is within 3x of recorded
  log_r2             R-squared on log10 EAL (how much of the county-to-county spread it explains)
  r2_dollars         ordinary R-squared on raw dollars (to show why we avoid it)
  pred_over_recorded total predicted / total recorded
EAL-level metrics use counties with recorded EAL >= $10k per year (below that the 25-year
average is mostly noise). Out: results/metrics_eval.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from glm import roc_auc, spearman

HERE = Path(__file__).resolve().parent
MIN_EAL = 10_000


def gini(actual, pred):
    """Normalized Gini: how well `pred` orders units by `actual` dollars (1 = perfect)."""
    def raw(a, p):
        a = np.asarray(a, float)[np.lexsort((np.arange(len(p)), -np.asarray(p, float)))]
        return (a.cumsum().sum() / a.sum() - (len(a) + 1) / 2) / len(a)
    return raw(actual, pred) / raw(actual, actual)


def top_capture(actual, pred, share=0.10):
    k = max(1, int(len(pred) * share))
    top = np.argsort(-np.asarray(pred, float))[:k]
    return np.asarray(actual, float)[top].sum() / np.sum(actual)


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


d = pd.read_csv(HERE / "data" / "county_year_preds.csv", dtype={"fips": str})
rows = []
for (name, cat), g in [((s, c), d[d.strategy == s]) for s in d.strategy.unique() for c in ["total", "property", "crop", "people"]]:
    pred, rec = g[f"{cat}_pred"].to_numpy(), g[f"{cat}_rec"].to_numpy()
    eal = g.groupby("fips")[[f"{cat}_pred", f"{cat}_rec"]].sum() / 25
    ep, er = eal[f"{cat}_pred"], eal[f"{cat}_rec"]
    m = er >= MIN_EAL
    ratio = ep[m].clip(lower=1) / er[m]
    rows.append({
        "strategy": name, "category": cat,
        "cy_roc_auc": roc_auc(rec > 0, pred), "cy_spearman": spearman(pred, rec), "cy_gini": gini(rec, pred),
        "eal_spearman": spearman(ep, er), "eal_gini": gini(er, ep), "eal_top10_capture": top_capture(er, ep),
        "eal_typical_factor": 10 ** np.median(np.abs(np.log10(ratio))),
        "eal_within_3x": ((ratio >= 1 / 3) & (ratio <= 3)).mean(),
        "eal_log_r2": r2(np.log10(er[m]), np.log10(ep[m].clip(lower=1))),
        "eal_r2_dollars": r2(er, ep),
        "pred_over_recorded": pred.sum() / rec.sum(), "counties_scored": int(m.sum())})
out = pd.DataFrame(rows)
out.round(3).to_csv(HERE / "results" / "metrics_eval.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
t = out[out.category == "total"].drop(columns="category").set_index("strategy")
print("TOTAL damage (counties scored at EAL level:", int(t.counties_scored.iloc[0]), ")")
print(t.drop(columns="counties_scored").round(3).to_string())
# how much do metrics agree on which model is better?
print("\nrank correlation between metrics across the 8 models (total):")
print(t.drop(columns=["counties_scored"]).corr(method="spearman").round(2).to_string())
# a perfect-scale check: rescale each model so its total matches recorded, recompute typical factor / within 3x
print("\nafter rescaling each model's total to match recorded (does order or scale drive the error?):")
for name, g in d.groupby("strategy"):
    eal = g.groupby("fips")[["total_pred", "total_rec"]].sum() / 25
    k = eal.total_rec.sum() / eal.total_pred.sum()
    m = eal.total_rec >= MIN_EAL
    ratio = (eal.total_pred[m] * k).clip(lower=1) / eal.total_rec[m]
    print(f"  {name:24s} scale x{k:.2f}: typical factor {10 ** np.median(np.abs(np.log10(ratio))):.2f}, within 3x {((ratio >= 1/3) & (ratio <= 3)).mean():.2f}")
for cat in ["property", "crop", "people"]:
    print(f"\n{cat}:"); print(out[out.category == cat].set_index("strategy")[["cy_roc_auc", "eal_gini", "eal_top10_capture",
          "eal_typical_factor", "eal_within_3x", "eal_log_r2", "pred_over_recorded"]].round(2).to_string())
