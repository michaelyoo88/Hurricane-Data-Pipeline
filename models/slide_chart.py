"""One chart for the presentation: every model on three familiar metrics.
Reads the saved strategy results (run strategies.py first).
Out: results/slide_model_comparison.png, results/slide_model_comparison.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
BLUE = "#2b6cb0"
NAMES = {"0 basic two-part": "Two-part regression (baseline)", "1 damage table": "Damage table (wind bin x coast)",
         "2 top-down": "Top-down (storm total, then split)", "3 by damage type": "Separate wind / surge / rain models",
         "4a weights halfway": "Two-part, big storms count more", "4b weights by dollars": "Two-part, weighted by dollars",
         "4c tier rescale": "Two-part + storm-tier correction", "6 county-year two-part": "County-year two-part"}

storm = pd.read_csv(OUT / "strategies_storm_metrics.csv").query("category == 'total'").set_index("strategy")
year = pd.read_csv(OUT / "strategies_county_year.csv").query("category == 'total'").set_index("strategy")
t = pd.DataFrame({"model": pd.Series(NAMES),
                  "pr_auc": year.pr_auc_damage_that_year, "pr_auc_chance": year.base_rate,
                  "storm_rank_corr": storm.rank_corr_storms,
                  "pred_over_recorded": storm.pred_over_recorded.combine_first(year.pred_over_recorded),
                  "ci": storm.pred_over_recorded_95ci, "eal_B_per_yr": year.eal_B_per_yr}).loc[list(NAMES)]
t[["ci_lo", "ci_hi"]] = t.ci.str.split(" to ", expand=True).astype(float)
t.drop(columns="ci").round(3).to_csv(OUT / "slide_model_comparison.csv", index=False)
print(t.drop(columns="ci").round(2).to_string(index=False))

t = t.iloc[::-1]                                   # first model at the top
y = np.arange(len(t))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)

ax = axes[0]
ax.barh(y, t.pr_auc, color=BLUE)
ax.axvline(t.pr_auc_chance.iloc[0], color="black", linestyle="--", linewidth=1)
ax.text(t.pr_auc_chance.iloc[0] + 0.01, len(t) - 0.45, "random guess", fontsize=8)
ax.set_xlim(0, 0.62); ax.set_xlabel("PR-AUC (higher is better)")
ax.set_title("Did the county have damage that year?", fontsize=10)
for i, v in enumerate(t.pr_auc):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

ax = axes[1]
ax.barh(y, t.storm_rank_corr.fillna(0), color=BLUE)
ax.set_xlim(0, 1.05); ax.set_xlabel("Spearman rank correlation (1 = perfect order)")
ax.set_title("Are storms ranked from mild to severe?", fontsize=10)
for i, v in enumerate(t.storm_rank_corr):
    ax.text((v if pd.notna(v) else 0) + 0.02, i, f"{v:.2f}" if pd.notna(v) else "not a storm-level model", va="center", fontsize=8)

ax = axes[2]
lo = (t.pred_over_recorded - t.ci_lo).fillna(0); hi = (t.ci_hi - t.pred_over_recorded).fillna(0)
ax.errorbar(t.pred_over_recorded, y, xerr=[lo, hi], fmt="o", color=BLUE, capsize=3)
ax.axvline(1, color="black", linestyle="--", linewidth=1)
ax.set_xscale("log"); ax.set_xlim(0.2, 4)
ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.set_xlabel("Predicted $ / recorded $ (1 = exact; bars = 95% interval)")
ax.set_title("Is the dollar total right?", fontsize=10)

axes[0].set_yticks(y); axes[0].set_yticklabels(t.model, fontsize=9)
fig.suptitle("Preliminary results: each model tested on storms it never saw (194 storms, 2000-2024, total damage)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "slide_model_comparison.png", dpi=200)
