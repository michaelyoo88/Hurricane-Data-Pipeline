"""Slide 3 chart: the 2 best models and the weakest. Reads results/metrics_eval.csv (run metrics_eval.py first).

  ROC-AUC              did it rate county-years with damage above those without? (0.5 = coin flip)
  Top 10% capture      of all recorded losses, the share found in the model's 10% riskiest counties
                       (does it find WHERE the risk is?; random = 10%)
  Predicted / recorded total predicted / total recorded dollars (is the overall level right?)
Best and worst are ranked on "within 3x" of recorded county EAL (in metrics_eval.csv), the closest measure of the product.
Counties scored: recorded EAL >= $10k per year (943 counties). Out: results/presentation_models.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = {"0 basic two-part": "1. Two-part GLM",
          "1 damage table": "2. Damage table",
          "2 top-down": "3. Top-down (storm total, then split)",
          "3 by damage type": "4. Two-part GLM by damage type",
          "4a weights halfway": "5. Two-part GLM, halfway weights",
          "4b weights by dollars": "6. Two-part GLM, dollar weights",
          "4c tier rescale": "7. Two-part GLM + tier correction",
          "6 county-year two-part": "8. Two-part GLM (county x year)"}
PANELS = [("cy_roc_auc", "ROC-AUC: spots which counties get damaged", 0.5, "coin flip"),
          ("eal_top10_capture", "Losses found in model's top 10% of counties", 0.10, "random"),
          ("pred_over_recorded", "Predicted / recorded total dollars", 1.0, "right total")]

d = pd.read_csv(HERE / "results" / "metrics_eval.csv")
d = d[d.category == "total"].set_index("strategy")
keep = list(d.eal_within_3x.nlargest(2).index) + [d.eal_within_3x.idxmin()]
d = d.loc[keep]
names = [LABELS[s] for s in d.index][::-1]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
for ax, (col, title, ref, ref_label) in zip(axes, PANELS):
    values = d[col].to_numpy()[::-1]
    xmax = 1.7 if col == "pred_over_recorded" else 1.0
    pct = col == "eal_top10_capture"
    ax.barh(names, values, color="#2b6cb0")
    for i, v in enumerate(values):
        label = f"{v:.0%}" if pct else f"{v:.2f}"
        ax.text(v + xmax * 0.015, i, label, va="center", fontsize=8)
    if ref is not None:
        ax.axvline(ref, color="black", linestyle="--", linewidth=0.9)
        ax.text(ref, len(names) - 0.4, f" {ref_label}", fontsize=7, va="bottom")
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, len(names) - 0.1)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Best two models vs. the weakest: county EAL for total hurricane damage, 2000-2024 (held out)", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "results" / "presentation_models.png", dpi=200)
print("wrote results/presentation_models.png:", ", ".join(LABELS[s] for s in keep))
