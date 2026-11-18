"""Round 7 chart: do KNN-flagged $0 records hide real damage? Two outside checks.
Reads results/knn_flag_sensitivity.csv (property) and results/knn_crop_check.csv (crop).
Out: results/knn_validation.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
p = pd.read_csv(HERE / "results" / "knn_flag_sensitivity.csv")
c = pd.read_csv(HERE / "results" / "knn_crop_check.csv")
panels = [("Property: zeros under the old wind-only labels\nthat later showed surge / flood damage", p, "flagged_found_later", "not_flagged_found_later",
           [(25.0, 0.5), (33.0, 0.5)]),
          ("Crop: NOAA zeros where crop insurance\npaid a hurricane claim (USDA RMA)", c, "flagged_with_insurance_claims", "not_flagged_with_claims",
           [(17.5, 0.5), (25.0, 0.5)])]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (title, d, f_col, n_col, rules) in zip(axes, panels):
    labels, flagged, other = [], [], []
    for w, s in rules:
        r = d[(d.min_wind == w) & (d.min_neighbor_share == s)].iloc[0]
        labels.append(f"{w:g}+ m/s wind\n({int(r.flagged)} flagged)")
        flagged.append(r[f_col]); other.append(r[n_col])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, flagged, 0.36, color="#2b6cb0", label="Flagged by KNN")
    ax.bar(x + 0.18, other, 0.36, color="#2b6cb0", alpha=0.35, label="Similar zeros, not flagged")
    top = max(flagged + other)
    for i, (a, b) in enumerate(zip(flagged, other)):
        ax.text(i - 0.18, a + top * 0.02, f"{a:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.18, b + top * 0.02, f"{b:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(flagged + other) * 1.25)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Share with real damage found elsewhere")
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("KNN flags $0 records that hide real damage more often than chance (flag: half of 10 similar counties, same storm, recorded damage)", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "results" / "knn_validation.png", dpi=150)
