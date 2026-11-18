"""Round 6 summary: the finalists on all three checks, and one chart.

  dev       3 x 5-fold over the 134 dev storms (where the ideas were screened)
  all       fresh 3 x 5-fold over all 194 storms (models fixed in advance)
  test      fit on the 134 dev storms, scored on the 60 locked test storms. NOTE: this test was already
            opened once (round 5), so it is context only and is not used to pick a model.
Reads results/finalists_round6_dev.csv and finalists_round6_all.csv (run compare.py round6 dev / all).
Out: results/round6_test.csv, results/round6_summary.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare import ROUND6, stats, storm_table
from lib import evaluate, load, predict

HERE = Path(__file__).resolve().parent
N_BOOT, SEED = 1000, 42
df = load()
dev = (df.set == "dev").to_numpy(); test = ~dev
e, y = df.bldg_value.to_numpy(), df.prop_usd.to_numpy()
base = df.loc[test, ["storm_id", "storm_strength", "fips", "bldg_value"]].assign(rec=y[test], null=(y[dev].sum() / e[dev].sum() * e)[test])
rng = np.random.default_rng(SEED)
storms = base.storm_id.unique()
draws = [rng.choice(storms, len(storms), replace=True) for _ in range(N_BOOT)]
rows = []
for label, spec in ROUND6.items():
    d = base.assign(pred=(predict(df, dev, spec) * e)[test])
    m, st = evaluate(d), storm_table(d)
    b = pd.DataFrame([stats(st.loc[s]) for s in draws])
    rows.append({"model": label, **m, "d2_lo": b.d2.quantile(.025), "d2_hi": b.d2.quantile(.975)})
t = pd.DataFrame(rows)
t.round(3).to_csv(HERE / "results" / "round6_test.csv", index=False)

# ---- chart: deviance explained on the three checks, with 95% intervals ----
devr = pd.read_csv(HERE / "results" / "finalists_round6_dev.csv").set_index("model")
allr = pd.read_csv(HERE / "results" / "finalists_round6_all.csv").set_index("model")
tr = t.set_index("model")
ci = lambda s: [float(x) for x in s.split(" to ")]   # noqa: E731
models = list(ROUND6)[::-1]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (title, get) in zip(axes, [
        ("Dev storms (134), where ideas were screened", lambda m: (devr.d2[m], *ci(devr.d2_95ci[m]))),
        ("All storms (194), fresh folds", lambda m: (allr.d2[m], *ci(allr.d2_95ci[m]))),
        ("Locked test (60), seen before: context only", lambda m: (tr.d2[m], tr.d2_lo[m], tr.d2_hi[m]))]):
    v = [get(m) for m in models]
    pts, lo, hi = np.array([x[0] for x in v]), np.array([x[1] for x in v]), np.array([x[2] for x in v])
    ref = pts[models.index("K0 J2 reference (4 inputs)")]
    ax.axvline(ref, color="grey", linewidth=0.8, linestyle=":")
    ax.errorbar(pts, range(len(models)), xerr=[pts - lo, hi - pts], fmt="o", color="#2b6cb0", capsize=3, markersize=5)
    ax.set_title(title, fontsize=9); ax.set_xlabel("Deviance explained (D2), 95% interval")
    ax.set_xlim(-0.2, 1.0)
axes[0].set_yticks(range(len(models))); axes[0].set_yticklabels(models, fontsize=8)
fig.suptitle("Round 6: literature-based ideas vs. the 4-input model (dotted line), property damage", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "results" / "round6_summary.png", dpi=150)

pd.set_option("display.width", 250)
print("LOCKED TEST (seen before, context only), 60 storms:")
print(t[["model", "d2", "d2_lo", "d2_hi", "pred_rec", "tier_ts", "tier_c12", "tier_c3", "tier_miss", "storm_miss", "rho_storm", "roc_auc", "top10", "within3x"]].round(2).to_string(index=False))
