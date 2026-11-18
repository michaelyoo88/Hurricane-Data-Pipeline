"""The locked test, run ONCE: fit on all 134 dev storms, score on the 60 test storms never used before.

The final model was chosen in CHOICE.md before this script was run. The other finalists are scored for
context (does the dev ranking hold up?). 95% intervals come from a bootstrap over the test storms.
Also applies the final recipe to crop and people, and reports the total.
Out: results/final_test.csv, final_test_by_storm.csv, final_test_all_categories.csv, final_coefficients.csv,
     final_wind_curve.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare import FINALISTS, stats, storm_table
from lib import BASE, evaluate, load, predict, two_part

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from common import VSL, cpi_factors   # noqa: E402

N_BOOT, SEED = 1000, 42
FINAL = "J2 FINAL: 4 inputs"
FINAL_FEATURES = ["wind_cap", "coastal_x_wind_cap", "log_dist_km", "coastal"]
MODELS = {FINAL: {"features": FINAL_FEATURES}, **FINALISTS}

df = load()
dev = (df.set == "dev").to_numpy()
test = ~dev
e, y = df.bldg_value.to_numpy(), df.prop_usd.to_numpy()
null = y[dev].sum() / e[dev].sum() * e
base = df.loc[test, ["storm_id", "storm_strength", "fips", "bldg_value", "name", "year"]].assign(rec=y[test], null=null[test])

preds = {label: (predict(df, dev, spec) * e)[test] for label, spec in MODELS.items()}
lg = HERE / "results" / "lgbm_test_pred.csv"
if lg.exists():
    l = pd.read_csv(lg, dtype={"fips": str})
    preds["H1 LightGBM monotone"] = base[["storm_id", "fips"]].merge(l, on=["storm_id", "fips"])["H1 LightGBM monotone"].to_numpy()

rng = np.random.default_rng(SEED)
storms = base.storm_id.unique()
draws = [rng.choice(storms, len(storms), replace=True) for _ in range(N_BOOT)]
rows = []
for label, pred in preds.items():
    d = base.assign(pred=pred)
    m, st = evaluate(d), storm_table(d)
    b = pd.DataFrame([stats(st.loc[s]) for s in draws])
    ci = lambda c: f"{b[c].quantile(.025):.2f} to {b[c].quantile(.975):.2f}"   # noqa: E731
    rows.append({"model": label, "d2": m["d2"], "d2_95ci": ci("d2"), "pred_rec": m["pred_rec"], "pred_rec_95ci": ci("pred_rec"),
                 "tier_ts": m["tier_ts"], "tier_c12": m["tier_c12"], "tier_c3": m["tier_c3"], "tier_miss": m["tier_miss"],
                 "storm_miss": m["storm_miss"], "rho_storm": m["rho_storm"], "roc_auc": m["roc_auc"],
                 "top10": m["top10"], "within3x": m["within3x"]})
out = pd.DataFrame(rows)
out.round(3).to_csv(HERE / "results" / "final_test.csv", index=False)

d = base.assign(pred=preds[FINAL])
by_storm = d.groupby("storm_id").agg(name=("name", "first"), year=("year", "first"), tier=("storm_strength", "first"),
                                     recorded_B=("rec", lambda s: s.sum() / 1e9), predicted_B=("pred", lambda s: s.sum() / 1e9))
by_storm["pred_over_rec"] = by_storm.predicted_B / by_storm.recorded_B
by_storm.sort_values("recorded_B", ascending=False).round(3).to_csv(HERE / "results" / "final_test_by_storm.csv")

# ---- the same recipe for crop and people, then the total ----
VSL_2023 = VSL * cpi_factors()[2024]
CATS = {"property": ("prop_usd", "bldg_value", 1.0, 1.0), "crop": ("crop_usd", "farm_timber_value", 1.0, 10.0),
        "people": ("people_equiv", "population", VSL_2023, 1.0)}
cat_rows, total = [], {"final": 0.0, "old": 0.0, "rec": 0.0, "null": 0.0}
for cat, (target, exposure, usd, cap) in CATS.items():
    ex, yy = df[exposure].to_numpy(float), df[target].to_numpy(float) * usd
    nul = yy[dev].sum() / ex[dev].sum() * ex
    for which, feats in [("final", FINAL_FEATURES), ("old", BASE)]:
        pred = two_part(df, dev, target, features=feats, exposure=exposure, cap=cap) * ex * usd
        total[which] = total[which] + pred
        dd = df.loc[test, ["storm_id", "storm_strength", "fips"]].assign(bldg_value=np.maximum(ex[test], 1), pred=pred[test], rec=yy[test], null=np.maximum(nul[test], 1e-9))
        cat_rows.append({"category": cat, "model": "J2 final recipe" if which == "final" else "old base", **evaluate(dd)})
    total["rec"], total["null"] = total["rec"] + yy, total["null"] + nul
for which in ["final", "old"]:
    dd = df.loc[test, ["storm_id", "storm_strength", "fips", "bldg_value"]].assign(pred=total[which][test], rec=total["rec"][test], null=total["null"][test])
    cat_rows.append({"category": "total", "model": "J2 final recipe" if which == "final" else "old base", **evaluate(dd)})
cats = pd.DataFrame(cat_rows)
cats.round(3).to_csv(HERE / "results" / "final_test_all_categories.csv", index=False)

# ---- explain the final model: coefficients (fit on dev) and its wind curve ----
_, coefs = two_part(df, dev, "prop_usd", features=FINAL_FEATURES, details=True)
coefs.round(3).to_csv(HERE / "results" / "final_coefficients.csv", index=False)
winds = np.arange(5, 75.1, 1.0)
med_dist = df.loc[dev & (df.wind_ms >= 17.5), "log_dist_km"].median()
grid = pd.DataFrame([{"wind_ms": w, "coastal": c, "log_dist_km": med_dist} for c in (0.0, 1.0) for w in winds])
grid["wind_cap"] = grid.wind_ms.clip(upper=50)
grid["coastal_x_wind_cap"] = grid.coastal * (grid.wind_ms - 17.5).clip(0, 32.5)
both = pd.concat([df[dev], grid.assign(bldg_value=1.0, prop_usd=0.0, storm_id="curve")], ignore_index=True)
fit = np.r_[np.ones(dev.sum(), bool), np.zeros(len(grid), bool)]
grid["pct_lost"] = 100 * two_part(both, fit, "prop_usd", features=FINAL_FEATURES)[~fit]
fig, ax = plt.subplots(figsize=(7, 4.2))
for c, style, label in [(0.0, "-", "Inland county"), (1.0, "--", "Coastal county")]:
    g = grid[grid.coastal == c]
    ax.plot(g.wind_ms, g.pct_lost, linestyle=style, color="#2b6cb0", label=label)
ax.set_yscale("log")
for w, t in [(17.5, "tropical storm"), (33, "Cat 1"), (50, "Cat 3: curve held flat")]:
    ax.axvline(w, color="grey", linewidth=0.6, linestyle=":")
    ax.text(w + 0.5, ax.get_ylim()[0] * 1.5, t, fontsize=7, rotation=90, va="bottom")
ax.set_xlabel("Wind at the county (m/s)")
ax.set_ylabel("Expected share of building value lost per storm (%, log)")
ax.set_title("Final model: damage curve (storm at typical distance for tropical-storm wind)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "results" / "final_wind_curve.png", dpi=150)

# ---- print ----
assert test.sum() + dev.sum() == 97_179 and base.storm_id.nunique() == 60
pd.set_option("display.width", 250)
print(f"LOCKED TEST: {base.storm_id.nunique()} storms, {test.sum():,} rows, recorded property ${y[test].sum() / 1e9:.1f}B\n")
print(out.round(2).to_string(index=False))
print("\nfinal model, biggest test storms ($B):")
print(by_storm.sort_values("recorded_B", ascending=False).head(12).round(2).to_string())
print("\nall categories on the locked test (final recipe vs old base):")
print(cats[["category", "model", "d2", "pred_rec", "tier_ts", "tier_c12", "tier_c3", "storm_miss", "rho_storm", "roc_auc", "top10", "within3x"]].round(2).to_string(index=False))
print("\nfinal model coefficients (standardized inputs):"); print(coefs.round(2).to_string(index=False))
