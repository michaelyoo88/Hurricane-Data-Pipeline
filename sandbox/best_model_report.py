"""Numbers for the class write-up on the round-8 best model (Y4) and its comparison with FEMA (2026-11-18).

Y4 = average of (XGBoost chance of damage on the rich inputs x gamma GLM size on J2 + latitude) and the
wind / surge / rain two-part model; trained on KNN-filled property labels.
  1. Confusion matrices of the "any damage?" step, held out (fold_all_r0): logistic J2 (old), logistic J2+lat
     (current K12 member), XGBoost rich (Y4). Cutoff = flag as many county-storms as really had damage.
  2. Y4 property EAL per county, fit on all 194 storms, 2024 exposure, 25 seasons. Crop and people stay from
     hurricane_eal_by_county.csv (round-8 work was property only).
  3. Comparison with FEMA NRI hurricane EAL (2023 dollars).
Out: results/best_confusion.csv, results/best_eal_by_county.csv, results/best_vs_fema_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lib import J2, REGIONS, fit_logistic, spearman
from round8 import LAT, RICH, by_type, chance_tree, prep, size_glm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
df = prep()
res = HERE / "results"

# ---------------- 1. confusion matrices (held out)
y = (df.prop_usd > 0).to_numpy()
fold = df.fold_all_r0.to_numpy()


def logistic_prob(feats):
    X = df[feats].to_numpy(float); p = np.zeros(len(df))
    for k in range(5):
        fit, held = fold != k, fold == k
        m, s = X[fit].mean(0), X[fit].std(0)
        X1 = np.column_stack([np.ones(len(X)), (X - m) / s])
        b = fit_logistic(X1[fit], y[fit].astype(float), np.r_[0.0, np.ones(len(feats))])
        p[held] = 1 / (1 + np.exp(-np.clip(X1[held] @ b, -30, 30)))
    return p


def xgb_prob():
    p = np.zeros(len(df))
    tmp = df.assign(prop_usd_imp=df.prop_usd)          # score the classifier on recorded labels, like confusion.py
    for k in range(5):
        fit, held = fold != k, fold == k
        p[held] = chance_tree(tmp, fit, "xgb", RICH)[held]
    return p


rows = []
for name, p in [("Logistic, 4 inputs (J2)", logistic_prob(J2)), ("Logistic, 4 inputs + latitude (K12)", logistic_prob(LAT)),
                ("XGBoost, rich inputs (best model, Y4)", xgb_prob())]:
    cut = np.sort(p)[::-1][y.sum() - 1]
    f = p >= cut
    tp, fp, fn, tn = (f & y).sum(), (f & ~y).sum(), (~f & y).sum(), (~f & ~y).sum()
    strong = df.wind_ms.to_numpy() >= 25
    rows.append({"model": name, "cutoff": cut, "true_pos": tp, "false_pos": fp, "false_neg": fn, "true_neg": tn,
                 "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                 "specificity": tn / (tn + fp), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                 "base_rate": y.mean(), "recall_strong_wind": (f & y & strong).sum() / (y & strong).sum()})
conf = pd.DataFrame(rows)
conf.round(4).to_csv(res / "best_confusion.csv", index=False)

# ---------------- 2. Y4 property EAL
everything = np.ones(len(df), bool)
hyb = np.minimum(chance_tree(df, everything, "xgb", RICH) * size_glm(df, everything), 1.0)
ratio = np.minimum((hyb + by_type(df, everything)) / 2, 1.0)
prop = pd.Series(ratio * df.bldg_value_2024.to_numpy(), index=df.index).groupby(df.fips).sum() / 25
old = pd.read_csv(res / "hurricane_eal_by_county.csv", dtype={"fips": str}).set_index("fips")
eal = old[["crop_eal_usd", "people_eal_usd"]].join(prop.rename("property_eal_usd")).fillna(0.0)
eal["k12_property_eal_usd"] = old.property_eal_usd
eal["total_eal_usd"] = eal.crop_eal_usd + eal.property_eal_usd + eal.people_eal_usd
eal.round(0).to_csv(res / "best_eal_by_county.csv")

# ---------------- 3. vs FEMA
co = pd.read_csv(ROOT / "final" / "counties.csv", dtype={"fips": str}).set_index("fips")
f = co[["state", "county_name", "coastal", "population"]].join(eal)
f["fema_property"] = co.nri_hrcn_eal_bldg_usd2023.fillna(0)
f["fema_people"] = co.nri_hrcn_eal_pope_usd2023.fillna(0)
f["fema_crop"] = co.nri_hrcn_eal_ag_usd2023.fillna(0)
f["fema_total"] = f.fema_property + f.fema_people + f.fema_crop
f = f.fillna(0)
region = pd.Series("Inland / other", index=f.index)
names = {"reg_tx_la": "Gulf: TX-LA", "reg_ms_al": "Gulf: MS-AL", "reg_fl": "Florida", "reg_se_atl": "Southeast Atlantic: GA-SC-NC",
         "reg_mid_atl": "Mid-Atlantic: VA-NY", "reg_new_eng": "New England"}
for k, st in REGIONS.items():
    region[f.state.isin(st)] = names[k]
f["region"] = region

cat = pd.DataFrame({"ours_B": [f.property_eal_usd.sum(), f.crop_eal_usd.sum(), f.people_eal_usd.sum(), f.total_eal_usd.sum()],
                    "fema_B": [f.fema_property.sum(), f.fema_crop.sum(), f.fema_people.sum(), f.fema_total.sum()]},
                   index=["property", "crop", "people", "total"]) / 1e9
cat["ours_over_fema"] = cat.ours_B / cat.fema_B
cat["ours_share"] = cat.ours_B / cat.loc["total", "ours_B"]; cat["fema_share"] = cat.fema_B / cat.loc["total", "fema_B"]
cat.round(3).to_csv(res / "best_vs_fema_categories.csv")


def concentration(s):
    v = np.sort(s.to_numpy())[::-1]; c = v.cumsum() / v.sum()
    return {"top10_share": c[9], "top50_share": c[49], "top100_share": c[99],
            "counties_for_50pct": int(np.searchsorted(c, .5) + 1), "counties_for_90pct": int(np.searchsorted(c, .9) + 1),
            "counties_with_eal": int((v > 0).sum())}


conc = pd.DataFrame({"ours": concentration(f.total_eal_usd), "fema": concentration(f.fema_total)})
conc.to_csv(res / "best_vs_fema_concentration.csv")

reg = f.groupby("region")[["property_eal_usd", "fema_property", "crop_eal_usd", "fema_crop", "total_eal_usd", "fema_total"]].sum() / 1e9
reg["ratio_total"] = reg.total_eal_usd / reg.fema_total
reg["ratio_property"] = reg.property_eal_usd / reg.fema_property
reg = reg.sort_values("fema_total", ascending=False)
reg.round(3).to_csv(res / "best_vs_fema_regions.csv")

stt = f.groupby("state")[["total_eal_usd", "fema_total", "property_eal_usd", "fema_property"]].sum() / 1e9
stt["ratio"] = stt.total_eal_usd / stt.fema_total
stt = stt.sort_values("fema_total", ascending=False)
stt.round(3).to_csv(res / "best_vs_fema_states.csv")

hi = f[(f.total_eal_usd >= 1e6) | (f.fema_total >= 1e6)]
r = (hi.property_eal_usd.clip(lower=1) / hi.fema_property.clip(lower=1))
coast = {c: (lambda g: g.property_eal_usd.sum() / g.fema_property.sum())(f[f.coastal == c]) for c in (0, 1)}
summ = {"rank_corr_property_all": spearman(f.property_eal_usd, f.fema_property),
        "rank_corr_total_all": spearman(f.total_eal_usd, f.fema_total),
        "rank_corr_property_high": spearman(hi.property_eal_usd, hi.fema_property),
        "high_risk_counties": len(hi), "within_3x_property_high": float(((r >= 1/3) & (r <= 3)).mean()),
        "typical_gap_x_property_high": float(10 ** np.median(np.abs(np.log10(r)))),
        "median_ratio_property_high": float(r.median()),
        "ours_over_fema_coastal_property": coast[1], "ours_over_fema_inland_property": coast[0],
        "coastal_share_ours_property": f[f.coastal == 1].property_eal_usd.sum() / f.property_eal_usd.sum(),
        "coastal_share_fema_property": f[f.coastal == 1].fema_property.sum() / f.fema_property.sum(),
        "y4_over_k12_property_total": f.property_eal_usd.sum() / f.k12_property_eal_usd.sum()}
pd.Series(summ).to_csv(res / "best_vs_fema_summary.csv")
f[["state", "county_name", "coastal", "region", "property_eal_usd", "crop_eal_usd", "people_eal_usd", "total_eal_usd",
   "fema_property", "fema_crop", "fema_people", "fema_total"]].round(0).to_csv(res / "best_vs_fema_counties.csv")

pd.set_option("display.width", 250)
print(conf.round(3).to_string(index=False)); print(cat.round(3)); print(conc); print(reg.round(2)); print(stt.head(12).round(2))
print(pd.Series(summ).round(3))
top = f.nlargest(10, "total_eal_usd")[["state", "county_name", "total_eal_usd", "fema_total"]].copy()
top[["total_eal_usd", "fema_total"]] /= 1e6
print(top.round(0)); t2 = f.nlargest(10, "fema_total")[["state", "county_name", "total_eal_usd", "fema_total"]].copy(); t2[["total_eal_usd", "fema_total"]] /= 1e6; print(t2.round(0))
