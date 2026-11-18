"""Error metrics for the damage-estimation model (the averaged model used for the group CSV, property damage).

Held-out predictions: 3 repeats of 5-fold over all 194 storms, whole storms held out, trained on KNN-filled
labels, scored against the RECORDED damage. Predictions are averaged over the 3 repeats.
Levels: storm x county rows, damaged rows only, storm totals, county EAL (25-season average).
Also compares the model EAL with a plain average of each county's recorded damage.
Out: results/error_metrics.csv, results/eal_vs_recorded_average.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from knn_impute import K12_IMP, add_imputation
from lib import load, oof, spearman

HERE = Path(__file__).resolve().parent
df = load()
st = pd.read_csv(HERE / "data" / "knn_stats.csv", dtype={"fips": str})
df = df.merge(st[["storm_id", "fips", "nb_share", "nb_median_pos"]], on=["storm_id", "fips"], how="left", validate="one_to_one")
add_imputation(df, 25.0, "expected")
runs = [oof(df, K12_IMP, r, scope="all") for r in range(3)]
d = runs[0].assign(pred=np.mean([r.pred for r in runs], axis=0))
d["year"] = df.year.to_numpy()


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def block(level, y, f, note):
    y, f = np.asarray(y, float), np.asarray(f, float)
    ly, lf = np.log10(1 + y), np.log10(1 + f)
    pos = y > 0
    return {"level": level, "units": len(y), "recorded_total_B": y.sum() / 1e9, "predicted_total_B": f.sum() / 1e9,
            "MAE_$M": np.abs(f - y).mean() / 1e6, "RMSE_$M": np.sqrt(((f - y) ** 2).mean()) / 1e6,
            "R2_dollars": r2(y, f), "R2_log10": r2(ly, lf), "mean_abs_log10_err": np.abs(lf - ly).mean(),
            "typical_miss_x": 10 ** np.median(np.abs(np.log10(np.maximum(f[pos], 1) / y[pos]))) if pos.any() else np.nan,
            "within_3x": float(np.mean((np.maximum(f[pos], 1) / y[pos] >= 1 / 3) & (np.maximum(f[pos], 1) / y[pos] <= 3))) if pos.any() else np.nan,
            "rank_corr": spearman(f, y), "note": note}


rows = [block("storm x county, all rows", d.rec, d.pred, "94% of rows are $0"),
        block("storm x county, damaged rows only", d.rec[d.rec > 0], d.pred[d.rec > 0], "prediction = chance x size, so lower by design"),
        ]
s = d.groupby("storm_id")[["rec", "pred"]].sum()
rows.append(block("storm totals", s.rec, s.pred, "194 storms"))
cy = d.groupby(["fips", "year"])[["rec", "pred"]].sum()
rows.append(block("county x year", cy.rec, cy.pred, "annual loss"))
eal = d.groupby("fips")[["rec", "pred"]].sum() / 25
rows.append(block("county EAL (25-season average)", eal.rec, eal.pred, "same exposure basis as the records"))
big = eal[eal.rec >= 1e6]
rows.append(block("county EAL, counties with recorded EAL >= $1M/yr", big.rec, big.pred, f"{len(big)} counties"))
out = pd.DataFrame(rows)
out.round(3).to_csv(HERE / "results" / "error_metrics.csv", index=False)

# ---- model EAL vs plain recorded average ----
csv = pd.read_csv(HERE / "results" / "hurricane_eal_by_county.csv", dtype={"fips": str}).set_index("fips")
recavg = (df.groupby("fips").prop_usd.sum() / 25).reindex(csv.index).fillna(0)
names = pd.read_csv(HERE.parent / "final" / "counties.csv", dtype={"fips": str}, usecols=["fips", "state", "county_name"]).set_index("fips")
cmp_ = names.join(pd.DataFrame({"model_property_eal": csv.property_eal_usd, "recorded_average": recavg}))
cmp_["model_over_recorded"] = cmp_.model_property_eal / cmp_.recorded_average.replace(0, np.nan)
cmp_.round(2).to_csv(HERE / "results" / "eal_vs_recorded_average.csv")

pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 50)
print(out.drop(columns="units").round(3).to_string(index=False))
z = cmp_[(cmp_.recorded_average == 0) & (cmp_.model_property_eal > 0)]
print(f"\nmodel EAL vs recorded average: rank corr {spearman(cmp_.model_property_eal, cmp_.recorded_average):.2f}; "
      f"{len(z)} counties have $0 recorded but a model EAL > 0 (top: " +
      ", ".join(f"{r.county_name} {r.state} ${r.model_property_eal / 1e6:.1f}M" for _, r in z.nlargest(3, "model_property_eal").iterrows()) + ")")
for fips in ["12071", "48201", "22071", "12086"]:
    r = cmp_.loc[fips]
    print(f"  {r.county_name}, {r.state}: recorded average ${r.recorded_average / 1e6:,.0f}M/yr, model EAL ${r.model_property_eal / 1e6:,.0f}M/yr")
