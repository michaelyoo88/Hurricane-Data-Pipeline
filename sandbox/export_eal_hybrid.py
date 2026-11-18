"""County EAL from the round-8 best model (Y4 hybrid) and its comparison with FEMA NRI hurricane EAL.

Property: Y4 = average of (XGBoost chance on 11 inputs x gamma share lost on J2 + latitude) and the wind / surge /
rain two-part models, fit on ALL 194 storms (KNN-filled labels), every 2000-2024 storm priced on 2024 building
value, summed per county / 25 seasons. Crop and people: unchanged from results/hurricane_eal_by_county.csv.
Out: results/hurricane_eal_by_county_hybrid.csv, results/eal_hybrid_vs_fema_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from round8 import hybrid, prep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
df = prep()
ratio = hybrid("xgb", with_type=True)(df, np.ones(len(df), bool))
prop = pd.Series(ratio * df.bldg_value_2024.to_numpy(), index=df.index).groupby(df.fips).sum() / 25

old = pd.read_csv(HERE / "results" / "hurricane_eal_by_county.csv", dtype={"fips": str}).set_index("fips")
new = old.copy()
new["property_eal_usd"] = prop.reindex(new.index).fillna(0).round(0).astype("int64")
new["total_eal_usd"] = new.crop_eal_usd + new.property_eal_usd + new.people_eal_usd
new.reset_index().to_csv(HERE / "results" / "hurricane_eal_by_county_hybrid.csv", index=False)

co = pd.read_csv(ROOT / "final" / "counties.csv", dtype={"fips": str}).set_index("fips")
c = co[["state", "county_name", "coastal", "pop_lat", "bldg_value_usd2023"]].join(new[["property_eal_usd", "crop_eal_usd", "people_eal_usd", "total_eal_usd"]])
c["fema_property"] = co.nri_hrcn_eal_bldg_usd2023.fillna(0)
c["fema_crop"] = co.nri_hrcn_eal_ag_usd2023.fillna(0)
c["fema_people"] = co.nri_hrcn_eal_pope_usd2023.fillna(0)
c["fema_total"] = c.fema_property + c.fema_crop + c.fema_people
c = c.rename(columns={"property_eal_usd": "ours_property", "crop_eal_usd": "ours_crop", "people_eal_usd": "ours_people", "total_eal_usd": "ours_total"})

# 1. totals by category
tot = pd.DataFrame({k: {"ours_B": c[f"ours_{k}"].sum() / 1e9, "fema_B": c[f"fema_{k}"].sum() / 1e9,
                        "ours_counties_gt0": int((c[f"ours_{k}"] > 1000).sum()), "fema_counties_gt0": int((c[f"fema_{k}"] > 1000).sum())}
                    for k in ["property", "crop", "people", "total"]}).T
tot["ours_over_fema"] = tot.ours_B / tot.fema_B
tot.to_csv(HERE / "results" / "eal_hybrid_vs_fema_totals.csv")

# 2. concentration: share of national total EAL in the top N counties (each model ranks its own)
conc = []
for n in (10, 25, 50, 100, 250, 500):
    conc.append({"top_n": n, "ours_share": c.ours_total.nlargest(n).sum() / c.ours_total.sum(),
                 "fema_share": c.fema_total.nlargest(n).sum() / c.fema_total.sum()})
conc = pd.DataFrame(conc); conc.to_csv(HERE / "results" / "eal_hybrid_vs_fema_concentration.csv", index=False)

# 3. agreement by county (counties where either side has >= $1M/yr total)
def spear(a, b):
    return a.rank().corr(b.rank())
big = c[(c.ours_total >= 1e6) | (c.fema_total >= 1e6)]
r = big.ours_total.clip(lower=1) / big.fema_total.clip(lower=1)
agree = {"rank_corr_all": spear(c.ours_total, c.fema_total), "counties_1M": len(big),
         "median_ratio": r.median(), "typical_miss_x": 10 ** np.median(np.abs(np.log10(r))),
         "within_3x": ((r >= 1 / 3) & (r <= 3)).mean(), "ours_higher_share": (r > 1).mean()}

# 4. coastal vs inland, by state, by latitude band
c["where"] = np.where(c.coastal, "Coastal", "Inland")
geo = c.groupby("where")[["ours_total", "fema_total"]].sum() / 1e9
st = (c.groupby("state")[["ours_property", "fema_property", "ours_crop", "fema_crop", "ours_people", "fema_people", "ours_total", "fema_total"]].sum() / 1e6)
st = st.sort_values("fema_total", ascending=False)
st["ours_over_fema"] = st.ours_total / st.fema_total
st.round(1).to_csv(HERE / "results" / "eal_hybrid_vs_fema_by_state.csv")
top = c.sort_values("ours_total", ascending=False).head(15)[["state", "county_name", "ours_total", "fema_total"]]
top = top.assign(ours_M=top.ours_total / 1e6, fema_M=top.fema_total / 1e6)
pd.set_option("display.width", 250)
print(tot.round(3).to_string()); print(conc.round(3).to_string(index=False)); print({k: round(v, 3) for k, v in agree.items()})
print(geo.round(2).to_string(), "\n", (geo / geo.sum()).round(3).to_string())
print(st.head(14).round(0).to_string())
print(top[["state", "county_name", "ours_M", "fema_M"]].round(0).to_string())
# 5. where we differ most (counties with either >= $5M)
c["ratio"] = c.ours_total.clip(lower=1) / c.fema_total.clip(lower=1)
b5 = c[(c.ours_total >= 5e6) | (c.fema_total >= 5e6)]
print("ours >> FEMA:"); print(b5.sort_values("ratio", ascending=False).head(8)[["state", "county_name", "coastal", "ours_total", "fema_total", "ratio"]].round(1).to_string())
print("ours << FEMA:"); print(b5.sort_values("ratio").head(8)[["state", "county_name", "coastal", "ours_total", "fema_total", "ratio"]].round(2).to_string())
lat = pd.cut(c.pop_lat, [0, 30, 33, 36, 39, 50], labels=["<30", "30-33", "33-36", "36-39", "39+"])
print((c.groupby(lat, observed=True)[["ours_total", "fema_total"]].sum() / 1e9).round(2).to_string())
print("mix ours:", (tot.ours_B[:3] / tot.ours_B[3]).round(3).to_dict(), "fema:", (tot.fema_B[:3] / tot.fema_B[3]).round(3).to_dict())
