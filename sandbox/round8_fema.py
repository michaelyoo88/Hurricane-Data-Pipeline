"""Round 8 best model (Y4: XGBoost chance x gamma size, averaged with the wind/surge/rain model) vs FEMA NRI.

Fit on ALL 194 storms (2000-2024) on KNN-filled property labels, predict every storm x county, price on 2024
building value, sum per county / 25 seasons (same recipe as export_eal.py). Compared with FEMA NRI v1.20
hurricane building EAL (2023 dollars). K12 (current group file) shown alongside.
Out: results/round8_fema_county.csv, results/round8_fema_summary.csv, results/round8_fema_states.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lib import spearman
from round8 import by_type, hybrid, k12, prep, RICH

HERE = Path(__file__).resolve().parent
df = prep()
everything = np.ones(len(df), bool)
v24 = df.bldg_value_2024.to_numpy(float)
wind_only = __import__("lib").two_part(df, everything, "prop_wind_usd_imp", features=__import__("lib").capped())
preds = {"y4": hybrid("xgb", RICH, with_type=True)(df, everything), "k12": k12(df, everything)}
eal = pd.DataFrame({k: v * v24 for k, v in preds.items()}).assign(fips=df.fips.to_numpy(), y4_wind=wind_only * v24,
                                                                   recorded=df.prop_usd.to_numpy()).groupby("fips").sum() / 25
co = pd.read_csv(HERE.parent / "final" / "counties.csv", dtype={"fips": str}).set_index("fips")
t = co[["state", "county_name", "nri_hrcn_eal_bldg_usd2023"]].rename(columns={"nri_hrcn_eal_bldg_usd2023": "fema"}).join(eal).fillna(0)
t.round(0).to_csv(HERE / "results" / "round8_fema_county.csv")

rows = []
for m in ["y4", "k12", "y4_wind", "recorded"]:
    either = (t[m] > 0) | (t.fema > 0)
    big = (t[m] >= 1e6) | (t.fema >= 1e6)
    both = big & (t[m] > 0) & (t.fema > 0)
    r = t.loc[both, m] / t.loc[both, "fema"]
    rows.append({"model": m, "ours_B": t[m].sum() / 1e9, "fema_B": t.fema.sum() / 1e9, "rank_corr": spearman(t.loc[either, m], t.loc[either, "fema"]),
                 "high_risk_counties": int(big.sum()), "typical_gap_x": 10 ** np.median(np.abs(np.log10(r))),
                 "within_3x": float(((r >= 1 / 3) & (r <= 3)).mean()), "median_ours_over_fema": float(r.median()),
                 "share_ours_above": float((r > 1).mean())})
s = pd.DataFrame(rows)
s.round(3).to_csv(HERE / "results" / "round8_fema_summary.csv", index=False)
st = t.groupby("state")[["y4", "fema", "y4_wind", "k12", "recorded"]].sum().sort_values("fema", ascending=False) / 1e6
st.round(1).to_csv(HERE / "results" / "round8_fema_states.csv")
print(s.round(3).to_string(index=False))
print(st.head(14).round(0).to_string())
big = t[(t.y4 >= 5e6) | (t.fema >= 5e6)].assign(ratio=lambda d: d.y4 / d.fema.replace(0, np.nan))
print("\nabove FEMA:\n", big.nlargest(6, "ratio")[["state", "county_name", "y4", "fema", "ratio"]].round(2).to_string())
print("\nbelow FEMA:\n", big.nsmallest(6, "ratio")[["state", "county_name", "y4", "fema", "ratio"]].round(2).to_string())
