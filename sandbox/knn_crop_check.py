"""Round 7, part 3: KNN on CROP damage, checked against an outside answer key (USDA crop insurance).

NOAA records crop damage in few of the places where crop insurance paid hurricane claims. Does KNN find them?
  flag   a storm x county with $0 recorded crop damage, farm value > 0, tropical-storm wind (17.5+ m/s), and at
         least half of its 10 most similar counties hit by the same storm recorded crop damage
  truth  USDA RMA hurricane / tropical depression indemnity paid in that county, in the storm's year and month
         (analysis/cache/rma_indemnities_2000_2024.csv.gz, from analysis/rma_crop_eda.py)
Out: results/knn_crop_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from knn_explore import FEATS, knn_stats
from lib import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
df = load()
df["log_value_pc"] = np.log(df.bldg_value / df.population)

# ---- RMA hurricane indemnity matched to storm x county by year and month ----
rma = pd.read_csv(ROOT / "analysis" / "cache" / "rma_indemnities_2000_2024.csv.gz",
                  dtype={"state_code": str, "county_code": str, "cause_code": str}, usecols=["state_code", "county_code", "cause_code", "year_loss", "month_loss", "indemnity"])
rma = rma[rma.cause_code.isin(["92", "67", "63"])]
rma["fips"] = rma.state_code.str.zfill(2) + rma.county_code.str.zfill(3)
rma = rma.groupby(["fips", "year_loss", "month_loss"]).indemnity.sum().reset_index()
sc = pd.read_csv(ROOT / "final" / "storm_county.csv", dtype={"fips": str}, usecols=["storm_id", "fips", "enter_500km_utc", "exit_500km_utc"],
                 parse_dates=["enter_500km_utc", "exit_500km_utc"])
months = pd.concat([sc.assign(month=sc.enter_500km_utc.dt.month, year=sc.enter_500km_utc.dt.year),
                    sc.assign(month=sc.exit_500km_utc.dt.month, year=sc.exit_500km_utc.dt.year)]).drop_duplicates(["storm_id", "fips", "month"])
paid = months.merge(rma, left_on=["fips", "year", "month"], right_on=["fips", "year_loss", "month_loss"]) \
    .groupby(["storm_id", "fips"]).indemnity.sum()
df["rma_paid"] = df.set_index(["storm_id", "fips"]).index.map(paid).fillna(0).to_numpy() > 0

nb = knn_stats(df, "crop_usd")
farm = (df.farm_timber_value > 0).to_numpy()
zero = (df.crop_usd == 0).to_numpy()
rows = []
for wmin in (17.5, 25.0):
    for smin in (0.3, 0.5, 0.7):
        pool = farm & zero & (df.wind_ms >= wmin).to_numpy() & nb.nb_share.notna().to_numpy()
        flag = pool & (nb.nb_share >= smin).to_numpy()
        rows.append({"min_wind": wmin, "min_neighbor_share": smin, "zeros_in_pool": int(pool.sum()), "flagged": int(flag.sum()),
                     "flagged_with_insurance_claims": df.rma_paid[flag].mean() if flag.any() else np.nan,
                     "not_flagged_with_claims": df.rma_paid[pool & ~flag].mean(),
                     "lift": (df.rma_paid[flag].mean() / df.rma_paid[pool & ~flag].mean()) if flag.any() else np.nan,
                     "crop_$M_if_imputed_expected": ((nb.nb_share * nb.nb_median_pos).fillna(0) * df.farm_timber_value)[flag].sum() / 1e6})
out = pd.DataFrame(rows)
out.round(3).to_csv(HERE / "results" / "knn_crop_check.csv", index=False)
both = farm & (df.wind_ms >= 17.5).to_numpy()
print(f"storm x county rows with farms and 17.5+ m/s wind: {both.sum():,}; NOAA crop damage recorded in {(~zero[both]).mean():.1%}; "
      f"crop insurance paid in {df.rma_paid[both].mean():.1%}; insurance paid where NOAA has $0: {df.rma_paid[both & zero].mean():.1%}")
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
print(f"recorded crop damage 2000-2024: ${df.crop_usd.sum() / 1e9:.1f}B")
