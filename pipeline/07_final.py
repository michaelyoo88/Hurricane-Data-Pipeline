"""Step 07: publish the final dataset (star schema) to final/.

  storm_county.csv   FACT       one row per storm x county within 500 km
  storms.csv         DIMENSION  one row per storm, 1975-2025
  counties.csv       DIMENSION  one row per county + FEMA NRI hurricane benchmark
  storm_tracks.csv   DETAIL     one row per storm x hour (for wind-field upgrades)
  data_dictionary.csv           every column of every table

Keys: storm_id (e.g. AL092022), fips (5-character text). Join with
pipeline/load_data.py -> model_table().
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("07_final")
FINAL = ROOT / "final"
FINAL.mkdir(exist_ok=True)
U = "usd2023"

# ---------------- counties (dimension) --------------------------------------
# Start from the clean county table, add the _usd2023 suffix to money columns,
# and copy FEMA's hurricane columns. nri_map pairs each source column with
# (new name, is_money); money columns are converted to 2023 dollars. Note the
# tuple unpacking in the loop: `for src, (dst, money) in nri_map.items()`.
c = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str})
f_nri = cpi_factors()[NRI_DOLLAR_YEAR]
H = "nri_hrcn_"
nri_map = {  # NRI column (snake) -> final name, is_money
    "number_of_events": ("nri_hrcn_n_events", False),
    "annualized_frequency": ("nri_hrcn_afreq", False),
    "exposure_building_value": (f"nri_hrcn_exp_bldg_{U}", True),
    "exposure_population": ("nri_hrcn_exp_pop", False),
    "exposure_agriculture_value": (f"nri_hrcn_exp_ag_{U}", True),
    "historic_loss_ratio_buildings": ("nri_hrcn_hlr_bldg", False),
    "historic_loss_ratio_population": ("nri_hrcn_hlr_pop", False),
    "historic_loss_ratio_agriculture": ("nri_hrcn_hlr_ag", False),
    "expected_annual_loss_building_value": (f"nri_hrcn_eal_bldg_{U}", True),
    "expected_annual_loss_population": ("nri_hrcn_eal_pop", False),
    "expected_annual_loss_population_equivalence": (f"nri_hrcn_eal_pope_{U}", True),
    "expected_annual_loss_agriculture_value": (f"nri_hrcn_eal_ag_{U}", True),
    "expected_annual_loss_total": (f"nri_hrcn_eal_total_{U}", True),
    "expected_annual_loss_score": ("nri_hrcn_eal_score", False),
    "expected_annual_loss_rating": ("nri_hrcn_eal_rating", False),
}
counties = c[["fips", "state", "county_name", "pop_lat", "pop_lon", "population"]].copy()
counties[f"bldg_value_{U}"] = c.bldg_value_2023
counties[f"ag_value_{U}"] = c.ag_value_2023
counties["area_sqmi"] = c.area_sqmi
counties["density_per_sqmi"] = c.density.round(2)
counties["coastal"] = c.coastal
for src, (dst, money) in nri_map.items():
    counties[dst] = c[H + src] * (f_nri if money else 1)

# ---------------- storms (dimension) ----------------------------------------
t6 = pd.read_csv(CLEAN / "tracks_6h.csv", parse_dates=["time_utc"])
sc = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str})
ls = pd.read_csv(CLEAN / "losses.csv", dtype={"fips": str})


# Saffir-Simpson category from peak wind in knots. pd.cut puts each value in a
# bin: (-1, 33] TD, (33, 63] TS, (63, 82] Cat 1, and so on up to Cat 5.
def category(kt):
    bins = [-1, 33, 63, 82, 95, 112, 136, 1e9]
    return pd.cut(kt, bins, labels=["TD", "TS", "1", "2", "3", "4", "5"]).astype(str)


# One row per storm: basic facts from the 6-hourly track (name, start / end,
# peak wind), then two groupby .agg() summaries merged on:
#   near  how many counties it reached and how close it came. This is named
#         aggregation syntax: new_col=(source_col, function)
#   obs   total observed losses
# Storms in labeled years with no losses get 0; earlier years stay blank.
g = t6.groupby("storm_id")
storms = pd.DataFrame({
    "name": g.name.first(), "year": g.year.first(),
    "start_utc": g.time_utc.min(), "end_utc": g.time_utc.max(),
    "peak_vmax_ms": g.vmax_ms.max().round(2), "min_pmin_mb": g.pmin_mb.min(),
}).reset_index()
storms["peak_category"] = category(g.vmax_kt.max()).to_numpy()
near = sc.groupby("storm_id").agg(
    n_counties_500km=("fips", "size"),
    n_counties_100km=("hours_within_100km", lambda s: int((s > 0).sum())),
    closest_county_km=("min_dist_km", "min"))
obs = ls.groupby("storm_id").agg(
    obs_prop_dmg=("prop_dmg_2023", "sum"), obs_crop_dmg=("crop_dmg_2023", "sum"),
    obs_deaths=("deaths_direct", "sum"))
obs.columns = [f"obs_prop_dmg_{U}", f"obs_crop_dmg_{U}", "obs_deaths_direct"]
storms = storms.merge(near, on="storm_id", how="left").merge(obs, on="storm_id", how="left")
storms[["n_counties_500km", "n_counties_100km"]] = storms[["n_counties_500km", "n_counties_100km"]].fillna(0).astype(int)
storms["has_labels"] = storms.year.between(LABEL_YEAR_MIN, LABEL_YEAR_MAX)
lab = storms.has_labels
for col in [f"obs_prop_dmg_{U}", f"obs_crop_dmg_{U}", "obs_deaths_direct"]:
    storms.loc[lab, col] = storms.loc[lab, col].fillna(0)
storms = storms.sort_values(["year", "start_utc"])

# ---------------- storm_county (fact) ---------------------------------------
# Start from storm_county, give columns their final names, and left-join the
# summed losses. Labeled years: missing loss = 0. Unlabeled years: forced to NaN
# (unknown, not zero). The merge with storms only sorts rows by storm start
# time; that helper column is dropped right after.
fact = sc.rename(columns={"vmax_peak_within_100km_ms": "vmax_peak_100km_ms",
                          "time_at_min": "time_at_min_utc",
                          "t_enter_500km": "enter_500km_utc",
                          "t_exit_500km": "exit_500km_utc"})
loss_cols = {"prop_dmg_2023": f"prop_dmg_{U}", "crop_dmg_2023": f"crop_dmg_{U}",
             "crop_dmg_area_2023": f"crop_dmg_area_{U}",
             "deaths_direct": "deaths_direct", "deaths_indirect": "deaths_indirect",
             "injuries_direct": "injuries_direct", "injuries_indirect": "injuries_indirect",
             "n_records": "n_loss_records", "n_ambiguous": "n_ambiguous_records"}
fact = fact.merge(ls[["storm_id", "fips"] + list(loss_cols)].rename(columns=loss_cols),
                  on=["storm_id", "fips"], how="left")
fact["has_labels"] = fact.year.between(LABEL_YEAR_MIN, LABEL_YEAR_MAX)
L = list(loss_cols.values())
fact.loc[fact.has_labels, L] = fact.loc[fact.has_labels, L].fillna(0)
fact.loc[~fact.has_labels, L] = np.nan
fact = fact[["storm_id", "fips", "year", "month", "has_labels", "min_dist_km",
             "vmax_at_min_ms", "vmax_peak_100km_ms", "hours_within_100km",
             "time_at_min_utc", "enter_500km_utc", "exit_500km_utc"] + L]
fact = fact.merge(storms[["storm_id", "start_utc"]], on="storm_id").sort_values(
    ["start_utc", "fips"]).drop(columns="start_utc")

# ---------------- storm_tracks (detail) -------------------------------------
# Hourly tracks, keeping only the columns needed downstream.
tracks = pd.read_csv(CLEAN / "tracks_hourly.csv").rename(columns={"time_utc": "time_utc"})
tracks = tracks[["storm_id", "time_utc", "lat", "lon", "vmax_ms", "pmin_mb", "status"]]

# ---------------- checks -----------------------------------------------------
# Database-style integrity checks: each key is unique, and every key in the
# fact table exists in its dimension table (referential integrity).
log.check(counties.fips.is_unique, "counties: fips is a unique key")
log.check(storms.storm_id.is_unique, "storms: storm_id is a unique key")
log.check(not fact.duplicated(["storm_id", "fips"]).any(), "storm_county: (storm_id, fips) is a unique key")
log.check(fact.fips.isin(counties.fips).all(), "storm_county.fips all found in counties")
log.check(fact.storm_id.isin(storms.storm_id).all(), "storm_county.storm_id all found in storms")
log.check(tracks.storm_id.isin(storms.storm_id).all(), "storm_tracks.storm_id all found in storms")
log.check(fact.loc[fact.has_labels, L].notna().all().all(), "labeled rows have no blank loss values")
log.check(np.isclose(fact[f"prop_dmg_{U}"].sum(), ls.prop_dmg_2023.sum()), "all matched loss dollars carried into the fact table")
log.check(counties.state.nunique() == 49, f"{counties.state.nunique()} states incl. DC (expect 49)")

# damage larger than the exposure it is divided by (kept, flagged for modelers)
ex = fact.merge(counties[["fips", f"bldg_value_{U}", f"ag_value_{U}"]], on="fips")
over_b = (ex[f"prop_dmg_{U}"] > ex[f"bldg_value_{U}"]).sum()
over_a = (ex[f"crop_dmg_{U}"] > ex[f"ag_value_{U}"]).sum()
log.write(f"NOTE  rows where property damage > building value: {over_b} "
          "(Rita 2005, Cameron Parish: damage includes infrastructure)")
log.write(f"NOTE  rows where crop damage > agriculture value: {over_a} "
          "(NOAA crop damage includes timber; NRI agriculture value does not)")

# ---------------- data dictionary -------------------------------------------
# For every column of every table, look up (unit, description, source):
# the table-specific entry first, else the shared ("*", col) entry. `or` falls
# through to the second lookup when the first returns None. Also records the
# % blank, and lists any column nobody documented.
from data_dictionary import DESCRIPTIONS
tables = {"storm_county": fact, "storms": storms, "counties": counties, "storm_tracks": tracks}
rows, missing = [], []
for tname, df in tables.items():
    for col in df.columns:
        d = DESCRIPTIONS.get((tname, col)) or DESCRIPTIONS.get(("*", col))
        if d is None:
            missing.append(f"{tname}.{col}")
            d = ("", "", "")
        unit, desc, source = d
        rows.append({"table": tname, "column": col, "dtype": str(df[col].dtype),
                     "unit": unit, "description": desc, "source": source,
                     "pct_blank": round(df[col].isna().mean() * 100, 1)})
log.check(not missing, f"every column documented ({len(missing)} missing: {missing[:5]})")

# --- Write every table and the data dictionary to final/ ---
for tname, df in tables.items():
    df.to_csv(FINAL / f"{tname}.csv", index=False)
    log.write(f"final/{tname}.csv: {len(df):,} rows x {df.shape[1]} columns")
pd.DataFrame(rows).to_csv(FINAL / "data_dictionary.csv", index=False)
log.write(f"final/data_dictionary.csv: {len(rows)} columns documented")
log.close()
