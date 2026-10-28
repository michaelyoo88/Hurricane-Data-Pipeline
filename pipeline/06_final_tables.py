"""Step 06 - Publish the tables we model on.

  final/storm_county.csv  MAIN TABLE: one row per storm x county within 500 km
                          (storm features + recorded damage)
  final/storms.csv        one row per storm (name, strength, totals)
  final/counties.csv      one row per county (location, exposure, FEMA benchmark)
  final/storm_tracks.csv  one row per storm per hour (kept for a better wind model later)

Join them on storm_id and fips (see load_data.py).

Damage columns: 0 = a year with damage records but none for this pair;
blank = a year before damage records exist (1975-1995). Never fill blanks with 0.
"""
import numpy as np
import pandas as pd
from common import *

counties = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str})
tracks_6h = pd.read_csv(CLEAN / "tracks_6h.csv", parse_dates=["time_utc"])
tracks = pd.read_csv(CLEAN / "tracks_hourly.csv")
storm_county = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str})
losses = pd.read_csv(CLEAN / "losses.csv", dtype={"fips": str})
DAMAGE = [c for c in losses.columns if c not in ("storm_id", "fips")]


def has_labels(year):
    return year.between(*LABEL_YEARS)


# ---- storm_county: storms x counties + their damage ----
fact = storm_county.merge(losses, on=["storm_id", "fips"], how="left")
fact.insert(4, "has_labels", has_labels(fact.year))
fact.loc[fact.has_labels, DAMAGE] = fact.loc[fact.has_labels, DAMAGE].fillna(0)
fact.loc[~fact.has_labels, DAMAGE] = np.nan
fact = fact.rename(columns={"prop_dmg_2023": "prop_dmg_usd2023", "crop_dmg_2023": "crop_dmg_usd2023",
                            "crop_dmg_area_2023": "crop_dmg_area_usd2023"})

# ---- storms: one row per storm ----
g = tracks_6h.groupby("storm_id")
storms = pd.DataFrame({
    "name": g.name.first(), "year": g.year.first(),
    "start_utc": g.time_utc.min(), "end_utc": g.time_utc.max(),
    "peak_vmax_ms": g.vmax_ms.max().round(2), "min_pmin_mb": g.pmin_mb.min(),
}).reset_index()
peak_kt = storms.peak_vmax_ms / KT_TO_MS
storms["peak_category"] = pd.cut(peak_kt, [-1, 33, 63, 82, 95, 112, 136, 999],
                                 labels=["TD", "TS", "1", "2", "3", "4", "5"]).astype(str)
reach = storm_county.groupby("storm_id").agg(
    n_counties_500km=("fips", "size"),
    n_counties_100km=("hours_within_100km", lambda h: (h > 0).sum()),
    closest_county_km=("min_dist_km", "min"))
totals = losses.groupby("storm_id").agg(
    obs_prop_dmg_usd2023=("prop_dmg_2023", "sum"),
    obs_crop_dmg_usd2023=("crop_dmg_2023", "sum"),
    obs_deaths_direct=("deaths_direct", "sum"))
storms = storms.merge(reach, on="storm_id", how="left").merge(totals, on="storm_id", how="left")
storms[["n_counties_500km", "n_counties_100km"]] = storms[["n_counties_500km", "n_counties_100km"]].fillna(0).astype(int)
storms["has_labels"] = has_labels(storms.year)
obs = ["obs_prop_dmg_usd2023", "obs_crop_dmg_usd2023", "obs_deaths_direct"]
storms.loc[storms.has_labels, obs] = storms.loc[storms.has_labels, obs].fillna(0)
storms = storms.sort_values("start_utc")

# ---- quick checks: keys are unique and every join finds its partner ----
assert counties.fips.is_unique and storms.storm_id.is_unique
assert not fact.duplicated(["storm_id", "fips"]).any()
assert fact.fips.isin(counties.fips).all() and fact.storm_id.isin(storms.storm_id).all()
assert np.isclose(fact.prop_dmg_usd2023.sum(), losses.prop_dmg_2023.sum())

fact.to_csv(FINAL / "storm_county.csv", index=False)
storms.to_csv(FINAL / "storms.csv", index=False)
counties.to_csv(FINAL / "counties.csv", index=False)
tracks.to_csv(FINAL / "storm_tracks.csv", index=False)
print(f"06 final: storm_county {fact.shape}, storms {storms.shape}, "
      f"counties {counties.shape}, storm_tracks {tracks.shape}")
