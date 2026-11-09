"""Step 05 - Match each damage record to the storm that caused it, then add up.

Damage records have a county and dates but no storm name. Rule: a storm is a
candidate if its center was within 500 km of that county at some point during the
record's dates (+/- 1 day for time zones). If several storms qualify, the closest
one wins and the record is flagged "ambiguous". If none qualify, it's "unmatched"
and left out.
Storm surge / coastal flood records (source = "surge") must ALSO have the storm within
200 km and either tropical-storm wind at the county or the storm within 100 km, so a
winter nor'easter's coastal flood isn't pinned on a distant hurricane.

Out: clean/loss_match.csv  (one row per damage record: which storm, and how sure)
     clean/losses.csv      (matched damage added up per storm x county)
"""
import pandas as pd
from common import *

records = pd.read_csv(CLEAN / "loss_records.csv", dtype={"fips": str, "event_id": str})
storm_county = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str},
                           parse_dates=["enter_500km_utc", "exit_500km_utc"])

# Each record's date window
records["window_start"] = pd.to_datetime(records.begin_date) - pd.Timedelta(days=1)
records["window_end"] = pd.to_datetime(records.end_date) + pd.Timedelta(days=2)

# Every storm that reached the record's county during its window is a candidate
pairs = records[["record_id", "fips", "window_start", "window_end"]].merge(
    storm_county[["storm_id", "fips", "min_dist_km", "wind_est_ms", "hours_within_100km",
                  "enter_500km_utc", "exit_500km_utc"]],
    on="fips")
overlaps = (pairs.enter_500km_utc <= pairs.window_end) & (pairs.exit_500km_utc >= pairs.window_start)
is_surge = pairs.record_id.map(records.set_index("record_id").source) == "surge"
close = (pairs.min_dist_km <= SURGE_MAX_KM) & ((pairs.wind_est_ms >= TROPICAL_STORM_MS)
                                               | (pairs.hours_within_100km > 0))
pairs = pairs[overlaps & (~is_surge | close)]

closest = pairs.sort_values("min_dist_km").drop_duplicates("record_id")
match = records[["record_id", "source"]].merge(closest[["record_id", "storm_id", "min_dist_km"]],
                                     on="record_id", how="left")
match["n_candidates"] = match.record_id.map(pairs.groupby("record_id").size()).fillna(0).astype(int)
match["match_flag"] = "unique"
match.loc[match.n_candidates > 1, "match_flag"] = "ambiguous"
match.loc[match.n_candidates == 0, "match_flag"] = "unmatched"
match.to_csv(CLEAN / "loss_match.csv", index=False)

# How well did matching go? (share of records and share of dollars)
check = match.merge(records[["record_id", "prop_dmg_2023"]], on="record_id")
summary = check.groupby(["source", "match_flag"]).agg(records=("record_id", "size"), dollars=("prop_dmg_2023", "sum"))
print("05 matching (share of each source's records and $):\n",
      (summary / summary.groupby(level="source").transform("sum")).round(3))

# Add up matched damage per storm x county
matched = records.merge(match[match.match_flag != "unmatched"].drop(columns="source"), on="record_id")
matched["ambiguous"] = matched.match_flag == "ambiguous"
matched["surge"] = matched.source == "surge"
matched["prop_dmg_surge_2023"] = matched.prop_dmg_2023.where(matched.surge, 0)
AMOUNTS = ["prop_dmg_2023", "crop_dmg_2023", "crop_dmg_area_2023", "injuries_direct",
           "injuries_indirect", "deaths_direct", "deaths_indirect"]
losses = matched.groupby(["storm_id", "fips"]).agg(
    **{c: (c, "sum") for c in AMOUNTS},
    n_loss_records=("record_id", "size"),
    n_ambiguous_records=("ambiguous", "sum"),
    prop_dmg_surge_2023=("prop_dmg_surge_2023", "sum"),
    n_surge_records=("surge", "sum")).reset_index()
losses.to_csv(CLEAN / "losses.csv", index=False)
print(f"05 losses: {len(losses):,} storm-county pairs with damage, {losses.storm_id.nunique()} storms; "
      f"property ${losses.prop_dmg_2023.sum() / 1e9:.1f}B of which surge/coastal flood "
      f"${losses.prop_dmg_surge_2023.sum() / 1e9:.1f}B")
