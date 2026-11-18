"""Step 05 - Match each damage record to the storm that caused it, then add up.

Damage records have a county and dates but no storm name. Rule: a storm is a
candidate if its center was within 500 km of that county at some point during the
record's dates (+/- 1 day for time zones). If several storms qualify, the closest
one wins and the record is flagged "ambiguous". If none qualify, it's "unmatched"
and left out.
Records of the extra event types (surge, rain flood, other wind, tornado; see EXTRA_TYPES)
must ALSO have the storm within 400 km and either tropical-storm wind at the county or
the storm within 100 km, so a winter nor'easter's coastal flood or an ordinary summer
flash flood isn't pinned on a distant hurricane. Per-source columns keep them separable.

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

# Every storm that reached the record's county during its window is a candidate.
# Extra-type records (every US flood / wind / tornado record since 1996, ~460k) are paired
# only with storm-county rows that already pass their tighter distance/wind rule, and only
# if their dates come near SOME storm at all; otherwise the pairing doesn't fit in memory.
# This does not change which matches are found.
SC_COLS = ["storm_id", "fips", "min_dist_km", "enter_500km_utc", "exit_500km_utc"]
close = (storm_county.min_dist_km <= EXTRA_MAX_KM) & ((storm_county.wind_est_ms >= TROPICAL_STORM_MS)
                                                     | (storm_county.hours_within_100km > 0))
spans = storm_county[close].groupby("storm_id").agg(start=("enter_500km_utc", "min"), end=("exit_500km_utc", "max"))
is_extra = records.source != "wind"
near_a_storm = pd.Series(False, index=records.index)
for start, end in zip(spans.start, spans.end):
    near_a_storm |= (records.window_start <= end) & (records.window_end >= start)

REC_COLS = ["record_id", "fips", "window_start", "window_end"]
pairs = pd.concat([
    records.loc[~is_extra, REC_COLS].merge(storm_county[SC_COLS], on="fips"),
    records.loc[is_extra & near_a_storm, REC_COLS].merge(storm_county.loc[close, SC_COLS], on="fips"),
], ignore_index=True)
overlaps = (pairs.enter_500km_utc <= pairs.window_end) & (pairs.exit_500km_utc >= pairs.window_start)
pairs = pairs[overlaps]

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
# Keep each extra source separable: property $ by source, plus crop $ and deaths from all extras
EXTRA_SOURCES = sorted(set(EXTRA_TYPES.values()))
matched["extra"] = matched.source != "wind"
for src in EXTRA_SOURCES:
    matched[f"prop_dmg_{src}_2023"] = matched.prop_dmg_2023.where(matched.source == src, 0)
matched["crop_dmg_extra_2023"] = matched.crop_dmg_2023.where(matched.extra, 0)
matched["deaths_direct_extra"] = matched.deaths_direct.where(matched.extra, 0)
AMOUNTS = ["prop_dmg_2023", "crop_dmg_2023", "crop_dmg_area_2023", "injuries_direct",
           "injuries_indirect", "deaths_direct", "deaths_indirect"]
losses = matched.groupby(["storm_id", "fips"]).agg(
    **{c: (c, "sum") for c in AMOUNTS},
    n_loss_records=("record_id", "size"),
    n_ambiguous_records=("ambiguous", "sum"),
    **{f"prop_dmg_{s}_2023": (f"prop_dmg_{s}_2023", "sum") for s in EXTRA_SOURCES},
    crop_dmg_extra_2023=("crop_dmg_extra_2023", "sum"),
    deaths_direct_extra=("deaths_direct_extra", "sum"),
    n_extra_records=("extra", "sum")).reset_index()
losses.to_csv(CLEAN / "losses.csv", index=False)
print(f"05 losses: {len(losses):,} storm-county pairs with damage, {losses.storm_id.nunique()} storms; "
      f"property ${losses.prop_dmg_2023.sum() / 1e9:.1f}B; from extra types: "
      + ", ".join(f"{s} ${losses[f'prop_dmg_{s}_2023'].sum() / 1e9:.1f}B" for s in EXTRA_SOURCES)
      + f"; extra crop ${losses.crop_dmg_extra_2023.sum() / 1e9:.1f}B, extra deaths {losses.deaths_direct_extra.sum():.0f}")
