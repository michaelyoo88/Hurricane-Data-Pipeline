"""EDA: where does NOAA record crop damage during hurricanes?

Our labels only take crop damage from "Hurricane/Tropical Storm" records (+ surge/coastal
flood). But NOAA offices may file a hurricane's crop losses under other event types
(flood, high wind, tornado, heavy rain...). This script:
  1. Pulls EVERY record with crop damage from the NY Fed file (1996-2025, Housing split,
     same as the pipeline), any event type.
  2. Attaches each to a hurricane with the same rule used for surge records:
     dates -1/+2 days overlap the storm's passage, storm within EXTRA_MAX_KM (400 km), and
     tropical-storm wind at the county or storm within 100 km.
  3. Breaks hurricane-related crop $ down by event type, state, storm and year, and
     shows how much our current labels capture vs. miss.
Read-only. Out: eda_output/crop_damage_*.csv, eda_output/crop_damage_by_type.png,
eda_output/crop_damage_summary.txt
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import SOURCE, CLEAN, DROP_STATES, WIND_GROUP, EXTRA_TYPES, EXTRA_MAX_KM, \
    TROPICAL_STORM_MS, cpi_factors   # noqa: E402

OUT = ROOT / "analysis" / "eda_output"
CACHE = ROOT / "analysis" / "cache" / "crop_records.csv"

# ---- 1. Every crop-damage record (awk is much faster than pandas on 1.3 GB) ----
if not CACHE.exists() or CACHE.stat().st_size < 1000:
    cmd = (f"awk -F',' 'NR==1 || ($3==\"Housing\" && $12+0 > 0)' "
           f"'{SOURCE / 'nyc_fed_losses.csv'}' > '{CACHE}'")
    subprocess.run(cmd, shell=True, check=True)
rec = pd.read_csv(CACHE, dtype={"fips": str, "event_id": str})
rec["fips"] = rec.fips.str.zfill(5)
rec = rec[~rec.fips.str[:2].isin(DROP_STATES)].copy()
rec["year"] = rec.begin_date.str[:4].astype(int)
rec = rec[rec.year <= 2024]
rec["crop_2023"] = rec.damages_crops * rec.year.map(cpi_factors())
rec["record_id"] = rec.event_id + "_" + rec.fips
rec["in_labels"] = (rec.disaster_group == WIND_GROUP) | rec.event_type.isin(list(EXTRA_TYPES))

# ---- 2. Attach to hurricanes ----
sc = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str},
                 usecols=["storm_id", "fips", "min_dist_km", "wind_est_ms", "hours_within_100km",
                          "enter_500km_utc", "exit_500km_utc"],
                 parse_dates=["enter_500km_utc", "exit_500km_utc"])
sc = sc[(sc.min_dist_km <= EXTRA_MAX_KM) & ((sc.wind_est_ms >= TROPICAL_STORM_MS) | (sc.hours_within_100km > 0))]
p = rec[["record_id", "fips", "begin_date", "end_date"]].merge(sc, on="fips")
p = p[(p.enter_500km_utc <= pd.to_datetime(p.end_date) + pd.Timedelta(days=2)) &
      (p.exit_500km_utc >= pd.to_datetime(p.begin_date) - pd.Timedelta(days=1))]
best = p.sort_values("min_dist_km").drop_duplicates("record_id")[["record_id", "storm_id"]]
rec = rec.merge(best, on="record_id", how="left")
rec["hurricane_related"] = rec.storm_id.notna()
names = pd.read_csv(ROOT / "final" / "storms.csv", usecols=["storm_id", "name"]).set_index("storm_id").name
states = pd.read_csv(ROOT / "final" / "counties.csv", dtype={"fips": str}, usecols=["fips", "state"]).set_index("fips").state
rec["state"] = rec.fips.map(states)

B = lambda s: s.sum() / 1e9   # noqa: E731
total = rec.crop_2023.sum()
hr = rec[rec.hurricane_related]

# ---- 3. Breakdowns ----
overall = rec.groupby(["disaster_group"]).agg(records=("record_id", "size"), crop_B=("crop_2023", B)) \
    .sort_values("crop_B", ascending=False)
overall["share"] = overall.crop_B / overall.crop_B.sum()

by_type = hr.groupby(["event_type", "in_labels"]).agg(records=("record_id", "size"), crop_B=("crop_2023", B),
                                                      storms=("storm_id", "nunique")) \
    .reset_index().sort_values("crop_B", ascending=False)
by_type["share_of_hurricane_crop"] = by_type.crop_B / by_type.crop_B.sum()
by_type.to_csv(OUT / "crop_damage_by_event_type.csv", index=False)

by_state = hr.groupby("state").agg(crop_B=("crop_2023", B),
                                   captured_B=("crop_2023", lambda s: s[hr.loc[s.index, "in_labels"]].sum() / 1e9)) \
    .sort_values("crop_B", ascending=False)
by_state["missed_B"] = by_state.crop_B - by_state.captured_B
by_state.to_csv(OUT / "crop_damage_by_state.csv")

by_storm = hr.groupby("storm_id").agg(crop_B=("crop_2023", B),
                                      captured_B=("crop_2023", lambda s: s[hr.loc[s.index, "in_labels"]].sum() / 1e9),
                                      main_missed_type=("event_type", lambda s: (hr.loc[s.index][~hr.loc[s.index, "in_labels"]]
                                                        .groupby("event_type").crop_2023.sum().idxmax()
                                                        if (~hr.loc[s.index, "in_labels"]).any() else "")))
by_storm["missed_B"] = by_storm.crop_B - by_storm.captured_B
by_storm.insert(0, "name", by_storm.index.map(names))
by_storm = by_storm.sort_values("crop_B", ascending=False)
by_storm.to_csv(OUT / "crop_damage_by_storm.csv")

by_year = hr.groupby("year").agg(crop_B=("crop_2023", B),
                                 captured_B=("crop_2023", lambda s: s[hr.loc[s.index, "in_labels"]].sum() / 1e9))
by_year["missed_B"] = by_year.crop_B - by_year.captured_B
by_year.to_csv(OUT / "crop_damage_by_year.csv")

# Checks: our captured share must equal what the pipeline actually has
pipeline_crop = pd.read_csv(ROOT / "final" / "storm_county.csv", usecols=["year", "crop_dmg_usd2023"])
pipeline_crop = pipeline_crop[pipeline_crop.year.between(1996, 2024)].crop_dmg_usd2023.sum() / 1e9
captured = B(hr.crop_2023[hr.in_labels])

pd.set_option("display.width", 200)
lines = [
    f"All NOAA crop-damage records 1996-2024 (contiguous US, Housing split): {len(rec):,}, ${total / 1e9:.1f}B (2023 $)",
    "", "By disaster group (all records, hurricane-related or not):", overall.round(3).to_string(), "",
    f"Hurricane-related (matched to a storm, EXTRA_MAX_KM rule): {len(hr):,} records, ${B(hr.crop_2023):.1f}B "
    f"({hr.crop_2023.sum() / total:.0%} of all US crop damage)",
    f"  captured by our labels (hurricane group + all EXTRA_TYPES): ${captured:.1f}B; "
    f"pipeline storm_county has ${pipeline_crop:.1f}B (step 05 rule for wind records is looser, so may differ)",
    f"  NOT captured (other event types during hurricanes): ${B(hr.crop_2023[~hr.in_labels]):.1f}B", "",
    "Hurricane-related crop $ by event type:", by_type.round(3).to_string(index=False), "",
    "By state (top 12):", by_state.head(12).round(2).to_string(), "",
    "By storm (top 15):", by_storm.head(15).round(2).to_string(), "",
    "By year:", by_year.round(2).to_string(),
]
text = "\n".join(lines)
(OUT / "crop_damage_summary.txt").write_text(text)
print(text)

# ---- Chart: hurricane-related crop $ by event type, captured vs not (two shades of one color) ----
g = by_type.groupby("event_type").crop_B.sum().sort_values().tail(10)
cap = by_type[by_type.in_labels].set_index("event_type").crop_B.reindex(g.index).fillna(0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(g.index, g.values, color="#2b6cb0", alpha=0.35, label="not in our labels")
ax.barh(g.index, cap.values, color="#2b6cb0", label="in our labels")
ax.set_xlabel("Crop damage during hurricanes, 1996-2024 ($B, 2023 $)")
ax.set_title("Which NOAA event types carry hurricane crop damage?", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "crop_damage_by_type.png", dpi=150)
