"""Step 04 - Recorded hurricane damage from the NY Fed dataset.

The NY Fed file (13.5M rows, 1.3 GB) has one row per weather event x county x
"weighting scheme". The same event appears 6 times, once for each way the Fed
split a forecast zone's damage across counties - so we must keep only ONE scheme
per number, or we'd count every dollar 6 times.

Steps: keep hurricane / tropical storm / tropical depression events, fix county
codes that lost their leading zero, move Connecticut to planning regions, keep the
contiguous US, and convert dollars to 2023 dollars.

Out: clean/loss_records.csv  (one row per event x county)
"""
import pandas as pd
from common import *

COLUMNS = ["event_id", "episode_id", "fips", "weight_type", "event_type", "disaster_group",
           "begin_date", "end_date", "damages_property", "damages_crops",
           "injuries_direct", "injuries_indirect", "fatalities_direct", "fatalities_indirect"]
WEIGHTS = [PROPERTY_WEIGHT, PEOPLE_WEIGHT, CROP_CHECK_WEIGHT]

# The file is too big to load at once, so read it in chunks and keep what we need
chunks = []
for chunk in pd.read_csv(SOURCE / "nyc_fed_losses.csv", usecols=COLUMNS, chunksize=2_000_000,
                         dtype={"fips": str, "event_id": str, "episode_id": str}):
    keep = (chunk.disaster_group == "Hurricane/Tropical Storm") & chunk.weight_type.isin(WEIGHTS)
    chunks.append(chunk[keep])
raw = pd.concat(chunks, ignore_index=True)
raw["fips"] = raw.fips.str.zfill(5)          # "1001" -> "01001"

# One table per weighting scheme, then join them side by side
key = ["event_id", "fips"]
info = ["episode_id", "event_type", "begin_date", "end_date"]
prop = raw[raw.weight_type == PROPERTY_WEIGHT][key + info + ["damages_property", "damages_crops"]]
area = raw[raw.weight_type == CROP_CHECK_WEIGHT][key + ["damages_crops"]]
people = raw[raw.weight_type == PEOPLE_WEIGHT][key + ["injuries_direct", "injuries_indirect",
                                                      "fatalities_direct", "fatalities_indirect"]]
records = (prop.merge(area, on=key, how="left", suffixes=("", "_area"))
               .merge(people, on=key, how="left"))
records = records.rename(columns={
    "damages_property": "prop_dmg_nominal", "damages_crops": "crop_dmg_nominal",
    "damages_crops_area": "crop_dmg_area_nominal",
    "fatalities_direct": "deaths_direct", "fatalities_indirect": "deaths_indirect"})
AMOUNTS = ["prop_dmg_nominal", "crop_dmg_nominal", "crop_dmg_area_nominal",
           "injuries_direct", "injuries_indirect", "deaths_direct", "deaths_indirect"]
records[AMOUNTS] = records[AMOUNTS].fillna(0)

# Connecticut: split each old county's numbers across planning regions by population.
# Two old counties can feed the same region in one event, so add those up.
crosswalk = pd.read_csv(CLEAN / "ct_crosswalk.csv", dtype=str)
crosswalk["share"] = crosswalk.share.astype(float)
is_ct = records.fips.isin(CT_OLD_COUNTIES)
ct = records[is_ct].merge(crosswalk, left_on="fips", right_on="old_fips")
ct[AMOUNTS] = ct[AMOUNTS].mul(ct.share, axis=0)
ct["fips"] = ct.region_fips
ct = ct.groupby(key, as_index=False).agg({**{c: "first" for c in info}, **{c: "sum" for c in AMOUNTS}})
records = pd.concat([records[~is_ct], ct[records.columns]], ignore_index=True)

# Contiguous US only
records = records[~records.fips.str[:2].isin(DROP_STATES)]

# Inflation: convert each record to 2023 dollars using its own year
records["year"] = records.begin_date.str[:4].astype(int)
factor = records.year.map(cpi_factors())
for col in ["prop_dmg", "crop_dmg", "crop_dmg_area"]:
    records[f"{col}_2023"] = records[f"{col}_nominal"] * factor

records["record_id"] = records.event_id + "_" + records.fips
assert records.record_id.is_unique
records.to_csv(CLEAN / "loss_records.csv", index=False)
print(f"04 loss_records: {len(records):,} records, "
      f"${records.prop_dmg_2023.sum() / 1e9:.1f}B property damage (2023 $)")
