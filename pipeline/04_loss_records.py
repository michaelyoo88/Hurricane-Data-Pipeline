"""Step 04: NY Fed hurricane / tropical storm loss records, one row per event x county.

In : source/nyc_fed_losses.csv (1.3 GB, read in chunks), source/cpi_monthly.csv,
     clean/ct_crosswalk.csv (optional)
Out: clean/loss_records.csv
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("04_loss_records")
USE = ["event_id", "fips", "weight_type", "begin_date", "end_date", "geography",
       "event_type", "disaster_group", "episode_id", "damages_property",
       "damages_property_adj", "damages_crops", "injuries_direct",
       "injuries_indirect", "fatalities_direct", "fatalities_indirect"]
weights = [WEIGHT_PROPERTY, WEIGHT_PEOPLE, WEIGHT_SENSITIVITY]

parts, n_all = [], 0
for ch in pd.read_csv(SRC / "nyc_fed_losses.csv", usecols=USE, chunksize=2_000_000,
                      dtype={"fips": str, "event_id": str, "episode_id": str}):
    n_all += len(ch)
    parts.append(ch[(ch.disaster_group == "Hurricane/Tropical Storm")
                    & ch.weight_type.isin(weights)])
raw = pd.concat(parts, ignore_index=True)
log.write(f"NY Fed rows read: {n_all:,}; hurricane/tropical rows for weights "
          f"{weights}: {len(raw):,}")
short = raw.fips.str.len().lt(5).sum()
raw["fips"] = raw.fips.str.zfill(5)
log.write(f"FIPS codes missing their leading zero, fixed: {short:,}")

key = ["event_id", "fips"]
w = {k: raw[raw.weight_type == k].drop(columns="weight_type") for k in weights}
for k, v in w.items():
    log.check(not v.duplicated(key).any(), f"{k}: one row per event x county")
tot = {k: v.damages_property.sum() for k, v in w.items()}
log.check(np.allclose(list(tot.values()), tot[WEIGHT_PROPERTY], rtol=1e-6),
          "property totals identical across weight types (each splits the same events)")

hp = w[WEIGHT_PROPERTY]
df = hp[key + ["episode_id", "event_type", "geography", "begin_date", "end_date"]].copy()
df["prop_dmg_nominal"] = hp.damages_property.to_numpy()
df["crop_dmg_nominal"] = hp.damages_crops.to_numpy()
df["prop_dmg_nyfed_adj"] = hp.damages_property_adj.to_numpy()
area = w[WEIGHT_SENSITIVITY][key + ["damages_crops"]].rename(
    columns={"damages_crops": "crop_dmg_area_nominal"})
ppl = w[WEIGHT_PEOPLE][key + ["injuries_direct", "injuries_indirect",
                             "fatalities_direct", "fatalities_indirect"]].rename(
    columns={"fatalities_direct": "deaths_direct", "fatalities_indirect": "deaths_indirect"})
df = df.merge(area, on=key, how="left").merge(ppl, on=key, how="left")
AMOUNTS = ["prop_dmg_nominal", "crop_dmg_nominal", "crop_dmg_area_nominal",
           "prop_dmg_nyfed_adj", "injuries_direct", "injuries_indirect",
           "deaths_direct", "deaths_indirect"]
df[AMOUNTS] = df[AMOUNTS].fillna(0)

# Connecticut: old counties -> planning regions
ct = df.fips.isin(CT_OLD_COUNTIES)
if CT_CROSSWALK.exists():
    cw = pd.read_csv(CT_CROSSWALK, dtype={"old_fips": str, "region_fips": str})
    ctr = df[ct].merge(cw, left_on="fips", right_on="old_fips")
    ctr[AMOUNTS] = ctr[AMOUNTS].mul(ctr["share"], axis=0)
    ctr["fips"] = ctr["region_fips"]
    # two old counties can feed the same region in one event: sum them
    meta = ctr.groupby(key)[["episode_id", "event_type", "geography",
                            "begin_date", "end_date"]].first()
    ctr = meta.join(ctr.groupby(key)[AMOUNTS].sum()).reset_index()
    df = pd.concat([df[~ct], ctr[df.columns]], ignore_index=True)
    log.write(f"Connecticut: {ct.sum()} records re-allocated to planning regions")
else:
    log.write(f"WARN  Connecticut crosswalk missing: {ct.sum()} records "
              f"(${df.loc[ct, 'prop_dmg_nominal'].sum() / 1e6:,.0f}M nominal) dropped")
    df = df[~ct]

out = state_of(df.fips).isin(EXCLUDED_STATE_FIPS)
log.write(f"Dropped outside contiguous US + DC: {out.sum()} records "
          f"(${df.loc[out, 'prop_dmg_nominal'].sum() / 1e9:,.2f}B nominal property)")
df = df[~out].copy()

# Inflation to BASE_DOLLAR_YEAR
df["year"] = df.begin_date.str[:4].astype(int)
f = cpi_factors()
df["cpi_factor"] = df.year.map(f)
log.check(df.cpi_factor.notna().all(), "CPI available for every record year")
for c in ["prop_dmg", "crop_dmg", "crop_dmg_area"]:
    df[f"{c}_2023"] = df[f"{c}_nominal"] * df.cpi_factor

by_year = df.groupby("year")[["prop_dmg_2023", "prop_dmg_nyfed_adj"]].sum()
by_year = by_year[by_year.prop_dmg_nyfed_adj > 1e6]
ratio = by_year.prop_dmg_2023 / by_year.prop_dmg_nyfed_adj
log.check(ratio.max() / ratio.min() < 1.03,
          f"our CPI conversion tracks NY Fed's own adjustment (ratio {ratio.min():.3f}-"
          f"{ratio.max():.3f} across years; constant = same method, different base year)")

df["record_id"] = df.event_id + "_" + df.fips
log.check(df.record_id.is_unique, "record_id unique")
log.write(f"Records out: {len(df):,}; years {df.year.min()}-{df.year.max()}; "
          f"property ${df.prop_dmg_2023.sum() / 1e9:,.1f}B, crops "
          f"${df.crop_dmg_2023.sum() / 1e9:,.1f}B ({BASE_DOLLAR_YEAR} dollars); "
          f"deaths {df[['deaths_direct', 'deaths_indirect']].sum().sum():,.0f}")
log.write("By event type (records, property $B):")
for et, g in df.groupby("event_type"):
    log.write(f"    {et:20s} {len(g):6,}  {g.prop_dmg_2023.sum() / 1e9:8.2f}")
cols = ["record_id", "event_id", "episode_id", "fips", "event_type", "geography",
        "begin_date", "end_date", "year", "prop_dmg_nominal", "prop_dmg_2023",
        "crop_dmg_nominal", "crop_dmg_2023", "crop_dmg_area_nominal",
        "crop_dmg_area_2023", "injuries_direct", "injuries_indirect",
        "deaths_direct", "deaths_indirect", "prop_dmg_nyfed_adj", "cpi_factor"]
df[cols].to_csv(CLEAN / "loss_records.csv", index=False)
log.close()
