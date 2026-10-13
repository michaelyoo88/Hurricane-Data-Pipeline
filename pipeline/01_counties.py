"""Step 01: master county table (NRI) + centers of population.

In : source/nri_counties.csv, source/cenpop2020_county.txt, source/cpi_monthly.csv,
     clean/ct_crosswalk.csv (optional)
Out: clean/counties.csv, clean/logs/01_fips_mismatches.csv
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("01_counties")

nri = pd.read_csv(SRC / "nri_counties.csv", dtype=str, encoding="utf-8-sig")
nri = nri.rename(columns={"State-County FIPS Code": "fips"})
log.write(f"NRI rows in: {len(nri)}")
out_scope = state_of(nri["fips"]).isin(EXCLUDED_STATE_FIPS)
log.write(f"Dropped outside contiguous US + DC: {out_scope.sum()} "
          f"({', '.join(sorted(nri.loc[out_scope, 'State Name Abbreviation'].unique()))})")
nri = nri[~out_scope].copy()

base = {"fips": "fips", "State Name Abbreviation": "state",
        "County Name": "county_name", "Population (2020)": "population",
        "Building Value ($)": "bldg_value_nominal",
        "Agriculture Value ($)": "ag_value_nominal", "Area (sq mi)": "area_sqmi"}
hurricane = {c: "nri_hrcn_" + snake(c[len("Hurricane - "):])
             for c in nri.columns if c.startswith("Hurricane - ")}
coastal_col = "Coastal Flooding - Expected Annual Loss - Total"

df = nri[list(base) + list(hurricane)].rename(columns={**base, **hurricane})
text_cols = {"fips", "state", "county_name"} | {c for c in df if c.endswith("rating")}
for c in df.columns.difference(list(text_cols)):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["coastal"] = pd.to_numeric(nri[coastal_col], errors="coerce").fillna(0).gt(0).values

f = cpi_factors()[NRI_DOLLAR_YEAR]
df["bldg_value_2023"] = df["bldg_value_nominal"] * f
df["ag_value_2023"] = df["ag_value_nominal"] * f
log.write(f"NRI dollars assumed {NRI_DOLLAR_YEAR}; CPI factor to {BASE_DOLLAR_YEAR} = {f:.4f}")
df["density"] = df["population"] / df["area_sqmi"]

# Centers of population
cp = pd.read_csv(SRC / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
cp["fips"] = cp["STATEFP"] + cp["COUNTYFP"]
cp["pop_lat"] = cp["LATITUDE"].astype(float)
cp["pop_lon"] = cp["LONGITUDE"].astype(float)
cp["cp_pop"] = cp["POPULATION"].astype(float)

if CT_CROSSWALK.exists():
    cw = pd.read_csv(CT_CROSSWALK, dtype={"old_fips": str, "region_fips": str})
    ct = cw.merge(cp[["fips", "pop_lat", "pop_lon", "cp_pop"]],
                  left_on="old_fips", right_on="fips")
    ct["w"] = ct["cp_pop"] * ct["share"]
    reg = ct.groupby("region_fips").apply(
        lambda g: pd.Series({"pop_lat": np.average(g.pop_lat, weights=g.w),
                             "pop_lon": np.average(g.pop_lon, weights=g.w)}),
        include_groups=False).reset_index().rename(columns={"region_fips": "fips"})
    cp = pd.concat([cp[["fips", "pop_lat", "pop_lon"]], reg])
    log.write(f"Connecticut: {len(reg)} planning-region centers built from crosswalk")
else:
    log.write("WARN  Connecticut crosswalk missing: planning regions get no "
              "coordinates and are dropped (run 00_ct_crosswalk.py)")

cp = cp[["fips", "pop_lat", "pop_lon"]]
mism = pd.concat([
    pd.DataFrame({"fips": sorted(set(df.fips) - set(cp.fips)), "found_in": "NRI only"}),
    pd.DataFrame({"fips": sorted(set(cp.fips) - set(df.fips)), "found_in": "cenpop only"})])
mism = mism[~state_of(mism.fips).isin(EXCLUDED_STATE_FIPS)]
mism.to_csv(LOGS / "01_fips_mismatches.csv", index=False)
log.write(f"FIPS mismatches (in scope): {len(mism)} -> logs/01_fips_mismatches.csv")

df = df.merge(cp, on="fips", how="left")
no_xy = df["pop_lat"].isna()
log.write(f"Dropped counties with no coordinates: {no_xy.sum()} "
          f"({', '.join(df.loc[no_xy, 'fips'])})")
df = df[~no_xy]

first = ["fips", "state", "county_name", "pop_lat", "pop_lon", "population",
         "bldg_value_nominal", "bldg_value_2023", "ag_value_nominal",
         "ag_value_2023", "area_sqmi", "density", "coastal"]
df = df[first + [c for c in df.columns if c not in first]]

log.check(df["fips"].is_unique, "fips unique")
log.check(df["fips"].str.len().eq(5).all(), "every fips is 5 characters")
log.check(df["state"].nunique() == 49, f"{df['state'].nunique()} states incl. DC (expect 49)")
log.write(f"Rows out: {len(df)}; hurricane columns kept: {len(hurricane)}")
df.to_csv(CLEAN / "counties.csv", index=False)
log.close()
