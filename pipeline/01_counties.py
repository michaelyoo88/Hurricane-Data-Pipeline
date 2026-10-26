"""Step 01: master county table (NRI) + centers of population.

In : source/nri_counties.csv, source/cenpop2020_county.txt, source/cpi_monthly.csv,
     clean/ct_crosswalk.csv (optional)
Out: clean/counties.csv, clean/logs/01_fips_mismatches.csv
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("01_counties")

# --- 1. Load NRI and drop out-of-scope states ---
# out_scope is a boolean Series (True/False per row); ~out_scope flips it, and
# df[mask] keeps rows where the mask is True. .copy() makes a standalone table,
# which avoids pandas' SettingWithCopyWarning when columns are added later.
nri = pd.read_csv(SRC / "nri_counties.csv", dtype=str, encoding="utf-8-sig")
nri = nri.rename(columns={"State-County FIPS Code": "fips"})
log.write(f"NRI rows in: {len(nri)}")
out_scope = state_of(nri["fips"]).isin(EXCLUDED_STATE_FIPS)
log.write(f"Dropped outside contiguous US + DC: {out_scope.sum()} "
          f"({', '.join(sorted(nri.loc[out_scope, 'State Name Abbreviation'].unique()))})")
nri = nri[~out_scope].copy()

# --- 2. Choose and rename columns ---
# base: hand-picked columns (old name -> new name).
# hurricane: a dict comprehension that renames every "Hurricane - ..." column
# to nri_hrcn_<snake_case>. {**base, **hurricane} merges the two dicts.
# Non-text columns are converted to numbers; errors="coerce" turns values that
# can't be parsed into NaN instead of crashing. coastal = True if FEMA expects
# any coastal-flood loss in the county.
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

# --- 3. Convert dollars to 2023 dollars; add population density ---
f = cpi_factors()[NRI_DOLLAR_YEAR]
df["bldg_value_2023"] = df["bldg_value_nominal"] * f
df["ag_value_2023"] = df["ag_value_nominal"] * f
log.write(f"NRI dollars assumed {NRI_DOLLAR_YEAR}; CPI factor to {BASE_DOLLAR_YEAR} = {f:.4f}")
df["density"] = df["population"] / df["area_sqmi"]

# Centers of population
# Each county's population-weighted center (lat/lon) stands in for "where the
# county is" when measuring storm distance. FIPS = 2-digit state + 3-digit county.
cp = pd.read_csv(SRC / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
cp["fips"] = cp["STATEFP"] + cp["COUNTYFP"]
cp["pop_lat"] = cp["LATITUDE"].astype(float)
cp["pop_lon"] = cp["LONGITUDE"].astype(float)
cp["cp_pop"] = cp["POPULATION"].astype(float)

# --- 4. Connecticut: build planning-region centers from old counties ---
# Each region's center = weighted average of the old-county centers feeding it,
# weighted by population x share (from step 00). groupby().apply(lambda g: ...)
# runs the function once per region; returning a pd.Series makes each result
# one row. The new rows are appended with pd.concat.
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

# --- 5. Log FIPS codes found in one source but not the other ---
# Python set difference (a - b) gives the codes in a that are missing from b.
cp = cp[["fips", "pop_lat", "pop_lon"]]
mism = pd.concat([
    pd.DataFrame({"fips": sorted(set(df.fips) - set(cp.fips)), "found_in": "NRI only"}),
    pd.DataFrame({"fips": sorted(set(cp.fips) - set(df.fips)), "found_in": "cenpop only"})])
mism = mism[~state_of(mism.fips).isin(EXCLUDED_STATE_FIPS)]
mism.to_csv(LOGS / "01_fips_mismatches.csv", index=False)
log.write(f"FIPS mismatches (in scope): {len(mism)} -> logs/01_fips_mismatches.csv")

# --- 6. Attach coordinates; drop counties that have none ---
# how="left" keeps every NRI county; unmatched ones get NaN coordinates.
df = df.merge(cp, on="fips", how="left")
no_xy = df["pop_lat"].isna()
log.write(f"Dropped counties with no coordinates: {no_xy.sum()} "
          f"({', '.join(df.loc[no_xy, 'fips'])})")
df = df[~no_xy]

# --- 7. Reorder columns, run sanity checks, save ---
# The list comprehension appends every remaining column after the `first` ones.
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
