"""Step 01 - The master county list.

One row per county in the contiguous US + DC (3,109 counties). FEMA's NRI table
is the master list because NRI is what we compare against. For each county we keep:
  - where its people live (Census center of population) -> used to measure storm distance
  - what is at risk (building value, farm value, population) -> "exposure"
  - FEMA's own hurricane numbers -> the benchmark

Out: clean/counties.csv
"""
import numpy as np
import pandas as pd
from common import *

nri = pd.read_csv(SOURCE / "nri_counties.csv", dtype=str, encoding="utf-8-sig")
nri["fips"] = nri["State-County FIPS Code"]
nri = nri[~nri.fips.str[:2].isin(DROP_STATES)]          # contiguous US + DC only


def number(col):
    return pd.to_numeric(nri[col], errors="coerce")


to_2023 = cpi_factors()[NRI_DOLLAR_YEAR]   # 1.0 while NRI_DOLLAR_YEAR is 2023

counties = pd.DataFrame({
    "fips": nri.fips,
    "state": nri["State Name Abbreviation"],
    "county_name": nri["County Name"],
    "population": number("Population (2020)"),
    "bldg_value_usd2023": number("Building Value ($)") * to_2023,
    "ag_value_usd2023": number("Agriculture Value ($)") * to_2023,
    "area_sqmi": number("Area (sq mi)"),
})
counties["density_per_sqmi"] = (counties.population / counties.area_sqmi).round(2)
# "Coastal" = FEMA gives the county some coastal flooding loss
counties["coastal"] = number("Coastal Flooding - Expected Annual Loss - Total").fillna(0) > 0

# FEMA's hurricane benchmark (blank = FEMA considers the county not exposed)
FEMA = {  # new name: (NRI column, is it money?)
    "nri_hrcn_n_events": ("Hurricane - Number of Events", False),
    "nri_hrcn_afreq": ("Hurricane - Annualized Frequency", False),
    "nri_hrcn_exp_bldg_usd2023": ("Hurricane - Exposure - Building Value", True),
    "nri_hrcn_exp_pop": ("Hurricane - Exposure - Population", False),
    "nri_hrcn_exp_ag_usd2023": ("Hurricane - Exposure - Agriculture Value", True),
    "nri_hrcn_hlr_bldg": ("Hurricane - Historic Loss Ratio - Buildings", False),
    "nri_hrcn_hlr_pop": ("Hurricane - Historic Loss Ratio - Population", False),
    "nri_hrcn_hlr_ag": ("Hurricane - Historic Loss Ratio - Agriculture", False),
    "nri_hrcn_eal_bldg_usd2023": ("Hurricane - Expected Annual Loss - Building Value", True),
    "nri_hrcn_eal_pop": ("Hurricane - Expected Annual Loss - Population", False),
    "nri_hrcn_eal_pope_usd2023": ("Hurricane - Expected Annual Loss - Population Equivalence", True),
    "nri_hrcn_eal_ag_usd2023": ("Hurricane - Expected Annual Loss - Agriculture Value", True),
    "nri_hrcn_eal_total_usd2023": ("Hurricane - Expected Annual Loss - Total", True),
    "nri_hrcn_eal_score": ("Hurricane - Expected Annual Loss Score", False),
}
for new, (old, is_money) in FEMA.items():
    counties[new] = number(old) * (to_2023 if is_money else 1)
counties["nri_hrcn_eal_rating"] = nri["Hurricane - Expected Annual Loss Rating"]

# Where each county's people live
cenpop = pd.read_csv(SOURCE / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
cenpop["fips"] = cenpop.STATEFP + cenpop.COUNTYFP
cenpop["pop_lat"] = cenpop.LATITUDE.astype(float)
cenpop["pop_lon"] = cenpop.LONGITUDE.astype(float)
cenpop["pop"] = cenpop.POPULATION.astype(float)

# Connecticut regions have no center in the 2020 file: average the old-county
# centers that feed each region, weighted by the people each one contributes
ct = pd.read_csv(CLEAN / "ct_crosswalk.csv", dtype=str)
ct["share"] = ct.share.astype(float)
ct = ct.merge(cenpop, left_on="old_fips", right_on="fips")
ct["w"] = ct["pop"] * ct.share
regions = ct.groupby("region_fips").apply(
    lambda g: pd.Series({"pop_lat": np.average(g.pop_lat, weights=g.w),
                         "pop_lon": np.average(g.pop_lon, weights=g.w)}),
    include_groups=False).reset_index().rename(columns={"region_fips": "fips"})

centers = pd.concat([cenpop[["fips", "pop_lat", "pop_lon"]], regions])
counties = counties.merge(centers, on="fips", how="left")
assert counties.pop_lat.notna().all(), "some counties have no center of population"
assert counties.fips.is_unique

first = ["fips", "state", "county_name", "pop_lat", "pop_lon"]
counties = counties[first + [c for c in counties.columns if c not in first]]
counties.to_csv(CLEAN / "counties.csv", index=False)
print(f"01 counties: {len(counties)} counties in {counties.state.nunique()} states incl. DC")
