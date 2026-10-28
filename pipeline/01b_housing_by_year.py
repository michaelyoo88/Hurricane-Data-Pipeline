"""Step 01b - Housing units and building value for every county, every year.

FEMA gives one building value per county (a snapshot). To get a value for each
year, we use Census housing-unit counts:
    value per housing unit = FEMA building value / housing units in 2020
    building value in year Y = value per housing unit x housing units in year Y
Assumptions: value per unit stays fixed (in 2023 dollars), and non-housing
buildings grow at the same rate as housing.

Sources (Census housing unit estimates, July 1 of each year):
    Housing_2000_2010.csv   2000-2009, has county codes
    Housing_2010_2020.xlsx  2010-2019, names only ("Autauga County, Alabama")
    Housing_2020_2025.xlsx  2020-2025, names only; Connecticut uses planning regions

Out: clean/housing_by_year.csv  (fips, year, housing_units, bldg_value_usd2023)
"""
import pandas as pd
from common import *

# ---- Name -> county code lookup (the xlsx files only have names) ----
nri = pd.read_csv(SOURCE / "nri_counties.csv", dtype=str, encoding="utf-8-sig")
nri["key"] = (nri["County Name"] + " " + nri["County Type"] + ", " + nri["State Name"]).str.lower()
states = pd.read_csv(SOURCE / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
states = states[["STATEFP", "STNAME"]].drop_duplicates()

h2000 = pd.read_csv(SOURCE / "Housing_2000_2010.csv", dtype=str, encoding="latin-1")
h2000 = h2000[h2000.SUMLEV == "050"].merge(states, left_on="STATE", right_on="STATEFP")  # counties only
h2000["fips"] = h2000.STATE + h2000.COUNTY
h2000["key"] = (h2000.CTYNAME + ", " + h2000.STNAME).str.lower()

lookup = pd.concat([nri[["key", "State-County FIPS Code"]].rename(columns={"State-County FIPS Code": "fips"}),
                    h2000[["key", "fips"]]]).drop_duplicates("key")

# ---- 2000-2009 (CSV, wide: one column per year) ----
cols = {f"HUEST_{y}": y for y in range(2000, 2010)}
part1 = (h2000[["fips"] + list(cols)].rename(columns=cols)
         .melt(id_vars="fips", var_name="year", value_name="housing_units"))


# ---- 2010-2019 and 2020-2025 (xlsx: a title block, then rows like ".Autauga County, Alabama") ----
def read_census_xlsx(path, year_columns):
    raw = pd.read_excel(path, header=None)
    # County rows look like ".Autauga County, Alabama" with numbers after them. A few rows
    # in the Census files are missing the leading ".", so look for "Name, State" + a number.
    is_county = raw[0].astype(str).str.contains(", ") & pd.to_numeric(raw[2], errors="coerce").notna()
    rows = raw[is_county].copy()
    rows["key"] = rows[0].str.lstrip(".").str.lower()
    rows = rows.merge(lookup, on="key", how="left")
    assert rows.fips.notna().all(), "a county name didn't match"
    wide = rows[["fips"] + list(year_columns)].rename(columns=year_columns)
    return wide.melt(id_vars="fips", var_name="year", value_name="housing_units")


part2 = read_census_xlsx(SOURCE / "Housing_2010_2020.xlsx", {c: 2008 + c for c in range(2, 12)})  # cols 2-11 = 2010-2019
part3 = read_census_xlsx(SOURCE / "Housing_2020_2025.xlsx", {c: 2018 + c for c in range(2, 8)})   # cols 2-7 = 2020-2025

housing = pd.concat([part1, part2, part3], ignore_index=True)
housing["year"] = housing.year.astype(int)
housing["housing_units"] = pd.to_numeric(housing.housing_units)

# ---- County changes since 2000 ----
# Bedford city, VA merged into Bedford County in 2013; Shannon County, SD was renamed Oglala Lakota in 2015
housing["fips"] = housing.fips.replace({"51515": "51019", "46113": "46102"})
# Connecticut before 2020 uses old counties: split them into planning regions by population share
crosswalk = pd.read_csv(CLEAN / "ct_crosswalk.csv", dtype=str)
crosswalk["share"] = crosswalk.share.astype(float)
old_ct = housing.fips.isin(CT_OLD_COUNTIES)
ct = housing[old_ct].merge(crosswalk, left_on="fips", right_on="old_fips")
ct["housing_units"] = ct.housing_units * ct.share
ct["fips"] = ct.region_fips
housing = pd.concat([housing[~old_ct], ct[["fips", "year", "housing_units"]]])
housing = housing.groupby(["fips", "year"], as_index=False).housing_units.sum()

# ---- Keep our 3,109 counties, 2000-2024 ----
counties = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str},
                       usecols=["fips", "bldg_value_usd2023"])
housing = housing[housing.fips.isin(counties.fips) & housing.year.between(2000, 2024)]
missing = len(counties) * 25 - len(housing)

# ---- Building value per year ----
units_2020 = housing[housing.year == 2020].set_index("fips").housing_units
per_unit = counties.set_index("fips").bldg_value_usd2023 / units_2020
housing["bldg_value_usd2023"] = (housing.fips.map(per_unit) * housing.housing_units).round(0)
housing["housing_units"] = housing.housing_units.round(0).astype(int)

housing.sort_values(["fips", "year"]).to_csv(CLEAN / "housing_by_year.csv", index=False)
print(f"01b housing_by_year: {len(housing):,} county-years (2000-2024), {missing} missing; "
      f"median value per housing unit ${per_unit.median():,.0f}")
