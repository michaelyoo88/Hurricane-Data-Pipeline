"""Step 01b - Housing units, population and building value for every county, every year.

FEMA gives one building value per county (a snapshot). To get a value for each
year, we use Census housing-unit counts:
    value per housing unit = FEMA building value / housing units in 2020
    building value in year Y = value per housing unit x housing units in year Y
Assumptions: value per unit stays fixed (in 2023 dollars), and non-housing
buildings grow at the same rate as housing.

Sources (Census estimates, July 1 of each year). Housing and population come in the
same three layouts:
    *_2000_2010.csv    2000-2009, has county codes (HUEST_2000 / POPESTIMATE2000 ...)
    *_2010_2020.xlsx   2010-2019, names only (".Autauga County, Alabama")
    *_2020_2025.xlsx   2020-2025, names only; Connecticut uses planning regions

Out: clean/census_by_year.csv  (fips, year, housing_units, population, bldg_value_usd2023)
"""
import pandas as pd
from common import *

SOURCES = {  # measure: (2000s csv, column prefix in that csv, 2010s xlsx, 2020s xlsx)
    "housing_units": ("Housing_2000_2010.csv", "HUEST_", "Housing_2010_2020.xlsx", "Housing_2020_2025.xlsx"),
    "population": ("population2000_2010.csv", "POPESTIMATE", "population2010_2020.xlsx", "population2020_2025.xlsx"),
}

# ---- Name -> county code lookup (the xlsx files only have names) ----
nri = pd.read_csv(SOURCE / "nri_counties.csv", dtype=str, encoding="utf-8-sig")
nri["key"] = (nri["County Name"] + " " + nri["County Type"] + ", " + nri["State Name"]).str.lower()
states = pd.read_csv(SOURCE / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
states = states[["STATEFP", "STNAME"]].drop_duplicates()


def read_2000s_csv(path, prefix):
    """2000-2009 CSV (wide: one column per year), counties only."""
    d = pd.read_csv(SOURCE / path, dtype=str, encoding="latin-1")
    d = d[d.SUMLEV.str.zfill(3) == "050"].copy()
    d["fips"] = d.STATE.str.zfill(2) + d.COUNTY.str.zfill(3)
    cols = {f"{prefix}{y}": y for y in range(2000, 2010)}
    return d, d[["fips"] + list(cols)].rename(columns=cols).melt(id_vars="fips", var_name="year", value_name="value")


h2000, _ = read_2000s_csv("Housing_2000_2010.csv", "HUEST_")
h2000 = h2000.merge(states, left_on=h2000.STATE.str.zfill(2), right_on="STATEFP")
h2000["key"] = (h2000.CTYNAME + ", " + h2000.STNAME).str.lower()
lookup = pd.concat([nri[["key", "State-County FIPS Code"]].rename(columns={"State-County FIPS Code": "fips"}),
                    h2000[["key", "fips"]]]).drop_duplicates("key")


def read_census_xlsx(path, year_columns):
    """xlsx: a title block, then rows like '.Autauga County, Alabama' followed by numbers."""
    raw = pd.read_excel(SOURCE / path, header=None)
    # A few rows in the Census files are missing the leading ".", so look for "Name, State" + a number.
    is_county = raw[0].astype(str).str.contains(", ") & pd.to_numeric(raw[2], errors="coerce").notna()
    rows = raw[is_county].copy()
    rows["key"] = rows[0].str.lstrip(".").str.lower()
    rows = rows.merge(lookup, on="key", how="left")
    assert rows.fips.notna().all(), f"a county name didn't match in {path}"
    wide = rows[["fips"] + list(year_columns)].rename(columns=year_columns)
    return wide.melt(id_vars="fips", var_name="year", value_name="value")


def split_ct(df):
    """Connecticut before 2020 uses old counties: split into planning regions by population share."""
    crosswalk = pd.read_csv(CLEAN / "ct_crosswalk.csv", dtype=str)
    crosswalk["share"] = crosswalk.share.astype(float)
    old = df.fips.isin(CT_OLD_COUNTIES)
    ct = df[old].merge(crosswalk, left_on="fips", right_on="old_fips")
    ct["value"] = ct.value * ct.share
    ct["fips"] = ct.region_fips
    return pd.concat([df[~old], ct[["fips", "year", "value"]]])


def read_measure(csv2000, prefix, xlsx2010, xlsx2020):
    """One measure (housing or population) -> fips, year, value for 2000-2025."""
    _, part1 = read_2000s_csv(csv2000, prefix)
    part2 = read_census_xlsx(xlsx2010, {c: 2008 + c for c in range(2, 12)})  # cols 2-11 = 2010-2019
    part3 = read_census_xlsx(xlsx2020, {c: 2018 + c for c in range(2, 8)})   # cols 2-7 = 2020-2025
    df = pd.concat([part1, part2, part3], ignore_index=True)
    df["year"] = df.year.astype(int)
    df["value"] = pd.to_numeric(df.value)
    # Bedford city, VA merged into Bedford County in 2013; Shannon County, SD was renamed Oglala Lakota in 2015
    df["fips"] = df.fips.replace({"51515": "51019", "46113": "46102"})
    df = split_ct(df)
    return df.groupby(["fips", "year"]).value.sum()


census = pd.concat({m: read_measure(*files) for m, files in SOURCES.items()}, axis=1).reset_index()

# ---- Keep our 3,109 counties, 2000-2024 ----
counties = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str}, usecols=["fips", "bldg_value_usd2023"])
census = census[census.fips.isin(counties.fips) & census.year.between(2000, 2024)]
assert len(census) == len(counties) * 25 and census.notna().all().all(), "missing county-years"

# ---- Building value per year ----
units_2020 = census[census.year == 2020].set_index("fips").housing_units
per_unit = counties.set_index("fips").bldg_value_usd2023 / units_2020
census["bldg_value_usd2023"] = (census.fips.map(per_unit) * census.housing_units).round(0)
census[["housing_units", "population"]] = census[["housing_units", "population"]].round(0).astype(int)

census.sort_values(["fips", "year"]).to_csv(CLEAN / "census_by_year.csv", index=False)
print(f"01b census_by_year: {len(census):,} county-years (2000-2024), none missing; "
      f"median value per housing unit ${per_unit.median():,.0f}; "
      f"US population 2000 {census[census.year == 2000].population.sum() / 1e6:.0f}M -> "
      f"2024 {census[census.year == 2024].population.sum() / 1e6:.0f}M")
