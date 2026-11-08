"""Step 01d - County stats by year: one row per county per year, 2000-2024.

Puts the yearly exposure numbers from steps 01b (Census) and 01c (USDA) in one
table, with no storm or damage columns, so anyone (models, dashboard) can use
the same numbers.

Two columns are computed here so everyone uses the same definition:
    farm_timber_value_usd2023 = farm sales + timber sales  (crop-model exposure weight)
    density_per_sqmi          = that year's population / land area

Out: final/county_stats_by_year.csv  (3,109 counties x 25 years = 77,725 rows)
"""
import pandas as pd
from common import *

counties = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str},
                       usecols=["fips", "state", "county_name", "area_sqmi"])
census = pd.read_csv(CLEAN / "census_by_year.csv", dtype={"fips": str})
ag = pd.read_csv(CLEAN / "ag_by_year.csv", dtype={"fips": str})

# Both inputs already have exactly one row per county per year, so "inner" loses nothing
# (the assert below checks that).
stats = counties.merge(census, on="fips").merge(ag, on=["fips", "year"])

stats["farm_timber_value_usd2023"] = stats.farm_value_usd2023 + stats.forest_value_usd2023
stats["density_per_sqmi"] = (stats.population / stats.area_sqmi).round(2)

stats = stats[["fips", "state", "county_name", "year", "population", "housing_units",
               "density_per_sqmi", "bldg_value_usd2023", "farm_value_usd2023",
               "crop_value_usd2023", "forest_value_usd2023", "farm_timber_value_usd2023",
               "cropland_acres", "woodland_acres"]].sort_values(["fips", "year"])

# ---- Checks ----
n_years = 2024 - 2000 + 1
assert len(stats) == len(counties) * n_years, "a county-year went missing in the merge"
assert not stats.duplicated(["fips", "year"]).any(), "duplicate county-years"
assert stats.notna().all().all(), "blank values"
assert (stats.drop(columns=["fips", "state", "county_name"]) >= 0).all().all(), "negative values"
assert counties.area_sqmi.gt(0).all(), "zero land area would break density"

stats.to_csv(FINAL / "county_stats_by_year.csv", index=False)
y00, y24 = stats[stats.year == 2000], stats[stats.year == 2024]
print(f"01d county_stats_by_year: {len(stats):,} rows ({len(counties):,} counties x {n_years} years); "
      f"US population {y00.population.sum() / 1e6:.0f}M -> {y24.population.sum() / 1e6:.0f}M; "
      f"building value ${y00.bldg_value_usd2023.sum() / 1e12:.1f}T -> ${y24.bldg_value_usd2023.sum() / 1e12:.1f}T; "
      f"farm+timber ${y00.farm_timber_value_usd2023.sum() / 1e9:.0f}B -> ${y24.farm_timber_value_usd2023.sum() / 1e9:.0f}B")
