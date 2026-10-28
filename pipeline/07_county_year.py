"""Step 07 - One row per county per year, 1975-2025 (158,559 rows).

For each county and year: the strongest wind any storm brought, whether that was a
hurricane-force hit, how often the county gets hit (its annual hit probability),
and the damage recorded that year.

Note: p_hurricane_hit uses all 51 years, so it includes the year being described.
For modeling, recompute it from the training years only (otherwise the model sees
the answer).

Out: final/county_year.csv
"""
import numpy as np
import pandas as pd
from common import *

storm_county = pd.read_csv(FINAL / "storm_county.csv", dtype={"fips": str})
counties = pd.read_csv(FINAL / "counties.csv", dtype={"fips": str},
                       usecols=["fips", "state", "county_name"])
sc = storm_county.assign(
    ts_force=storm_county.wind_est_ms >= TROPICAL_STORM_MS,
    hurricane_force=storm_county.wind_est_ms >= HURRICANE_MS)

# Storms -> one row per county per year
yearly = sc.groupby(["fips", "year"]).agg(
    max_wind_ms=("wind_est_ms", "max"),
    n_storms_500km=("storm_id", "size"),
    n_storms_ts_force=("ts_force", "sum"),
    n_storms_hurricane_force=("hurricane_force", "sum"),
    prop_dmg_usd2023=("prop_dmg_usd2023", lambda x: x.sum(min_count=1)),
    crop_dmg_usd2023=("crop_dmg_usd2023", lambda x: x.sum(min_count=1)),
    deaths_direct=("deaths_direct", lambda x: x.sum(min_count=1)),
    injuries_direct=("injuries_direct", lambda x: x.sum(min_count=1)),
).reset_index()

# Every county x every year, including quiet years when no storm came near
years = range(STORM_YEARS[0], STORM_YEARS[1] + 1)
grid = pd.MultiIndex.from_product([counties.fips, years], names=["fips", "year"]).to_frame(index=False)
cy = grid.merge(counties, on="fips").merge(yearly, on=["fips", "year"], how="left")
cy["max_wind_ms"] = cy.max_wind_ms.fillna(0)
counts = ["n_storms_500km", "n_storms_ts_force", "n_storms_hurricane_force"]
cy[counts] = cy[counts].fillna(0).astype(int)

# Damage: 0 in years with records, blank before 1996
cy["has_labels"] = cy.year.between(*LABEL_YEARS)
damage = ["prop_dmg_usd2023", "crop_dmg_usd2023", "deaths_direct", "injuries_direct"]
cy.loc[cy.has_labels, damage] = cy.loc[cy.has_labels, damage].fillna(0)
cy.loc[~cy.has_labels, damage] = np.nan
cy["total_dmg_usd2023"] = cy.prop_dmg_usd2023 + cy.crop_dmg_usd2023
cy["pop_loss_usd2023"] = (cy.deaths_direct + cy.injuries_direct / 10) * VSL

# What was at risk that year (housing units and building value; 2000-2024 only)
housing = pd.read_csv(CLEAN / "housing_by_year.csv", dtype={"fips": str})
cy = cy.merge(housing, on=["fips", "year"], how="left")

# Hits and how often they happen
cy["hurricane_hit"] = cy.max_wind_ms >= HURRICANE_MS
cy["ts_hit"] = cy.max_wind_ms >= TROPICAL_STORM_MS
rates = cy.groupby("fips").agg(p_hurricane_hit=("hurricane_hit", "mean"),
                               p_ts_hit=("ts_hit", "mean"),
                               mean_hurricanes_per_yr=("n_storms_hurricane_force", "mean")).round(4)
cy = cy.merge(rates.reset_index(), on="fips")

cy = cy[["fips", "state", "county_name", "year", "has_labels", "housing_units",
         "bldg_value_usd2023", "max_wind_ms", "hurricane_hit",
         "ts_hit", "n_storms_500km", "n_storms_ts_force", "n_storms_hurricane_force",
         "p_hurricane_hit", "p_ts_hit", "mean_hurricanes_per_yr", "prop_dmg_usd2023",
         "crop_dmg_usd2023", "total_dmg_usd2023", "deaths_direct", "injuries_direct",
         "pop_loss_usd2023"]].sort_values(["fips", "year"])
assert len(cy) == len(counties) * len(years)
cy.to_csv(FINAL / "county_year.csv", index=False)

hit = cy[cy.has_labels & cy.hurricane_hit]
print(f"07 county_year: {len(cy):,} rows; {cy.hurricane_hit.sum()} hurricane-force hits, "
      f"{(hit.total_dmg_usd2023 > 0).mean():.0%} of labeled hits had recorded damage")
