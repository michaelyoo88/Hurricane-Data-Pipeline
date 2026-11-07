"""Load the final tables.

    import sys; sys.path.insert(0, "pipeline")
    from load_data import load, model_table
    counties = load("counties")
    df = model_table()      # labeled storm x county rows, ready for modeling
"""
import pandas as pd
from common import FINAL

DATES = {"storm_county": ["time_at_min_utc", "enter_500km_utc", "exit_500km_utc"],
         "storms": ["start_utc", "end_utc"], "storm_tracks": ["time_utc"]}


def load(table):
    """Read one final table; fips stays text so leading zeros survive."""
    return pd.read_csv(FINAL / f"{table}.csv", dtype={"fips": str},
                       parse_dates=DATES.get(table, []))


def model_table(labeled_only=True):
    """storm_county + county exposure + storm info, with damage ratios added.

    Ratios (damage / what was at risk) are the modeling targets:
      prop_ratio   = property damage / building value that year (from 2000 on)
      crop_ratio   = crop damage / (farm sales + farm timber sales) that year
                     (USDA; from 2000 on; blank if the county has no farms)
      deaths_per_100k = direct deaths per 100,000 people that year (from 2000 on)
    population (from counties) is the 2020 snapshot; population_yr is that year's.
    """
    yearly = load("county_year").rename(columns={"population": "population_yr"})[["fips", "year", "population_yr", "housing_units", "bldg_value_usd2023",
                                  "farm_value_usd2023", "crop_value_usd2023", "forest_value_usd2023",
                                  "cropland_acres", "woodland_acres"]]
    df = (load("storm_county")
          .merge(load("counties")[["fips", "state", "population", "ag_value_usd2023",
                                   "density_per_sqmi", "coastal"]], on="fips")
          .merge(yearly, on=["fips", "year"], how="left")       # building + farm value for that year
          .merge(load("storms")[["storm_id", "name", "peak_category"]], on="storm_id"))
    if labeled_only:
        df = df[df.has_labels].copy()
    df["prop_ratio"] = df.prop_dmg_usd2023 / df.bldg_value_usd2023   # blank before 2000
    farm_and_timber = df.farm_value_usd2023 + df.forest_value_usd2023
    df["crop_ratio"] = (df.crop_dmg_usd2023 / farm_and_timber).where(farm_and_timber > 0)
    df["deaths_per_100k"] = df.deaths_direct / df.population_yr * 1e5
    # Damage can exceed value: NOAA crop damage includes timber on all forest land,
    # USDA only counts timber sold from farms
    df["crop_exceeds_value"] = df.crop_ratio > 1
    return df.reset_index(drop=True)
