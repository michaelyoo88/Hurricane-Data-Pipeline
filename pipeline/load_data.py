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
      crop_ratio   = crop damage / farm value (blank if the county has no farm value)
      deaths_per_100k = direct deaths per 100,000 people
    """
    yearly = load("county_year")[["fips", "year", "housing_units", "bldg_value_usd2023"]]
    df = (load("storm_county")
          .merge(load("counties")[["fips", "state", "population", "ag_value_usd2023",
                                   "density_per_sqmi", "coastal"]], on="fips")
          .merge(yearly, on=["fips", "year"], how="left")       # building value for that year
          .merge(load("storms")[["storm_id", "name", "peak_category"]], on="storm_id"))
    if labeled_only:
        df = df[df.has_labels].copy()
    df["prop_ratio"] = df.prop_dmg_usd2023 / df.bldg_value_usd2023   # blank before 2000
    df["crop_ratio"] = (df.crop_dmg_usd2023 / df.ag_value_usd2023).where(df.ag_value_usd2023 > 0)
    df["deaths_per_100k"] = df.deaths_direct / df.population * 1e5
    # Damage can exceed value: NOAA crop damage includes timber, NRI farm value doesn't
    df["crop_exceeds_value"] = df.crop_ratio > 1
    return df.reset_index(drop=True)
