"""Load the final dataset. Usage (from the pipeline folder or with it on sys.path):

    from load_data import load, model_table
    counties = load("counties")
    df = model_table()            # labeled storm x county rows, ready for modeling
"""
from pathlib import Path
import pandas as pd

FINAL = Path(__file__).resolve().parents[1] / "final"
# Which columns to parse as dates in each table (leading _ = internal use).
_DATES = {"storm_county": ["time_at_min_utc", "enter_500km_utc", "exit_500km_utc"],
          "storms": ["start_utc", "end_utc"], "storm_tracks": ["time_utc"]}


def load(table):
    """Read one final table with correct types (fips stays text)."""
    return pd.read_csv(FINAL / f"{table}.csv", dtype={"fips": str},
                       parse_dates=_DATES.get(table, []))


def model_table(labeled_only=True):
    """storm_county joined to exposure and storm attributes, plus damage ratios.

    Ratios are computed here, not stored, so the exposure choice can change:
      prop_ratio = property damage / building value
      crop_ratio = crop damage / agriculture value
      deaths_per_100k = direct deaths per 100,000 residents
    """
    sc = load("storm_county")
    c = load("counties")[["fips", "state", "population", "bldg_value_usd2023",
                          "ag_value_usd2023", "density_per_sqmi", "coastal"]]
    s = load("storms")[["storm_id", "name", "peak_category"]]
    # Join the fact table to its dimension tables (star schema), optionally keep
    # labeled years only, then compute ratios. .where(cond) blanks crop_ratio where
    # agriculture value is 0, avoiding division by zero (inf).
    df = sc.merge(c, on="fips").merge(s, on="storm_id")
    if labeled_only:
        df = df[df.has_labels].copy()
    df["prop_ratio"] = df.prop_dmg_usd2023 / df.bldg_value_usd2023
    df["crop_ratio"] = (df.crop_dmg_usd2023 / df.ag_value_usd2023).where(df.ag_value_usd2023 > 0)
    df["deaths_per_100k"] = df.deaths_direct / df.population * 1e5
    # Damage can exceed exposure: NOAA crop damage includes timber, NRI
    # agriculture value does not; one Rita 2005 row includes infrastructure.
    df["prop_exceeds_value"] = df.prop_ratio > 1
    df["crop_exceeds_value"] = df.crop_ratio > 1
    return df.reset_index(drop=True)
