"""Build the one table the basic model uses: one row per storm x county, 2000-2024.

Each row has
  features   what the storm did at the county + two county traits
  exposure   what was at risk THAT year (building value, farm+timber sales, population)
             and in 2024 (for pricing every storm on today's exposure -> EAL)
  targets    TOTAL recorded damage, all record types (NY Fed / NOAA):
             property $, crop $, people = direct deaths + direct injuries / 10 (FEMA's rule)
Out: models/data/model_table.csv
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
from common import VSL, TROPICAL_STORM_MS, cpi_factors   # noqa: E402
from load_data import load                               # noqa: E402

VSL_2023 = VSL * cpi_factors()[2024]     # FEMA's $13.7M is in 2024 dollars
FEATURES = ["wind_ms", "wind_excess", "log_wind_exposure", "log_dist_km",
            "coastal", "coastal_x_wind", "log_density"]


def build():
    df = load("storm_county")
    df = df[df.has_labels & df.year.between(2000, 2024)].copy()

    stats = load("county_stats_by_year")[["fips", "year", "population", "density_per_sqmi",
                                          "bldg_value_usd2023", "farm_timber_value_usd2023"]]
    stats = stats.rename(columns={"bldg_value_usd2023": "bldg_value", "farm_timber_value_usd2023": "farm_timber_value"})
    df = df.merge(stats, on=["fips", "year"], how="left")
    now = stats[stats.year == 2024].drop(columns=["year", "density_per_sqmi"]).add_suffix("_2024") \
        .rename(columns={"fips_2024": "fips"})
    df = df.merge(now, on="fips", how="left")
    df = df.merge(load("counties")[["fips", "state", "coastal"]], on="fips")
    df = df.merge(load("storms")[["storm_id", "name"]], on="storm_id")

    # Features (all known from the storm track and the county before the storm)
    df["wind_ms"] = df.wind_est_ms
    df["wind_excess"] = np.maximum(0, df.wind_est_ms - TROPICAL_STORM_MS)   # damage starts near TS force
    df["log_wind_exposure"] = np.log1p(df.wind_exposure)                    # wind x hours, very skewed
    df["log_dist_km"] = np.log1p(df.min_dist_km)
    df["coastal"] = df.coastal.astype(float)
    df["coastal_x_wind"] = df.coastal * df.wind_excess                      # surge needs coast AND wind
    df["log_density"] = np.log(df.density_per_sqmi.clip(lower=0.1))

    # Targets
    df["prop_usd"] = df.prop_dmg_usd2023
    df["crop_usd"] = df.crop_dmg_usd2023
    df["people_equiv"] = df.deaths_direct + df.injuries_direct / 10
    # Property split by record type (for the "separate models by damage type" strategy)
    df["prop_surge_usd"] = df.prop_dmg_surge_usd2023
    df["prop_rain_usd"] = df.prop_dmg_rain_usd2023
    df["prop_wind_usd"] = (df.prop_usd - df.prop_surge_usd - df.prop_rain_usd).clip(lower=0)   # hurricane + other wind + tornado

    # Storm strength = strongest wind any county felt (for reporting by category)
    peak = df.groupby("storm_id").wind_est_ms.transform("max")
    df["storm_strength"] = np.select([peak < 33, peak < 50], ["TD / TS", "Cat 1-2"], "Cat 3+")

    keep = (["storm_id", "name", "storm_strength", "fips", "state", "year"] + FEATURES
            + ["bldg_value", "farm_timber_value", "population",
               "bldg_value_2024", "farm_timber_value_2024", "population_2024",
               "prop_usd", "crop_usd", "people_equiv", "prop_wind_usd", "prop_surge_usd", "prop_rain_usd"])
    return df[keep].sort_values(["year", "storm_id", "fips"]).reset_index(drop=True)


def build_catalog():
    """Every storm x county row 1975-2025 (51 seasons) with wind, coast and 2024 exposure.
    No damage columns: used only to price older storms with a fitted model (longer-catalog EAL)."""
    df = load("storm_county")[["storm_id", "fips", "year", "wind_est_ms"]].rename(columns={"wind_est_ms": "wind_ms"})
    stats = load("county_stats_by_year")
    now = stats[stats.year == 2024][["fips", "bldg_value_usd2023", "farm_timber_value_usd2023", "population"]]
    now.columns = ["fips", "bldg_value_2024", "farm_timber_value_2024", "population_2024"]
    df = df.merge(now, on="fips").merge(load("counties")[["fips", "coastal"]], on="fips")
    df["coastal"] = df.coastal.astype(float)
    return df


if __name__ == "__main__":
    df = build()
    assert not df.duplicated(["storm_id", "fips"]).any()
    assert df.notna().all().all(), df.isna().sum()[lambda s: s > 0]
    assert (df.bldg_value > 0).all() and (df.population > 0).all()
    df.to_csv(HERE / "data" / "model_table.csv", index=False)
    print(f"model_table: {len(df):,} rows, {df.storm_id.nunique()} storms, {df.fips.nunique():,} counties, 2000-2024")
    print(f"  property ${df.prop_usd.sum() / 1e9:.1f}B in {(df.prop_usd > 0).sum():,} rows | "
          f"crop ${df.crop_usd.sum() / 1e9:.1f}B in {(df.crop_usd > 0).sum():,} rows | "
          f"people {df.people_equiv.sum():,.0f} death-equivalents (${df.people_equiv.sum() * VSL_2023 / 1e9:.1f}B) "
          f"in {(df.people_equiv > 0).sum():,} rows")
    print("  storms by strength:", df.groupby("storm_strength").storm_id.nunique().to_dict())
    print(f"  property by type: wind ${df.prop_wind_usd.sum() / 1e9:.1f}B, surge ${df.prop_surge_usd.sum() / 1e9:.1f}B "
          f"(inland share {df.prop_surge_usd[df.coastal == 0].sum() / df.prop_surge_usd.sum():.1%}), "
          f"rain ${df.prop_rain_usd.sum() / 1e9:.1f}B")
    cat = build_catalog()
    assert cat.notna().all().all()
    cat.to_csv(HERE / "data" / "catalog_table.csv", index=False)
    print(f"catalog_table: {len(cat):,} rows, {cat.storm_id.nunique()} storms, {cat.year.min()}-{cat.year.max()}")
