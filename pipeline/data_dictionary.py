"""Writes final/data_dictionary.csv: what every column in final/ means.
Run last (run_all.py does this). It stops if a column has no description."""
import pandas as pd
from common import *

USD = "USD (2023)"
HURDAT, NRI, NYFED = "NOAA HURDAT2", "FEMA NRI", "NY Fed Losses from Natural Disasters"

# (table, column) -> (unit, description, source).  "*" = same meaning in every table.
DESCRIPTIONS = {
    ("*", "storm_id"): ("", "Storm id: basin + number + year, e.g. AL092022 = Ian", HURDAT),
    ("*", "fips"): ("", "5-digit county code, stored as text so the leading zero stays", NRI),
    ("*", "state"): ("", "State abbreviation", NRI),
    ("*", "county_name"): ("", "County name (Connecticut: 2022 planning regions)", NRI),
    ("*", "year"): ("", "Hurricane season year", HURDAT),
    ("*", "has_labels"): ("", "True for 1996-2024, the years with damage records; damage is blank when False", "derived"),
    ("*", "name"): ("", "Storm name", HURDAT),
    # storm_county
    ("storm_county", "month"): ("", "Month of closest approach", HURDAT),
    ("storm_county", "min_dist_km"): ("km", "Closest distance from storm center to the county's population center", "derived"),
    ("storm_county", "vmax_at_min_ms"): ("m/s", "Storm's max sustained wind at closest approach", HURDAT),
    ("storm_county", "rmax_km_at_min"): ("km", "Storm size (radius of max wind, Willoughby 2006) at closest approach", "derived"),
    ("storm_county", "wind_est_ms"): ("m/s", "Estimated peak wind at the county: strongest over all hours (Rankine vortex, Willoughby size)", "derived"),
    ("storm_county", "hours_ts_force"): ("hours", "Hours with estimated wind at the county >= 17.5 m/s (tropical-storm force)", "derived"),
    ("storm_county", "hours_hurricane_force"): ("hours", "Hours with estimated wind at the county >= 32.9 m/s (hurricane force)", "derived"),
    ("storm_county", "wind_exposure"): ("m/s-hours", "Estimated wind above 17.5 m/s, summed over all hours: strength x duration", "derived"),
    ("storm_county", "hours_within_100km"): ("hours", "Hours the storm center spent within 100 km", "derived"),
    ("storm_county", "time_at_min_utc"): ("UTC", "Time of closest approach", HURDAT),
    ("storm_county", "enter_500km_utc"): ("UTC", "First hour within 500 km", HURDAT),
    ("storm_county", "exit_500km_utc"): ("UTC", "Last hour within 500 km", HURDAT),
    ("storm_county", "prop_dmg_usd2023"): (USD, "Recorded property damage", NYFED),
    ("storm_county", "crop_dmg_usd2023"): (USD, "Recorded crop damage (includes timber)", NYFED),
    ("storm_county", "crop_dmg_area_usd2023"): (USD, "Crop damage split by land area instead of housing (sensitivity check)", NYFED),
    ("storm_county", "injuries_direct"): ("people", "Direct injuries (can be fractional: zone totals split across counties)", NYFED),
    ("storm_county", "injuries_indirect"): ("people", "Indirect injuries", NYFED),
    ("storm_county", "deaths_direct"): ("people", "Direct deaths (can be fractional)", NYFED),
    ("storm_county", "deaths_indirect"): ("people", "Indirect deaths", NYFED),
    ("storm_county", "n_loss_records"): ("count", "Damage records added up into this row", "derived"),
    ("storm_county", "n_ambiguous_records"): ("count", "Of those, records where another storm also qualified", "derived"),
    # storms
    ("storms", "start_utc"): ("UTC", "First reading", HURDAT),
    ("storms", "end_utc"): ("UTC", "Last reading", HURDAT),
    ("storms", "peak_vmax_ms"): ("m/s", "Strongest wind in the storm's life", HURDAT),
    ("storms", "min_pmin_mb"): ("mb", "Lowest central pressure", HURDAT),
    ("storms", "peak_category"): ("", "Saffir-Simpson category at peak (TD, TS, 1-5)", "derived"),
    ("storms", "n_counties_500km"): ("count", "Counties reached within 500 km", "derived"),
    ("storms", "n_counties_100km"): ("count", "Counties reached within 100 km", "derived"),
    ("storms", "closest_county_km"): ("km", "Closest approach to any county; blank = never within 500 km", "derived"),
    ("storms", "obs_prop_dmg_usd2023"): (USD, "Total recorded property damage", NYFED),
    ("storms", "obs_crop_dmg_usd2023"): (USD, "Total recorded crop damage", NYFED),
    ("storms", "obs_deaths_direct"): ("people", "Total recorded direct deaths", NYFED),
    # counties
    ("counties", "pop_lat"): ("degrees", "Latitude of the county's 2020 center of population", "Census"),
    ("counties", "pop_lon"): ("degrees", "Longitude of the county's 2020 center of population", "Census"),
    ("counties", "population"): ("people", "2020 population", NRI),
    ("counties", "bldg_value_usd2023"): (USD, "Total building value (exposure)", NRI),
    ("counties", "ag_value_usd2023"): (USD, "Agriculture value (exposure; excludes timber)", NRI),
    ("counties", "area_sqmi"): ("sq mi", "Land area", NRI),
    ("counties", "density_per_sqmi"): ("people/sq mi", "Population / area", "derived"),
    ("counties", "coastal"): ("", "True if FEMA gives the county any coastal flooding loss", NRI),
    ("counties", "nri_hrcn_n_events"): ("count", "FEMA: recorded hurricane events", NRI),
    ("counties", "nri_hrcn_afreq"): ("events/yr", "FEMA: hurricane annualized frequency", NRI),
    ("counties", "nri_hrcn_exp_bldg_usd2023"): (USD, "FEMA: building value exposed to hurricanes", NRI),
    ("counties", "nri_hrcn_exp_pop"): ("people", "FEMA: population exposed", NRI),
    ("counties", "nri_hrcn_exp_ag_usd2023"): (USD, "FEMA: agriculture value exposed", NRI),
    ("counties", "nri_hrcn_hlr_bldg"): ("fraction", "FEMA: historic loss ratio, buildings", NRI),
    ("counties", "nri_hrcn_hlr_pop"): ("fraction", "FEMA: historic loss ratio, population", NRI),
    ("counties", "nri_hrcn_hlr_ag"): ("fraction", "FEMA: historic loss ratio, agriculture", NRI),
    ("counties", "nri_hrcn_eal_bldg_usd2023"): (USD + "/yr", "FEMA: expected annual loss, buildings", NRI),
    ("counties", "nri_hrcn_eal_pop"): ("people/yr", "FEMA: expected annual deaths (10 injuries = 1)", NRI),
    ("counties", "nri_hrcn_eal_pope_usd2023"): (USD + "/yr", "FEMA: expected annual population loss in dollars", NRI),
    ("counties", "nri_hrcn_eal_ag_usd2023"): (USD + "/yr", "FEMA: expected annual loss, agriculture", NRI),
    ("counties", "nri_hrcn_eal_total_usd2023"): (USD + "/yr", "FEMA: total hurricane expected annual loss (the benchmark)", NRI),
    ("counties", "nri_hrcn_eal_score"): ("percentile", "FEMA: national percentile of hurricane EAL", NRI),
    ("counties", "nri_hrcn_eal_rating"): ("", "FEMA: hurricane EAL rating (Very Low to Very High)", NRI),
    # storm_tracks
    ("storm_tracks", "time_utc"): ("UTC", "Hour (interpolated between 6-hourly readings)", HURDAT),
    ("storm_tracks", "lat"): ("degrees", "Storm center latitude", HURDAT),
    ("storm_tracks", "lon"): ("degrees", "Storm center longitude (west is negative)", HURDAT),
    ("storm_tracks", "vmax_ms"): ("m/s", "Max sustained wind", HURDAT),
    ("storm_tracks", "pmin_mb"): ("mb", "Central pressure (blank where never recorded)", HURDAT),
    ("storm_tracks", "status"): ("", "Storm type: HU hurricane, TS tropical storm, TD depression, EX extratropical", HURDAT),
    ("storm_tracks", "rmax_km"): ("km", "Storm size: radius of max wind from the Willoughby (2006) formula", "derived"),
    # county_year
    ("county_year", "housing_units"): ("units", "Housing units that year (Census estimates, July 1); blank before 2000", "Census housing unit estimates"),
    ("county_year", "bldg_value_usd2023"): (USD, "Building value that year = FEMA value per 2020 housing unit x that year's housing units; blank before 2000", "derived from FEMA NRI + Census"),
    ("county_year", "max_wind_ms"): ("m/s", "Strongest estimated wind from any storm that year; 0 if none within 500 km", "derived"),
    ("county_year", "hurricane_hit"): ("", "max_wind_ms >= 32.9 m/s (64 kt)", "derived"),
    ("county_year", "ts_hit"): ("", "max_wind_ms >= 17.5 m/s (34 kt)", "derived"),
    ("county_year", "n_storms_500km"): ("count", "Storms that passed within 500 km", "derived"),
    ("county_year", "n_storms_ts_force"): ("count", "Storms bringing tropical-storm-force wind", "derived"),
    ("county_year", "n_storms_hurricane_force"): ("count", "Storms bringing hurricane-force wind", "derived"),
    ("county_year", "p_hurricane_hit"): ("probability", "Share of 1975-2025 years with a hurricane-force hit (same every year)", "derived"),
    ("county_year", "p_ts_hit"): ("probability", "Share of years with tropical-storm-force wind", "derived"),
    ("county_year", "mean_hurricanes_per_yr"): ("storms/yr", "Average hurricane-force storms per year", "derived"),
    ("county_year", "prop_dmg_usd2023"): (USD, "Recorded property damage that year", NYFED),
    ("county_year", "crop_dmg_usd2023"): (USD, "Recorded crop damage that year", NYFED),
    ("county_year", "total_dmg_usd2023"): (USD, "Property + crop damage", NYFED),
    ("county_year", "deaths_direct"): ("people", "Recorded direct deaths that year", NYFED),
    ("county_year", "injuries_direct"): ("people", "Recorded direct injuries that year", NYFED),
    ("county_year", "pop_loss_usd2023"): ("USD", "(deaths + injuries/10) x $13.7M (FEMA's value of a life)", "derived"),
}

rows = []
for table in ["storm_county", "storms", "counties", "storm_tracks", "county_year"]:
    df = pd.read_csv(FINAL / f"{table}.csv", nrows=50_000, low_memory=False)
    for col in df.columns:
        entry = DESCRIPTIONS.get((table, col)) or DESCRIPTIONS.get(("*", col)) \
            or DESCRIPTIONS.get(("storm_county", col))
        assert entry, f"no description for {table}.{col} - add it to data_dictionary.py"
        unit, description, source = entry
        rows.append({"table": table, "column": col, "unit": unit,
                     "description": description, "source": source})
pd.DataFrame(rows).to_csv(FINAL / "data_dictionary.csv", index=False)
print(f"data_dictionary: {len(rows)} columns documented")
