# Hurricane Data Pipeline

> Project goal, current results and what worked: see `PROJECT_README.md`. This file covers the data pipeline.

Builds our hurricane modeling dataset from untouched source files.

```
python3 pipeline/run_all.py        # about 2-3 minutes
```

```python
import sys; sys.path.insert(0, "pipeline")
from load_data import load, model_table
df = model_table()          # labeled storm x county rows + exposure + damage ratios
```

## Folders

| Folder | What's in it |
|---|---|
| `source/` | Raw downloads, never edited (see `SOURCES.md`) |
| `pipeline/` | One script per step; `common.py` holds every setting |
| `clean/` | In-between tables, rebuilt every run |
| `final/` | **The tables we model on**, rebuilt every run |
| `docs/` | HURDAT2 format guide, NY Fed methodology |

## Steps

| Step | What it does |
|---|---|
| 00 ct_crosswalk | Connecticut's old counties -> 2022 planning regions (by population) |
| 01 counties | Master county list (contiguous US + DC), where people live, exposure, FEMA's hurricane numbers |
| 01b census_by_year | Housing units and population per county per year (Census, 2000-2024) and building value per year = FEMA value per 2020 unit x units |
| 01c ag_by_year | Farm, crop and timber sales and cropland/woodland acres per county per year (USDA Census of Agriculture, straight line between census years) |
| 01d county_stats | One table of yearly county stats (population, housing, building value, farm/crop/timber value, cropland, woodland), 2000-2024 |
| 02 tracks | HURDAT2 storms 1975-2025, filled in to hourly; storm size from the Willoughby (2006) formula |
| 03 storm_county | Every storm x county within 500 km: closest distance, estimated peak wind, time window |
| 04 loss_records | NY Fed hurricane damage + other event types that carry it (surge, coastal flood, flood, flash flood, high/strong wind, tornado): one weighting scheme, fixed county codes, 2023 dollars |
| 05 losses | Match each damage record to its storm (closest within 500 km on those dates), then add up |
| 06 final_tables | Publish `storm_county`, `storms`, `counties`, `storm_tracks` |
| 07 county_year | One row per county per year: max wind, hit, hit probability, damage |
| data_dictionary | Writes `final/data_dictionary.csv` |

## Final tables

| Table | One row per | Rows |
|---|---|---|
| `storm_county.csv` | storm x county within 500 km (**main table**) | 164,592 |
| `storms.csv` | storm, 1975-2025 | 839 |
| `counties.csv` | county | 3,109 |
| `storm_tracks.csv` | storm x hour | 133,061 |
| `county_year.csv` | county x year | 158,559 |
| `county_stats_by_year.csv` | county x year, 2000-2024: exposure stats only, nothing blank | 77,725 |

## Things to know

- `fips` is text; use `load()` or `dtype={"fips": str}` so leading zeros survive.
- Money ending `_usd2023` is in 2023 dollars.
- Damage is **0** in years with records (1996-2024) and **blank** before 1996. Never fill blanks with 0.
- Damage covers hurricane / tropical storm / depression records (any storm within 500 km during the record's dates) PLUS these other NOAA event types when a storm was clearly responsible:
  - storm surge, coastal flood -> `prop_dmg_surge_usd2023`
  - flood, flash flood (rain) -> `prop_dmg_rain_usd2023` (Harvey 2017 is ~90% of it)
  - high wind, strong wind -> `prop_dmg_other_wind_usd2023` (e.g. Sandy 2012 filed as post-tropical high wind)
  - tornado -> `prop_dmg_tornado_usd2023`
  - Condition for these extra types: dates overlap the storm's passage (-1/+2 days), storm within 400 km, AND tropical-storm wind (>= 17.5 m/s) at the county or storm within 100 km. Heavy rain (~$0) and drought (long records overlap storms by chance) are left out. Crop $ and direct deaths from extra types: `crop_dmg_extra_usd2023`, `deaths_direct_extra`. Sensitivity of these rules: `analysis/surge_matching_eda.py`, `analysis/candidate_matching_eda.py`.
  - Note FEMA's NRI treats coastal flooding and riverine flooding as separate hazards from hurricane, so compare like with like.
- Wind estimate: modified Rankine vortex, storm size from Willoughby, Darling & Rahn (2006), peak over all hours.
- Exposure varies by year (2000-2024): population and housing from Census estimates, building value = FEMA value per 2020 housing unit x that year's units, farm/crop/timber value and acres from the USDA census (every 5 years, straight line between). All in `county_stats_by_year.csv`.
- Settings (years, radius, dollar year, seed) are in `pipeline/common.py`.

## Models

`models/` holds the basic EAL model (plain numpy, no extra packages). Run from this folder:

    python3 models/build_table.py    # storm x county table 2000-2024 -> models/data/model_table.csv
    python3 models/basic_model.py    # fit, storm-held-out test, county EAL -> models/results/
    python3 models/damage_table.py   # damage table by wind bin x coast, scored against the basic model
    python3 models/slide_chart.py    # one comparison chart for the presentation (run after strategies.py)
    python3 models/strategies.py     # all model ideas side by side (about 2.5 minutes); shared pieces in core.py, evaluate.py

- Same two-part model for property, crop and people: chance of any damage (logistic) x size if damaged
  (gamma on damage / exposure, every storm weighted equally) x exposure.
- Target = total recorded damage (all record types). People = direct deaths + injuries / 10, priced at FEMA's value of a statistical life in 2023 $.
- Tested by holding out whole storms (5 folds); EAL = held-out predictions priced on 2024 exposure, averaged over the 25 seasons 2000-2024.

## Sandbox

`sandbox/` is a separate work area for model experiments (see `sandbox/README.md`). It reads
`models/data/model_table.csv` and the final tables, and never changes `pipeline/` or `models/`.
It holds a stratified dev / locked-test split of the storms, the experiment rounds, and the chosen
4-input model (`sandbox/CHOICE.md`).
