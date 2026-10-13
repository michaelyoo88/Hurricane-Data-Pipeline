# Hurricane Data Pipeline

Builds the hurricane modeling dataset from untouched source files.

```
python3 pipeline/run_all.py        # about 2.5 minutes
```

```python
import sys; sys.path.insert(0, "pipeline")
from load_data import load, model_table
df = model_table()          # labeled storm x county rows + exposure + damage ratios
storms = load("storms")     # any single table
```

## Folders

| Folder | Contents | Edit by hand? |
|---|---|---|
| `source/` | Raw downloads (see `SOURCES.md`) | Never |
| `pipeline/` | One script per step; `common.py` holds every project setting | Yes |
| `clean/` | Intermediate tables + `logs/` (one log per step) | No, rebuilt |
| `final/` | **The dataset to use** | No, rebuilt |
| `docs/` | HURDAT2 format guide, NY Fed methodology | — |

## Final dataset (`final/`), star schema

| Table | Grain (one row per) | Key | Rows | Role |
|---|---|---|---|---|
| `storm_county.csv` | storm x county within 500 km | `storm_id` + `fips` | 164,592 | **Fact**: closest approach, storm strength, observed losses |
| `storms.csv` | storm, 1975-2025 | `storm_id` | 839 | Dimension: name, peak strength, category, totals |
| `counties.csv` | county (contiguous 48 + DC) | `fips` | 3,109 | Dimension: location, exposure, FEMA NRI hurricane benchmark |
| `county_year.csv` | county x year, 1975-2025 | `fips` + `year` | 158,559 | **Minimum modeling table**: max wind, hurricane hit, annual hit probability, damage |
| `storm_tracks.csv` | storm x hour | `storm_id` + `time_utc` | 133,061 | Detail: for upgrading the wind formula later |
| `data_dictionary.csv` | column | `table` + `column` | 69 | Unit, description, source and % blank for every column |

## Steps

| Step | What it does |
|---|---|
| 00 ct_crosswalk | Connecticut old counties -> 2022 planning regions (town crosswalk + population fitting) |
| 01 counties | NRI master county list, centers of population, exposure in 2023 $ |
| 02 tracks | HURDAT2 storms 1975-2025, parsed and interpolated to hours |
| 03 storm_county | Closest approach of every storm to every county within 500 km |
| 04 loss_records | NY Fed hurricane / tropical storm / depression records; CPI to 2023 $ |
| 05 loss_match | Each loss record -> closest storm within 500 km during its dates |
| 06 losses | Matched losses summed per storm x county |
| 07 final | Publishes `final/`, checks keys and joins, writes the data dictionary |
| 08 county_year | Collapses storms to one row per county per year; estimates wind (Rankine vortex), hits and hit probability |

## Rules worth knowing

- `fips` is 5-character text. Use `load()`, or `dtype={"fips": str}`.
- Money ending `_usd2023` is in 2023 dollars (annual-average CPI-U).
- Loss columns are **0** for labeled years (1996-2025) with no record and **blank** for 1975-1995 storms, which have no labels (`has_labels`).
- Losses are wind-event records only (Hurricane, Tropical Storm, Tropical Depression). Storm surge and flood are excluded, matching FEMA's separate Coastal Flooding hazard. This is why Harvey (2017) and Helene (2024) look small.
- NY Fed weights: Housing for property and crop damage, Population for deaths and injuries; `crop_dmg_area_usd2023` uses Area weights as a sensitivity check.
- Damage can exceed exposure: NOAA crop damage includes timber, NRI agriculture value does not (63 labeled rows, plus 35 in counties with zero recorded farm value). `model_table()` flags these.
- Scope: contiguous 48 states + DC. Alaska, Hawaii and territories are excluded.
- Every setting (years, radius, dollar year, weights) is in `pipeline/common.py`.
