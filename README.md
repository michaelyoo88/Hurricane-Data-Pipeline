# Hurricane Data Pipeline

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
| 01b housing_by_year | Housing units per county per year (Census, 2000-2024) and building value per year = FEMA value per 2020 unit x units |
| 02 tracks | HURDAT2 storms 1975-2025, filled in to hourly; storm size from the Willoughby (2006) formula |
| 03 storm_county | Every storm x county within 500 km: closest distance, estimated peak wind, time window |
| 04 loss_records | NY Fed hurricane damage: one weighting scheme, fixed county codes, 2023 dollars |
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

## Things to know

- `fips` is text; use `load()` or `dtype={"fips": str}` so leading zeros survive.
- Money ending `_usd2023` is in 2023 dollars.
- Damage is **0** in years with records (1996-2024) and **blank** before 1996. Never fill blanks with 0.
- Damage covers wind-type events only (hurricane, tropical storm, tropical depression). Storm surge and flooding are separate FEMA hazards, which is why Harvey and Helene look small.
- Wind estimate: modified Rankine vortex, storm size from Willoughby, Darling & Rahn (2006), peak over all hours.
- Building value varies by year (2000-2024) via Census housing units; farm value and population are still single snapshots from FEMA's NRI.
- Settings (years, radius, dollar year, seed) are in `pipeline/common.py`.
