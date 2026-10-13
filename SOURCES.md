# Sources

| File in `source/` | Source | Downloaded | Notes |
|---|---|---|---|
| `nri_counties.csv` | FEMA National Risk Index, All Counties table: https://hazards.fema.gov/nri/data-resources | 2026-10-12 | NRI version: ___ ; dollar year: ___ (set `NRI_DOLLAR_YEAR` in common.py) |
| `hurdat2_atlantic.txt` | NOAA NHC HURDAT2 Atlantic: https://www.nhc.noaa.gov/data/ | 2026-10-12 | Original file name: hurdat2-1851-2025-______.txt ; format guide: `hurdat2-format.pdf` |
| `nyc_fed_losses.csv` | Federal Reserve Bank of New York, Losses from Natural Disasters: https://www.newyorkfed.org/research/policy/natural-disaster-losses | 2026-08-11 | Covers 1996-2025; method: `nyc_fed_methodology.pdf` |
| `cenpop2020_county.txt` | Census 2020 Centers of Population, county: https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt | 2026-10-12 | Uses Connecticut's 8 old counties |
| `cpi_monthly.csv` | FRED CPIAUCSL (CPI-U, seasonally adjusted): https://fred.stlouisfed.org/series/CPIAUCSL | 2026-10-12 | Monthly 1947-01 to 2026-08; pipeline uses complete years only |
| `CenPop2020_Mean_TR09.txt` | Census 2020 Centers of Population, Connecticut tracts: https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR09.txt | 2026-10-12 | Not used by the pipeline; kept for reference |
| `ct_cou_to_cousub_crosswalk.txt` | Census Connecticut county-to-town crosswalk: https://www2.census.gov/geo/docs/reference/ct_change/ | 2026-10-12 | Used in step 00 |
