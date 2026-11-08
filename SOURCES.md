# Sources

| File in `source/` | Source | Downloaded | Notes |
|---|---|---|---|
| `nri_counties.csv` | FEMA National Risk Index, All Counties table: https://hazards.fema.gov/nri/data-resources | 2026-10-12 | NRI version: v1.20 (December 2025); dollars are December 2024 dollars per FEMA's metadata, converted to 2023 dollars in step 01 (`NRI_DOLLAR_MONTH` in common.py) |
| `hurdat2_atlantic.txt` | NOAA NHC HURDAT2 Atlantic: https://www.nhc.noaa.gov/data/ | 2026-10-12 | Original file name: hurdat2-1851-2025-______.txt ; format guide: `hurdat2-format.pdf` |
| `nyc_fed_losses.csv` | Federal Reserve Bank of New York, Losses from Natural Disasters: https://www.newyorkfed.org/research/policy/natural-disaster-losses | 2026-08-11 | Covers 1996-2025; method: `nyc_fed_methodology.pdf` |
| `cenpop2020_county.txt` | Census 2020 Centers of Population, county: https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt | 2026-10-12 | Uses Connecticut's 8 old counties |
| `cpi_monthly.csv` | FRED CPIAUCSL (CPI-U, seasonally adjusted): https://fred.stlouisfed.org/series/CPIAUCSL | 2026-10-12 | Monthly 1947-01 to 2026-08; pipeline uses yearly averages of complete years, plus Dec 2024 for NRI |
| `CenPop2020_Mean_TR09.txt` | Census 2020 Centers of Population, Connecticut tracts: https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR09.txt | 2026-10-12 | Not used by the pipeline; kept for reference |
| `ct_cou_to_cousub_crosswalk.txt` | Census Connecticut county-to-town crosswalk: https://www2.census.gov/geo/docs/reference/ct_change/ | 2026-10-12 | Used in step 00 |
| `Housing_2000_2010.csv` | Census county housing unit estimates, 2000-2010 intercensal: https://www.census.gov/programs-surveys/popest/data/data-sets.html | 2026-10-28 | Has county codes |
| `Housing_2010_2020.xlsx` | Census CO-EST2020INT-HU, 2010-2020 intercensal (Nov 2024 release) | 2026-10-28 | County names only; Connecticut old counties |
| `Housing_2020_2025.xlsx` | Census CO-EST2025-HU, 2020-2025 (May 2026 release) | 2026-10-28 | County names only; Connecticut planning regions |
| `population2000_2010.csv` | Census county population, 2000-2010 intercensal (co-est00int): https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-2000-2010-counties.html | 2026-11-06 | Has county codes; used in step 01b |
| `population2010_2020.xlsx` | Census CO-EST2020INT-POP, 2010-2020 intercensal: https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-2010-2020-counties.html | 2026-11-06 | County names only; Connecticut old counties |
| `population2020_2025.xlsx` | Census CO-EST2025-POP, 2020-2025: https://www.census.gov/data/datasets/time-series/demo/popest/2020s-counties-total.html | 2026-11-06 | County names only; Connecticut planning regions |
| `USDA_farmvalue.csv` | USDA NASS Quick Stats (https://quickstats.nass.usda.gov), Program CENSUS, County, Domain TOTAL, years 2002-2022, Data Item `COMMODITY TOTALS - SALES, MEASURED IN $` | 2026-11-06 | Used in step 01c; (D) = withheld |
| `USDA_cropvalue.csv` | USDA NASS Quick Stats (https://quickstats.nass.usda.gov), Program CENSUS, County, Domain TOTAL, years 2002-2022, Data Item `CROP TOTALS - SALES, MEASURED IN $` | 2026-11-06 | Used in step 01c |
| `USDA_croplandacres.csv` | USDA NASS Quick Stats (https://quickstats.nass.usda.gov), Program CENSUS, County, Domain TOTAL, years 2002-2022, Data Item `AG LAND, CROPLAND - ACRES` | 2026-11-06 | Used in step 01c |
| `USDA_woodlandacres.csv` | USDA NASS Quick Stats (https://quickstats.nass.usda.gov), Program CENSUS, County, Domain TOTAL, years 2002-2022, Data Item `AG LAND, WOODLAND - ACRES` | 2026-11-06 | Used in step 01c; farm woodland only, not all forest |
| `USDA_forestproduct.csv` | USDA NASS Quick Stats, Census, County, Sector ECONOMICS > INCOME > INCOME, FARM-RELATED, Data Item `...FOREST PRODUCTS... - RECEIPTS, MEASURED IN $` | 2026-11-06 | Used in step 01c; farm timber only, 24% withheld |
