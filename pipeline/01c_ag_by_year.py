"""Step 01c - Farm value, crop value, cropland and woodland for every county, every year.

Source: USDA Census of Agriculture (NASS Quick Stats), county level, taken every 5 years
(2002, 2007, 2012, 2017, 2022). Downloaded by hand, one file per data item:
    USDA_farmvalue.csv       COMMODITY TOTALS - SALES, MEASURED IN $   (crops + livestock)
    USDA_cropvalue.csv       CROP TOTALS - SALES, MEASURED IN $        (crops only)
    USDA_croplandacres.csv   AG LAND, CROPLAND - ACRES
    USDA_woodlandacres.csv   AG LAND, WOODLAND - ACRES
    USDA_forestproduct.csv   INCOME, FARM-RELATED, FOREST PRODUCTS - RECEIPTS, MEASURED IN $
                             (timber sold from farm woodland only; ~24% withheld)

How each county gets a number for every year 2000-2024:
  1. "(D)" = withheld by USDA for privacy -> filled from the same county's other census
     years (straight line between them, or the nearest one at the ends).
  2. Dollars are converted to 2023 dollars using the census year's CPI.
  3. Years between censuses: straight line between the two census years.
     2000-2001 use 2002; 2023-2024 use 2022.
Counties USDA never lists (Virginia independent cities, DC, Baltimore city, St. Louis
city, and similar urban places that USDA folds into a neighbor) get 0.
Connecticut: USDA uses the old 8 counties; split into planning regions by the
population shares in ct_crosswalk.csv (assumes farms are spread like people - rough).

Out: clean/ag_by_year.csv
     (fips, year, farm_value_usd2023, crop_value_usd2023, forest_value_usd2023,
      cropland_acres, woodland_acres)
"""
import numpy as np
import pandas as pd
from common import *

FILES = {"farm_value_usd2023": "USDA_farmvalue.csv",
         "crop_value_usd2023": "USDA_cropvalue.csv",
         "forest_value_usd2023": "USDA_forestproduct.csv",
         "cropland_acres": "USDA_croplandacres.csv",
         "woodland_acres": "USDA_woodlandacres.csv"}
CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
YEARS = range(2000, 2025)

counties = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str}, usecols=["fips"])
crosswalk = pd.read_csv(CLEAN / "ct_crosswalk.csv", dtype=str)
crosswalk["share"] = crosswalk.share.astype(float)
cpi = cpi_factors()


def read_usda(path):
    """One USDA file -> rows of (fips, year, value). '(D)' becomes blank."""
    d = pd.read_csv(SOURCE / path, dtype=str)
    d = d[d["County ANSI"].notna() & (d["County ANSI"] != "998")]      # 998 = "other counties combined"
    d["fips"] = d["State ANSI"].str.zfill(2) + d["County ANSI"].str.zfill(3)
    d["year"] = d.Year.astype(int)
    d["value"] = pd.to_numeric(d.Value.str.strip().str.replace(",", ""), errors="coerce")
    # County changes: Bedford city VA -> Bedford County; Shannon SD -> Oglala Lakota
    d["fips"] = d.fips.replace({"51515": "51019", "46113": "46102"})
    return d[["fips", "year", "value"]]


def fill_withheld(wide):
    """Fill '(D)' gaps from the same county's other census years."""
    return wide.interpolate(axis=1, limit_direction="both")


def split_ct(wide):
    """Old CT counties -> planning regions, by population share."""
    old = wide.index.isin(CT_OLD_COUNTIES)
    ct = wide[old].reset_index().merge(crosswalk, left_on="fips", right_on="old_fips")
    ct[CENSUS_YEARS] = ct[CENSUS_YEARS].mul(ct.share, axis=0)
    ct = ct.groupby("region_fips")[CENSUS_YEARS].sum()
    ct.index.name = "fips"
    return pd.concat([wide[~old], ct])


parts = []
report = []
for column, path in FILES.items():
    d = read_usda(path)
    n_withheld = d.value.isna().sum()
    # one row per county, one column per census year; min_count=1 keeps "(D)" blank
    wide = d.groupby(["fips", "year"]).value.sum(min_count=1).unstack().reindex(columns=CENSUS_YEARS)
    n_all_withheld = wide.isna().all(axis=1).sum()
    wide = fill_withheld(wide)
    if column.endswith("usd2023"):
        wide = wide.mul([cpi[y] for y in CENSUS_YEARS], axis=1)
    wide = split_ct(wide)

    wide = wide.reindex(counties.fips)
    n_zero = wide.isna().all(axis=1).sum()
    wide = wide.fillna(0)                                             # never listed -> no farms

    # Every year 2000-2024: straight line between census years, flat at the ends
    yearly = pd.DataFrame({y: [np.interp(y, CENSUS_YEARS, row) for row in wide.values] for y in YEARS},
                          index=wide.index)
    long = yearly.stack().rename(column).reset_index()
    long.columns = ["fips", "year", column]
    parts.append(long.set_index(["fips", "year"]))
    report.append(f"{column}: {n_withheld} withheld values filled; {n_zero} counties set to 0 "
                  f"({n_all_withheld} of them withheld in every census)")

ag = pd.concat(parts, axis=1).reset_index().round(0)
assert len(ag) == len(counties) * len(YEARS) and ag.notna().all().all()
ag.sort_values(["fips", "year"]).to_csv(CLEAN / "ag_by_year.csv", index=False)
print(f"01c ag_by_year: {len(ag):,} county-years (2000-2024)")
for line in report:
    print("    " + line)
