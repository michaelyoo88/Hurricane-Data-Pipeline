"""Step 08: one row per county per year, 1975-2025 (the minimum modeling table).

For every contiguous-US county and every year, the strongest wind any storm
brought, the damage recorded, and the county's annual probability of a hurricane hit.

In : final/storm_county.csv, final/counties.csv
Out: final/county_year.csv (+ its rows appended to final/data_dictionary.csv)
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("08_county_year")
FINAL = ROOT / "final"
U = "usd2023"
sc = pd.read_csv(FINAL / "storm_county.csv", dtype={"fips": str})
c = pd.read_csv(FINAL / "counties.csv", dtype={"fips": str}, usecols=["fips", "state", "county_name"])

# 1. Wind each county felt from each storm (simple Rankine vortex)
d = sc.min_dist_km.clip(lower=RMW_KM)
sc["wind_ms"] = sc.vmax_at_min_ms * (RMW_KM / d) ** RANKINE_EXP

# 2. Collapse storms to county x year
g = sc.groupby(["fips", "year"])
cy = pd.DataFrame({
    "max_wind_ms": g.wind_ms.max(),
    "n_storms_500km": g.size(),
    "n_storms_ts_force": g.wind_ms.apply(lambda w: int((w >= TROPICAL_STORM_MS).sum())),
    "n_storms_hurricane_force": g.wind_ms.apply(lambda w: int((w >= HURRICANE_MS).sum())),
    f"prop_dmg_{U}": g[f"prop_dmg_{U}"].sum(min_count=1),
    f"crop_dmg_{U}": g[f"crop_dmg_{U}"].sum(min_count=1),
    "deaths_direct": g.deaths_direct.sum(min_count=1),
    "injuries_direct": g.injuries_direct.sum(min_count=1),
}).reset_index()

# 3. Full grid: every county x every year, including years no storm came near
years = range(STORM_YEAR_MIN, STORM_YEAR_MAX + 1)
grid = pd.MultiIndex.from_product([c.fips, years], names=["fips", "year"]).to_frame(index=False)
cy = grid.merge(c, on="fips").merge(cy, on=["fips", "year"], how="left")
cy["max_wind_ms"] = cy.max_wind_ms.fillna(0).round(2)
for col in ["n_storms_500km", "n_storms_ts_force", "n_storms_hurricane_force"]:
    cy[col] = cy[col].fillna(0).astype(int)

# 4. Damage: 0 in labeled years with no record, blank before labels exist
cy["has_labels"] = cy.year.between(LABEL_YEAR_MIN, LABEL_YEAR_MAX)
dmg = [f"prop_dmg_{U}", f"crop_dmg_{U}", "deaths_direct", "injuries_direct"]
cy.loc[cy.has_labels, dmg] = cy.loc[cy.has_labels, dmg].fillna(0)
cy.loc[~cy.has_labels, dmg] = np.nan
cy[f"total_dmg_{U}"] = cy[f"prop_dmg_{U}"] + cy[f"crop_dmg_{U}"]
cy[f"pop_loss_{U}"] = (cy.deaths_direct + cy.injuries_direct / 10) * VSL_USD

# 5. Hit indicators and annual probabilities (share of the 51 years)
cy["hurricane_hit"] = cy.max_wind_ms >= HURRICANE_MS
cy["ts_hit"] = cy.max_wind_ms >= TROPICAL_STORM_MS
n_years = len(years)
p = cy.groupby("fips").agg(p_hurricane_hit=("hurricane_hit", "mean"),
                           p_ts_hit=("ts_hit", "mean"),
                           mean_hurricanes_per_yr=("n_storms_hurricane_force", "mean"))
cy = cy.merge(p.round(4).reset_index(), on="fips")

cols = ["fips", "state", "county_name", "year", "has_labels",
        "max_wind_ms", "hurricane_hit", "ts_hit", "n_storms_500km",
        "n_storms_ts_force", "n_storms_hurricane_force",
        "p_hurricane_hit", "p_ts_hit", "mean_hurricanes_per_yr",
        f"prop_dmg_{U}", f"crop_dmg_{U}", f"total_dmg_{U}",
        "deaths_direct", "injuries_direct", f"pop_loss_{U}"]
cy = cy[cols].sort_values(["fips", "year"])

# Checks
log.check(len(cy) == len(c) * n_years, f"{len(cy):,} rows = {len(c):,} counties x {n_years} years")
log.check(not cy.duplicated(["fips", "year"]).any(), "(fips, year) is a unique key")
log.check(np.isclose(cy[f"prop_dmg_{U}"].sum(), sc[f"prop_dmg_{U}"].sum()),
          "all property damage carried over from storm_county")
lab = cy[cy.has_labels]
log.write(f"County-years with a hurricane-force hit: {cy.hurricane_hit.sum():,} "
          f"({cy.hurricane_hit.mean():.2%}); tropical-storm force: {cy.ts_hit.sum():,}")
log.write(f"Counties ever hit by hurricane-force wind: {(p.p_hurricane_hit > 0).sum():,} of {len(c):,}")
log.write(f"Labeled county-years with damage > 0: {(lab[f'total_dmg_{U}'] > 0).sum():,} "
          f"({(lab[f'total_dmg_{U}'] > 0).mean():.2%})")
hit_dmg = lab.loc[lab.hurricane_hit, f"total_dmg_{U}"].gt(0).mean()
miss_dmg = lab.loc[~lab.hurricane_hit, f"total_dmg_{U}"].gt(0).mean()
log.write(f"Share with recorded damage: hit years {hit_dmg:.1%} vs other years {miss_dmg:.2%} "
          "(wind formula sanity check: should be far higher when hit)")
log.write("Highest annual hurricane-hit probability:")
top = p.nlargest(8, "p_hurricane_hit").reset_index().merge(c)
for r in top.itertuples():
    log.write(f"    {r.fips} {r.county_name:22s} {r.state}  {r.p_hurricane_hit:.3f}")

cy.to_csv(FINAL / "county_year.csv", index=False)
log.write(f"final/county_year.csv: {len(cy):,} rows x {cy.shape[1]} columns")

# Data dictionary rows for this table
from data_dictionary import DESCRIPTIONS
dd_path = FINAL / "data_dictionary.csv"
dd = pd.read_csv(dd_path)
dd = dd[dd.table != "county_year"]
rows, missing = [], []
for col in cy.columns:
    e = DESCRIPTIONS.get(("county_year", col)) or DESCRIPTIONS.get(("*", col)) \
        or DESCRIPTIONS.get(("counties", col)) or DESCRIPTIONS.get(("storm_county", col))
    if e is None:
        missing.append(col); e = ("", "", "")
    rows.append({"table": "county_year", "column": col, "dtype": str(cy[col].dtype),
                 "unit": e[0], "description": e[1], "source": e[2],
                 "pct_blank": round(cy[col].isna().mean() * 100, 1)})
log.check(not missing, f"every column documented (missing: {missing})")
pd.concat([dd, pd.DataFrame(rows)]).to_csv(dd_path, index=False)
log.close()
