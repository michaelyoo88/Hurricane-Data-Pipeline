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
# Reads final/ rather than clean/: this step builds on the published tables.
sc = pd.read_csv(FINAL / "storm_county.csv", dtype={"fips": str})
c = pd.read_csv(FINAL / "counties.csv", dtype={"fips": str}, usecols=["fips", "state", "county_name"])

# 1. Wind each county felt from each storm (simple Rankine vortex)
# clip(lower=RMW_KM) treats any distance under 40 km as 40 km, so counties
# inside the radius of max wind get the full Vmax; farther out, wind decays as
# (40 / d) ** 0.5. Vectorized: one line computes every storm x county row.
d = sc.min_dist_km.clip(lower=RMW_KM)
sc["wind_ms"] = sc.vmax_at_min_ms * (RMW_KM / d) ** RANKINE_EXP

# 2. Collapse storms to county x year
# Several results from one groupby. apply(lambda w: ...) counts the storms over
# a wind threshold in each group. sum(min_count=1) returns NaN (not 0) when all
# values are blank, so unlabeled years stay unknown.
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
# MultiIndex.from_product = every combination of county and year (a cross
# join). Left-merging the storm summary onto it leaves NaN in quiet years,
# which become 0 wind / 0 storms.
years = range(STORM_YEAR_MIN, STORM_YEAR_MAX + 1)
grid = pd.MultiIndex.from_product([c.fips, years], names=["fips", "year"]).to_frame(index=False)
cy = grid.merge(c, on="fips").merge(cy, on=["fips", "year"], how="left")
cy["max_wind_ms"] = cy.max_wind_ms.fillna(0).round(2)
for col in ["n_storms_500km", "n_storms_ts_force", "n_storms_hurricane_force"]:
    cy[col] = cy[col].fillna(0).astype(int)

# 4. Damage: 0 in labeled years with no record, blank before labels exist
# Same labeled / unlabeled rule as step 07. Population loss turns deaths
# (+ injuries / 10) into dollars using the value of a statistical life.
cy["has_labels"] = cy.year.between(LABEL_YEAR_MIN, LABEL_YEAR_MAX)
dmg = [f"prop_dmg_{U}", f"crop_dmg_{U}", "deaths_direct", "injuries_direct"]
cy.loc[cy.has_labels, dmg] = cy.loc[cy.has_labels, dmg].fillna(0)
cy.loc[~cy.has_labels, dmg] = np.nan
cy[f"total_dmg_{U}"] = cy[f"prop_dmg_{U}"] + cy[f"crop_dmg_{U}"]
cy[f"pop_loss_{U}"] = (cy.deaths_direct + cy.injuries_direct / 10) * VSL_USD

# 5. Hit indicators and annual probabilities (share of the 51 years)
# The mean of a True/False column is the share of Trues, so averaging
# hurricane_hit over a county's 51 years gives its annual hit probability.
# It is merged back so every row carries its county's probability.
cy["hurricane_hit"] = cy.max_wind_ms >= HURRICANE_MS
cy["ts_hit"] = cy.max_wind_ms >= TROPICAL_STORM_MS
n_years = len(years)
p = cy.groupby("fips").agg(p_hurricane_hit=("hurricane_hit", "mean"),
                           p_ts_hit=("ts_hit", "mean"),
                           mean_hurricanes_per_yr=("n_storms_hurricane_force", "mean"))
cy = cy.merge(p.round(4).reset_index(), on="fips")

# Final column order and sort.
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
# Same lookup as step 07, with extra fallbacks to the counties / storm_county
# entries for reused columns. Old county_year rows are removed first so a
# rerun doesn't duplicate them. A trailing backslash continues a long line.
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
