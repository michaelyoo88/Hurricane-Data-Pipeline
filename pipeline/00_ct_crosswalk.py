"""Step 00 - Connecticut: old counties -> 2022 planning regions.

Why: FEMA's NRI uses Connecticut's 9 planning regions (2022), but the NY Fed
damage data and Census 2020 use the 8 old counties. We need a way to move
numbers from old counties to regions.

How: every town belongs to exactly one old county and one region, so the Census
town crosswalk tells us which county feeds which region. To decide how much of
each county's population goes to each region, we start from the number of towns
in each pair and rescale (iterative proportional fitting) until the totals match
both the old-county populations (Census) and the region populations (NRI).

Out: clean/ct_crosswalk.csv  (old_fips, region_fips, share); shares sum to 1 per old county
"""
import pandas as pd
from common import *

# Towns: header cells contain line breaks, so keep only the first line of each
towns = pd.read_csv(SOURCE / "ct_cou_to_cousub_crosswalk.txt", sep="|", dtype=str,
                    encoding="utf-8-sig")
towns.columns = [c.split("\n")[0] for c in towns.columns]
towns["old_fips"] = towns.STATEFP + towns.OLD_COUNTYFP
towns["region_fips"] = towns.STATEFP + towns.NEW_COUNTYFP

# Starting guess: number of towns in each old-county x region pair
counts = pd.crosstab(towns.old_fips, towns.region_fips).astype(float)

# The two population totals we want to match
cenpop = pd.read_csv(SOURCE / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
cenpop["fips"] = cenpop.STATEFP + cenpop.COUNTYFP
old_pop = cenpop.set_index("fips").POPULATION.astype(float).reindex(counts.index)

nri = pd.read_csv(SOURCE / "nri_counties.csv", dtype=str, encoding="utf-8-sig")
region_pop = (nri.set_index("State-County FIPS Code")["Population (2020)"]
              .astype(float).reindex(counts.columns))
region_pop = region_pop * old_pop.sum() / region_pop.sum()   # totals differ by 0.05%

# Iterative proportional fitting: alternately fix the row totals, then the column totals
m = counts.to_numpy()
for _ in range(500):
    m = m * (old_pop.to_numpy() / m.sum(axis=1))[:, None]
    m = m * (region_pop.to_numpy() / m.sum(axis=0))[None, :]

flows = pd.DataFrame(m, index=counts.index, columns=counts.columns)
shares = flows.div(flows.sum(axis=1), axis=0)          # each row now sums to 1
out = shares.stack().rename("share").reset_index()
out = out[out.share > 0]
out.to_csv(CLEAN / "ct_crosswalk.csv", index=False)
print(f"00 ct_crosswalk: {len(out)} old-county -> region links")
