"""Step 00: Connecticut old counties -> 2022 planning regions.

NRI uses the 9 planning regions; the NY Fed data and Census 2020 centers of
population use the 8 old counties. Towns nest in both, so the Census town
crosswalk says which old county feeds which region. The population split is
then estimated by iterative proportional fitting (IPF): start from the number
of towns in each old-county x region pair and rescale until every old county
sums to its 2020 population (Census) and every region to its 2020 population
(NRI). Both totals are matched exactly.

In : source/ct_cou_to_cousub_crosswalk.txt, source/cenpop2020_county.txt,
     source/nri_counties.csv
Out: clean/ct_crosswalk.csv  (old_fips, region_fips, n_towns, pop_2020_est, share)
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("00_ct_crosswalk")
# --- 1. Read the town crosswalk; count towns per (old county, region) ---
# Each row is one town. sep="|" because the file is pipe-delimited; dtype=str
# keeps codes like "001" from becoming the number 1. The raw column names
# contain line breaks, so only the first line of each name is kept.
# groupby(...).size().unstack() builds a matrix: rows = old counties,
# columns = regions, cells = number of towns. This is the IPF "seed".
cw = pd.read_csv(SRC / "ct_cou_to_cousub_crosswalk.txt", sep="|", dtype=str,
                 encoding="utf-8-sig")
cw.columns = [c.split("\n")[0] for c in cw.columns]
cw["old_fips"] = cw.STATEFP + cw.OLD_COUNTYFP
cw["region_fips"] = cw.STATEFP + cw.NEW_COUNTYFP
seed = cw.groupby(["old_fips", "region_fips"]).size().unstack(fill_value=0)
log.write(f"Town rows: {len(cw)}; old counties: {seed.shape[0]}; regions: {seed.shape[1]}; "
          f"old-county x region pairs: {(seed > 0).sum().sum()}")

# --- 2. Population totals each side must match ---
# old_pop: 2020 population of the 8 old counties (Census).
# reg_pop: 2020 population of the 9 planning regions (NRI).
# .reindex() puts both in the same order as the seed matrix's rows / columns.
# The two sources disagree slightly, so regions are rescaled to the Census
# total; otherwise IPF could never satisfy both sets of totals.
cp = pd.read_csv(SRC / "cenpop2020_county.txt", dtype=str, encoding="utf-8-sig")
old_pop = (cp.assign(fips=cp.STATEFP + cp.COUNTYFP).set_index("fips")
           .POPULATION.astype(float).reindex(seed.index))
nri = pd.read_csv(SRC / "nri_counties.csv", dtype=str, encoding="utf-8-sig",
                  usecols=["State-County FIPS Code", "Population (2020)"])
reg_pop = (nri.set_index("State-County FIPS Code")["Population (2020)"]
           .astype(float).reindex(seed.columns))
log.write(f"Population totals: old counties {old_pop.sum():,.0f}; regions "
          f"{reg_pop.sum():,.0f} (gap {old_pop.sum() / reg_pop.sum() - 1:+.3%}); "
          "regions rescaled to the old-county total")
reg_pop = reg_pop * old_pop.sum() / reg_pop.sum()

# --- 3. Iterative proportional fitting (IPF) ---
# Alternately rescale rows to hit the old-county totals, then columns to hit
# the region totals, until both hold. m.sum(1) = row sums, m.sum(0) = column
# sums. [:, None] and [None, :] reshape a 1-D array into a column or a row so
# numpy broadcasts the multiply across the whole matrix. `_` = unused loop var.
m = seed.to_numpy(float)
for _ in range(1000):
    m *= (old_pop.to_numpy() / m.sum(1))[:, None]
    m *= (reg_pop.to_numpy() / m.sum(0))[None, :]
    if np.abs(m.sum(1) / old_pop.to_numpy() - 1).max() < 1e-9:
        break
flows = pd.DataFrame(m, index=seed.index, columns=seed.columns)
err = max(np.abs(flows.sum(1) / old_pop - 1).max(), np.abs(flows.sum(0) / reg_pop - 1).max())
log.check(err < 1e-3, f"IPF matches old-county and region population totals (max error {err:.3%})")

# --- 4. Back to one row per pair; compute each pair's share ---
# .stack() turns the matrix back into a long table (old_fips, region_fips, value).
# share = fraction of an old county's population living in each region.
# groupby().transform("sum") puts the group total on every row so we can divide
# row by row. Steps 01 and 04 use this share to split old-county data.
out = (flows.stack().rename("pop_2020_est").reset_index()
       .merge(seed.stack().rename("n_towns").reset_index()))
out = out[out.n_towns > 0].copy()
out["share"] = out.pop_2020_est / out.groupby("old_fips").pop_2020_est.transform("sum")
log.check(np.allclose(out.groupby("old_fips").share.sum(), 1), "shares sum to 1 per old county")
for r in out.itertuples():
    log.write(f"    {r.old_fips} -> {r.region_fips}: {r.n_towns:2d} towns, share {r.share:.3f}")
out.round({"pop_2020_est": 0, "share": 6}).to_csv(CT_CROSSWALK, index=False)
log.close()
