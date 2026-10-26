"""Step 06: matched losses summed per storm x county.

In : clean/loss_records.csv, clean/loss_match.csv
Out: clean/losses.csv
"""
import pandas as pd
from common import *

log = StepLog("06_losses")
# --- 1. Attach each record's match decision; drop unmatched ---
lr = pd.read_csv(CLEAN / "loss_records.csv", dtype={"fips": str, "event_id": str})
mt = pd.read_csv(CLEAN / "loss_match.csv")
df = lr.merge(mt, on="record_id", how="left")
log.check(df.match_flag.notna().all(), "every loss record has a match decision")
drop = df.match_flag == "unmatched"
log.write(f"Unmatched records dropped: {drop.sum():,} "
          f"(${df.loc[drop, 'prop_dmg_2023'].sum() / 1e9:.2f}B property)")
df = df[~drop]

# --- 2. Sum records into one row per storm x county ---
# One groupby object g is reused for several results: sums of the amount
# columns, the row count (size), and how many were ambiguous (summing a
# True/False column counts the Trues).
SUM = ["prop_dmg_nominal", "prop_dmg_2023", "crop_dmg_nominal", "crop_dmg_2023",
       "crop_dmg_area_2023", "injuries_direct", "injuries_indirect",
       "deaths_direct", "deaths_indirect"]
df["is_ambiguous"] = df.match_flag.eq("ambiguous")
g = df.groupby(["storm_id", "fips"])
out = g[SUM].sum()
out["n_records"] = g.size()
out["n_ambiguous"] = g["is_ambiguous"].sum()
out = out.reset_index()

# --- 3. Check no dollars were lost; save ---
log.check(abs(out.prop_dmg_2023.sum() - df.prop_dmg_2023.sum()) < 1, "no dollars lost in summing")
log.write(f"Storm x county rows out: {len(out):,} across {out.storm_id.nunique()} storms")
out.to_csv(CLEAN / "losses.csv", index=False)
log.close()
