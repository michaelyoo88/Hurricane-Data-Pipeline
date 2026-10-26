"""Step 05: assign each loss record to a storm.

Rule: a storm qualifies if it was within MATCH_RADIUS_KM of the county at some
time during the record's dates (+/- DATE_PAD_DAYS). The closest qualifying
storm wins. Flags: unique / ambiguous / unmatched.

In : clean/loss_records.csv, clean/storm_county.csv
Out: clean/loss_match.csv
"""
import pandas as pd
from common import *

log = StepLog("05_loss_match")
lr = pd.read_csv(CLEAN / "loss_records.csv", dtype={"fips": str, "event_id": str},
                 usecols=["record_id", "fips", "begin_date", "end_date", "prop_dmg_2023"])
sc = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str},
                 usecols=["storm_id", "fips", "min_dist_km", "t_enter_500km", "t_exit_500km"],
                 parse_dates=["t_enter_500km", "t_exit_500km"])

# --- 1. Date window for each loss record ---
# [begin_date - pad, end_date + 1 day + pad]. Timedelta adds or subtracts a
# duration; the extra day makes end_date inclusive (covers that whole day).
pad = pd.Timedelta(days=DATE_PAD_DAYS)
lr["win_start"] = pd.to_datetime(lr.begin_date) - pad
lr["win_end"] = pd.to_datetime(lr.end_date) + pd.Timedelta(days=1) + pad

# --- 2. Candidate storms, then pick the closest ---
# Merging on fips pairs each record with every storm that passed its county.
# Two time ranges overlap when each one starts before the other ends.
# best: sort by distance, then drop_duplicates keeps the first (closest) row per
# record. n_cand counts how many storms qualified.
m = lr.merge(sc, on="fips")
m = m[(m.t_enter_500km <= m.win_end) & (m.t_exit_500km >= m.win_start)]
n_cand = m.groupby("record_id").size().rename("n_candidates")
best = m.sort_values("min_dist_km").drop_duplicates("record_id")[
    ["record_id", "storm_id", "min_dist_km"]].rename(columns={"min_dist_km": "match_dist_km"})

# --- 3. Flag each record: unmatched / unique / ambiguous ---
# pd.cut bins n_candidates: (-1, 0] unmatched, (0, 1] unique, (1, inf) ambiguous.
out = lr[["record_id", "prop_dmg_2023"]].merge(best, on="record_id", how="left")
out = out.merge(n_cand, on="record_id", how="left")
out["n_candidates"] = out.n_candidates.fillna(0).astype(int)
out["match_flag"] = pd.cut(out.n_candidates, [-1, 0, 1, 1e9],
                           labels=["unmatched", "unique", "ambiguous"]).astype(str)

# --- 4. Report match rates and the largest unmatched records; save ---
tot_n, tot_d = len(out), out.prop_dmg_2023.sum()
log.write("Match results (share of records / share of property dollars):")
for flag in ["unique", "ambiguous", "unmatched"]:
    s = out[out.match_flag == flag]
    log.write(f"    {flag:10s} {len(s):6,} records ({len(s) / tot_n:6.1%})   "
              f"${s.prop_dmg_2023.sum() / 1e9:7.2f}B ({s.prop_dmg_2023.sum() / tot_d:6.1%})")
log.check(out.loc[out.match_flag == "unmatched", "prop_dmg_2023"].sum() / tot_d < 0.05,
          "under 5% of property dollars unmatched")
top = (out[out.match_flag == "unmatched"].nlargest(10, "prop_dmg_2023")
       .merge(lr[["record_id", "begin_date"]], on="record_id"))
log.write("Largest unmatched records (check these by hand):")
for r in top.itertuples():
    log.write(f"    {r.record_id:20s} {r.begin_date}  ${r.prop_dmg_2023 / 1e6:,.1f}M")
out.drop(columns="prop_dmg_2023").to_csv(CLEAN / "loss_match.csv", index=False)
log.close()
