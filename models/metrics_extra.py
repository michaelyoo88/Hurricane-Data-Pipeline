"""Extra checks on the county EAL metrics (total damage, held-out predictions).

1. Noise floor: how well can ANY model match a county's recorded 25-year average?
   Split the 25 seasons into two halves (odd vs even years) and compare a county's recorded
   EAL in one half with the other. A perfect model can't beat the recorded data's own consistency.
2. Tighter bands: within 1.5x, 2x, 3x.
3. Calibration by risk group: sort counties into 5 groups by predicted EAL, compare predicted and
   recorded dollars per group. Adding counties together smooths out the luck of single storms.
4. Deviance explained (D2): the standard skill score for skewed, zero-heavy data (Tweedie, power 1.5),
   vs a simple benchmark: one national loss ratio x each county's building value. 0 = no better
   than the benchmark, 1 = perfect.
Out: results/metrics_extra.csv, results/calibration_by_group.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_EAL, P = 10_000, 1.5

d = pd.read_csv(HERE / "data" / "county_year_preds.csv", dtype={"fips": str})
mt = pd.read_csv(HERE / "data" / "model_table.csv", dtype={"fips": str})
bldg = mt.groupby("fips").bldg_value.mean()


def within(ratio, k):
    return ((ratio >= 1 / k) & (ratio <= k)).mean()


def tweedie_dev(y, mu):
    mu = np.maximum(mu, 1e-9)
    return 2 * np.sum(y ** (2 - P) / ((1 - P) * (2 - P)) - y * mu ** (1 - P) / (1 - P) + mu ** (2 - P) / (2 - P))


# 1. noise floor from the recorded data itself (same for every model)
one = d[d.strategy == d.strategy.iloc[0]]
half = one.assign(h=one.year % 2).groupby(["fips", "h"]).total_rec.sum().unstack(fill_value=0)
half = half * 25 / one.assign(h=one.year % 2).groupby("h").year.nunique().reindex([0, 1]).to_numpy() / 25   # per-year average in each half
full = one.groupby("fips").total_rec.sum() / 25
m = full.reindex(half.index) >= MIN_EAL
r = (half[0][m].clip(lower=1) / half[1][m].clip(lower=1))
print(f"NOISE FLOOR: recorded EAL, even years vs odd years, {m.sum()} counties:")
print(f"  within 1.5x {within(r, 1.5):.0%}, within 2x {within(r, 2):.0%}, within 3x {within(r, 3):.0%}, "
      f"typical factor {10 ** np.median(np.abs(np.log10(r))):.1f}x, counties with $0 in one half {(half[m].min(axis=1) == 0).mean():.0%}")

rows, cal = [], []
for name, g in d.groupby("strategy"):
    eal = g.groupby("fips")[["total_pred", "total_rec"]].sum() / 25
    m = eal.total_rec >= MIN_EAL
    ratio = eal.total_pred[m].clip(lower=1) / eal.total_rec[m]
    # benchmark: one national ratio x building value (same counties)
    b = bldg.reindex(eal.index).fillna(0)
    bench = b * eal.total_rec.sum() / b.sum()
    d2 = 1 - tweedie_dev(eal.total_rec.to_numpy(), eal.total_pred.to_numpy()) / tweedie_dev(eal.total_rec.to_numpy(), bench.to_numpy())
    # calibration by 5 risk groups (all counties with any prediction)
    e = eal[eal.total_pred > 0].copy()
    e["group"] = pd.qcut(e.total_pred.rank(method="first"), 5, labels=["lowest", "low", "middle", "high", "highest"])
    c = e.groupby("group", observed=True)[["total_pred", "total_rec"]].sum()
    c["ratio"] = c.total_pred / c.total_rec
    cal.append(c.ratio.rename(name))
    rows.append({"strategy": name, "within_1.5x": within(ratio, 1.5), "within_2x": within(ratio, 2), "within_3x": within(ratio, 3),
                 "deviance_explained_D2": d2, **{f"cal_{k}": v for k, v in c.ratio.items()}})
out = pd.DataFrame(rows).set_index("strategy")
out.round(3).to_csv(HERE / "results" / "metrics_extra.csv")
pd.concat(cal, axis=1).T.round(2).to_csv(HERE / "results" / "calibration_by_group.csv")
pd.set_option("display.width", 220)
print("\nper model (total damage, county EAL):")
print(out.round(2).to_string())
