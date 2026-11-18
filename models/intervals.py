"""Turn "within 3x" into intervals (total damage, county EAL, held-out predictions).

A. How sure are we of the share within 3x? Bootstrap the 25 seasons: draw 25 seasons with
   replacement, recompute every county's predicted and recorded EAL, recompute the share.
   1,000 draws -> 95% interval. (Seasons are the unit because storm luck comes by season.)
B. How wide a band does each county need? From the held-out errors, recorded / predicted EAL
   per county: the band that holds 50%, 80% and 90% of counties, e.g. "between 3x below and
   2x above the prediction". This is an empirical prediction interval for a county.
Counties scored: recorded EAL >= $10k per year. Out: results/intervals.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_EAL, N_BOOT, SEED = 10_000, 1000, 42
NAMES = {"3 by damage type": "4. by damage type", "0 basic two-part": "1. two-part GLM", "2 top-down": "3. top-down",
         "1 damage table": "2. damage table", "4a weights halfway": "5. halfway weights",
         "4b weights by dollars": "6. dollar weights", "4c tier rescale": "7. tier correction",
         "6 county-year two-part": "8. county-year"}

d = pd.read_csv(HERE / "data" / "county_year_preds.csv", dtype={"fips": str})
years = np.arange(2000, 2025)
rng = np.random.default_rng(SEED)
draws = [rng.choice(years, 25, replace=True) for _ in range(N_BOOT)]
rows = []
for name, g in d.groupby("strategy"):
    # county x year grid of predicted and recorded (0 where no storm that year)
    P = g.pivot_table(index="fips", columns="year", values="total_pred", aggfunc="sum").reindex(columns=years, fill_value=0).fillna(0)
    R = g.pivot_table(index="fips", columns="year", values="total_rec", aggfunc="sum").reindex(columns=years, fill_value=0).fillna(0)
    Pv, Rv = P.to_numpy(), R.to_numpy()

    def share_within3(cols):
        ep, er = Pv[:, cols].mean(axis=1), Rv[:, cols].mean(axis=1)
        m = er >= MIN_EAL
        ratio = np.maximum(ep[m], 1) / er[m]
        return ((ratio >= 1 / 3) & (ratio <= 3)).mean()

    point = share_within3(np.arange(25))
    boot = np.array([share_within3(yrs - 2000) for yrs in draws])
    # B. empirical band from the full-period errors
    ep, er = Pv.mean(axis=1), Rv.mean(axis=1)
    m = er >= MIN_EAL
    log_err = np.log10(er[m] / np.maximum(ep[m], 1))          # + = recorded above prediction
    row = {"model": NAMES[name], "within_3x": point,
           "within_3x_95ci_low": np.quantile(boot, 0.025), "within_3x_95ci_high": np.quantile(boot, 0.975)}
    for cover in [0.5, 0.8, 0.9]:
        lo, hi = np.quantile(log_err, [(1 - cover) / 2, 1 - (1 - cover) / 2])
        row[f"band_{int(cover * 100)}_low_x"] = 10 ** lo     # recorded can be this many times the prediction (below 1 = lower)
        row[f"band_{int(cover * 100)}_high_x"] = 10 ** hi
    rows.append(row)
out = pd.DataFrame(rows).sort_values("within_3x", ascending=False)
out.round(3).to_csv(HERE / "results" / "intervals.csv", index=False)
pd.set_option("display.width", 220)
print("A. share of counties within 3x, with 95% interval (bootstrap over seasons):")
print(out[["model", "within_3x", "within_3x_95ci_low", "within_3x_95ci_high"]].round(2).to_string(index=False))
print("\nB. band around a county's prediction that holds 50% / 80% / 90% of counties (recorded = prediction x factor):")
fmt = lambda lo, hi: f"{1/lo:.0f}x below to {hi:.1f}x above" if lo < 1 else f"{lo:.1f}x to {hi:.1f}x above"   # noqa: E731
for _, r in out.iterrows():
    print(f"  {r.model:20s} 50%: {fmt(r.band_50_low_x, r.band_50_high_x):28s} 80%: {fmt(r.band_80_low_x, r.band_80_high_x):28s} "
          f"90%: {fmt(r.band_90_low_x, r.band_90_high_x)}")
