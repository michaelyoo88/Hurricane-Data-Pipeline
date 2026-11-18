"""Compare finalists with error bars, on the DEV storms or (round 6) on ALL storms.

    python3 sandbox/compare.py                  # round-5 finalists, dev storms (as before)
    python3 sandbox/compare.py round6 dev       # round-6 finalists, dev storms
    python3 sandbox/compare.py round6 all       # round-6 finalists, fresh 3 x 5-fold over all 194 storms

For each finalist: held-out predictions averaged over the 3 repeated splits, then a bootstrap over
storms (1,000 draws) for the deviance explained (d2) and the tier miss, plus the PAIRED difference
vs the reference model on the same resampled storms. A difference whose 95% interval excludes 0 is real.
Out: results/finalists_dev.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys

from lib import BASE, J2, POWER, REGIONS, REST, TIERS, capped, load, oof

HERE = Path(__file__).resolve().parent
N_BOOT, SEED = 1000, 42
SIMPLE = ["wind_cap", "coastal_x_wind_cap"] + REST
FINALISTS = {
    "A0 old base (uncapped wind)": {"features": BASE},
    "A1 damage table": {"kind": "table", "keys": ["wind_bin", "coast"]},
    "C4 capped wind (7 features)": {"features": capped()},
    "G2 capped, one wind term (6)": {"features": SIMPLE},
    "D7 capped + right of track (9)": {"features": capped() + ["right_of_track", "coastal_x_right"]},
    "E1 capped, by damage type": {"features": capped(), "by_type": True},
    "E2 capped + tier correction": {"features": capped(), "rescale": "storm_strength"},
}
REFERENCE = "C4 capped wind (7 features)"
ROUND6 = {
    "K0 J2 reference (4 inputs)": {"features": J2},
    "K1 asymmetric wind": {"features": ["wa_cap", "coastal_x_wa_cap", "log_dist_km", "coastal"]},
    "K2 Tweedie GLM, one equation": {"kind": "tweedie", "features": J2},
    "K4b Emanuel curve + floor": {"kind": "emanuel", "floor": True},
    "K5 J2 + region": {"features": J2 + list(REGIONS)},
    "K6 J2 + latitude": {"features": J2 + ["pop_lat"]},
    "K9 average: J2 + by type": {"kind": "average", "members": [{"features": J2}, {"features": capped(), "by_type": True}]},
    "K12 average: J2+lat + by type": {"kind": "average", "members": [{"features": J2 + ["pop_lat"]}, {"features": capped(), "by_type": True}]},
}


def row_deviance(y, mu):
    mu = np.maximum(mu, 1e-12)
    return 2 * (y ** (2 - POWER) / ((1 - POWER) * (2 - POWER)) - y * mu ** (1 - POWER) / (1 - POWER) + mu ** (2 - POWER) / (2 - POWER))


def storm_table(d):
    """Everything the bootstrap needs, added up per storm."""
    d = d.assign(dev_model=row_deviance(d.rec / 1e6, d.pred / 1e6), dev_null=row_deviance(d.rec / 1e6, d.null / 1e6))
    return d.groupby("storm_id").agg(tier=("storm_strength", "first"), pred=("pred", "sum"), rec=("rec", "sum"),
                                     dev_model=("dev_model", "sum"), dev_null=("dev_null", "sum"))


def stats(st):
    t = st.groupby("tier")[["pred", "rec"]].sum().reindex(TIERS)
    return {"d2": 1 - st.dev_model.sum() / st.dev_null.sum(), "pred_rec": st.pred.sum() / st.rec.sum(),
            "tier_miss": 10 ** np.mean(np.abs(np.log10(t.pred / t.rec)))}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "round5"
    scope = sys.argv[2] if len(sys.argv) > 2 else "dev"
    if which == "round6":
        FINALISTS, REFERENCE = ROUND6, "K0 J2 reference (4 inputs)"
    df = load()
    tables = {}
    for label, spec in FINALISTS.items():
        runs = [oof(df, spec, r, scope) for r in range(3)]
        d = runs[0].assign(pred=np.mean([r.pred for r in runs], axis=0), null=np.mean([r.null for r in runs], axis=0))
        tables[label] = storm_table(d)
    lg = HERE / "results" / "lgbm_oof.csv"
    if lg.exists() and which == "round5":                                         # LightGBM predictions come from lgbm_challenger.py
        l = pd.read_csv(lg, dtype={"fips": str})
        base = oof(df, FINALISTS[REFERENCE], 0)[["storm_id", "storm_strength", "fips", "rec"]]
        nulls = np.mean([oof(df, {"kind": "table", "keys": ["coast"]}, r).null for r in range(3)], axis=0)
        cols = [c for c in l.columns if c.startswith("H1")]
        d = base.assign(null=nulls).merge(l[["storm_id", "fips"]].assign(pred=l[cols].mean(axis=1)), on=["storm_id", "fips"])
        tables["H1 LightGBM monotone"] = storm_table(d)

    rng = np.random.default_rng(SEED)
    storms = tables[REFERENCE].index.to_numpy()
    draws = [rng.choice(storms, len(storms), replace=True) for _ in range(N_BOOT)]
    boot = {m: pd.DataFrame([stats(t.loc[s]) for s in draws]) for m, t in tables.items()}
    rows = []
    for m, t in tables.items():
        point, b, ref = stats(t), boot[m], boot[REFERENCE]
        diff = b.d2 - ref.d2
        rows.append({"model": m, "d2": point["d2"], "d2_95ci": f"{b.d2.quantile(.025):.2f} to {b.d2.quantile(.975):.2f}",
                     "d2_minus_ref": point["d2"] - stats(tables[REFERENCE])["d2"],
                     "diff_95ci": f"{diff.quantile(.025):+.3f} to {diff.quantile(.975):+.3f}",
                     "clearly_different": bool(diff.quantile(.025) > 0 or diff.quantile(.975) < 0),
                     "pred_rec": point["pred_rec"], "pred_rec_95ci": f"{b.pred_rec.quantile(.025):.2f} to {b.pred_rec.quantile(.975):.2f}",
                     "tier_miss": point["tier_miss"], "tier_miss_95ci": f"{b.tier_miss.quantile(.025):.2f} to {b.tier_miss.quantile(.975):.2f}"})
    out = pd.DataFrame(rows)
    name = "finalists_dev.csv" if which == "round5" else f"finalists_round6_{scope}.csv"
    out.round(3).to_csv(HERE / "results" / name, index=False)
    pd.set_option("display.width", 250)
    print(f"{scope} storms: {len(storms)}; reference = {REFERENCE}")
    print(out.round(3).to_string(index=False))
