"""Round 8: five model families scored on the agreed error metrics (2026-11-18).

    python3 sandbox/round8.py fit      # held-out predictions for every model -> data/round8/*.npz  (~10-15 min)
    python3 sandbox/round8.py score [seconds]   # optional time budget per call; metrics + paired storm bootstrap -> results/round8_*.csv, round8_summary.png

Design (same as round 6/7): 3 repeats x 5-fold over all 194 storms (fold_all_r*), whole storms held out.
Every model trains on the KNN-filled property labels (prop_usd_imp, round 7) and is scored against the
RECORDED property damage. Held-out predictions are averaged over the 3 repeats. Settings fixed in advance.

Metrics (agreed 2026-11-18)
  mae_eal     MAE of county EAL ($M per year; 25-season average, same exposure basis as the records), all 2,560 counties
  bias        total predicted / total recorded; also per storm tier (TD/TS, Cat 1-2, Cat 3+)
  gini        normalized Gini of county EAL: are the dollars in the right counties? (1 = perfect order)
  pr_auc      PR-AUC for "any recorded damage" at storm x county level (score = predicted loss ratio)
  d2          Tweedie deviance explained (power 1.5) vs one national loss ratio, storm x county level
Uncertainty: 1,000 bootstrap draws of whole storms; the PAIRED difference vs the reference (R0 = K12, the
current model) on the same draws. A difference whose 95% interval excludes 0 is real.
Distributional models also get interval coverage: share of damaged rows whose recorded loss ratio falls in
the model's 10-90% band (target 0.80), and the same for storm totals (simulated, target 0.80).

Families
  T  Tweedie GLM (one equation)            power, inputs, total recalibration
  G  LightGBM, Tweedie objective           power, leaves, min rows per leaf, storm vs dollar weights, inputs
  Y  Hybrid: tree classifier x GLM size    classifier (XGBoost / random forest), classifier inputs
  D  Distributional (two-part, full loss distribution)  gamma vs lognormal size, constant vs modeled spread
  E  Ensembles / stacking                  members, equal vs learned weights (nested CV, no leakage)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize

from knn_impute import add_imputation
from lib import J2, POWER, TIERS, capped, fit_gamma, fit_logistic, fit_tweedie, load, two_part

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "round8"
SEED, YEARS = 42, 25
N_BOOT = int(__import__("os").environ.get("N_BOOT", 1000))
LAT = J2 + ["pop_lat"]                                              # inputs of K12's first member
RAW = ["wind_ms", "log_wind_exposure", "log_dist_km", "coastal", "log_density", "pop_lat"]
RICH = RAW + ["right_of_track", "log_fwd_speed", "dvmax_24h", "storm_peak_wind", "log_footprint"]
DIRECTION = {"wind_ms": 1, "log_wind_exposure": 1, "log_dist_km": -1, "storm_peak_wind": 1}
TARGET = "prop_usd_imp"


# ---------------------------------------------------------------- shared pieces
def prep():
    df = load()
    st = pd.read_csv(HERE / "data" / "knn_stats.csv", dtype={"fips": str})
    df = df.merge(st[["storm_id", "fips", "nb_share", "nb_median_pos"]], on=["storm_id", "fips"], how="left", validate="one_to_one")
    add_imputation(df, 25.0, "expected")
    return df


def design(df, fit, feats):
    X = df[feats].to_numpy(float)
    m, s = X[fit].mean(0), X[fit].std(0)
    return np.column_stack([np.ones(len(X)), (X - m) / np.where(s > 0, s, 1)])


def ratio_of(df, col=TARGET):
    return df[col].to_numpy(float) / df.bldg_value.to_numpy(float)


def storm_weights(df, mask):
    """Every storm counts equally: row weight = exposure / its storm's exposure (as in two_part, alpha = 1)."""
    e = df.bldg_value.to_numpy(float)[mask]
    return e / pd.Series(e).groupby(df.storm_id.to_numpy()[mask]).transform("sum").to_numpy()


def recalibrate(df, fit, ratio, col=TARGET):
    """One factor so predicted dollars on the training storms equal recorded dollars."""
    e = df.bldg_value.to_numpy(float)
    return ratio * df[col].to_numpy(float)[fit].sum() / (ratio * e)[fit].sum()


def by_type(df, fit):
    return two_part(df, fit, "prop_wind_usd_imp", features=capped()) \
        + two_part(df, fit, "prop_surge_usd", features=capped(), rows=(df.coastal == 1).to_numpy()) \
        + two_part(df, fit, "prop_rain_usd", features=capped())


def lat_member(df, fit):
    return two_part(df, fit, TARGET, features=LAT)


def k12(df, fit):
    return np.minimum((lat_member(df, fit) + by_type(df, fit)) / 2, 1.0)


# ---------------------------------------------------------------- T: Tweedie GLM
def tweedie(feats=LAT, power=1.5, recal=False):
    def f(df, fit):
        X1, y = design(df, fit, feats), ratio_of(df)
        b = fit_tweedie(X1[fit], y[fit], storm_weights(df, fit), np.r_[0.0, np.ones(len(feats))], power=power)
        r = np.exp(np.clip(X1 @ b, -50, 5))
        return np.minimum(recalibrate(df, fit, r) if recal else r, 1.0)
    return f


# ---------------------------------------------------------------- G: LightGBM
def lgbm(feats=RAW, power=1.5, leaves=7, min_rows=50, weights="storm", recal=False):
    import lightgbm as lgb

    def f(df, fit):
        w = storm_weights(df, fit) if weights == "storm" else df.bldg_value.to_numpy(float)[fit]
        m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=power, learning_rate=0.05, n_estimators=300,
                              num_leaves=leaves, min_child_samples=min_rows, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=4,
                              monotone_constraints=[DIRECTION.get(c, 0) for c in feats])
        m.fit(df.loc[fit, feats], ratio_of(df)[fit], sample_weight=w / w.mean())
        r = m.predict(df[feats])
        return np.minimum(recalibrate(df, fit, r) if recal else r, 1.0)
    return f


# ---------------------------------------------------------------- Y: hybrid
def size_glm(df, fit, feats=LAT, col=TARGET):
    """Part 2 of the current model: gamma GLM on loss ratio of damaged rows, storms weighted equally."""
    y = ratio_of(df, col)
    pos = fit & (y > 0)
    X1 = design(df, fit, feats)
    b = fit_gamma(X1[pos], y[pos], storm_weights(df, pos), np.r_[0.0, np.ones(len(feats))])
    return np.exp(np.clip(X1 @ b, -40, 10))


def chance_tree(df, fit, kind, feats):
    y = (df[TARGET].to_numpy() > 0).astype(int)
    X = df[feats].to_numpy(float)
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=10, n_jobs=4, random_state=SEED, eval_metric="logloss")
    else:
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt", n_jobs=4, random_state=SEED)
    m.fit(X[fit], y[fit])
    return m.predict_proba(X)[:, 1]


def hybrid(kind="xgb", feats=RICH, with_type=False):
    def f(df, fit):
        r = np.minimum(chance_tree(df, fit, kind, feats) * size_glm(df, fit), 1.0)
        return np.minimum((r + by_type(df, fit)) / 2, 1.0) if with_type else r
    return f


# ---------------------------------------------------------------- D: distributional
def chance_glm(df, fit, feats=LAT):
    X1 = design(df, fit, feats)
    b = fit_logistic(X1[fit], (df[TARGET].to_numpy()[fit] > 0).astype(float), np.r_[0.0, np.ones(len(feats))])
    return 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))


def dist_gamma(feats=LAT):
    """Current two-part gamma with its fitted shape: same mean as K12's first member, plus a full distribution."""
    def f(df, fit):
        p, mu = chance_glm(df, fit, feats), size_glm(df, fit, feats)
        y = ratio_of(df); pos = fit & (y > 0)
        w = storm_weights(df, pos)
        # gamma shape by maximum likelihood given the fitted means: log k - digamma(k) = weighted mean deviance / 2
        dbar = np.sum(w * ((y[pos] - mu[pos]) / mu[pos] - np.log(y[pos] / mu[pos]))) / np.sum(w)
        from scipy.special import digamma
        from scipy.optimize import brentq
        shape = brentq(lambda k: np.log(k) - digamma(k) - dbar, 1e-4, 1e4)
        q10, q90 = sps.gamma.ppf([[.1], [.9]], shape, scale=mu / shape)
        return {"ratio": np.minimum(p * mu, 1.0), "chance": p, "q10": q10, "q90": q90,
                "draw": ("gamma", shape, mu)}
    return f


def dist_lognormal(feats=LAT, spread="constant", recal=False):
    """Two-part with a lognormal size: log loss ratio ~ Normal(m, s). Mean = exp(m + s^2/2).
    spread: 'constant' (one s) or 'modeled' (log s^2 regressed on the same inputs, GAMLSS-style)."""
    def f(df, fit):
        p = chance_glm(df, fit, feats)
        y = ratio_of(df); pos = fit & (y > 0)
        X1, w = design(df, fit, feats), storm_weights(df, pos)
        A, ly, W = X1[pos], np.log(y[pos]), w / w.mean()
        pen = np.diag(np.r_[0.0, np.ones(len(feats))])
        b = np.linalg.solve((A * W[:, None]).T @ A + pen, (A * W[:, None]).T @ ly)
        m = X1 @ b
        res2 = (ly - A @ b) ** 2
        if spread == "constant":
            s = np.full(len(df), np.sqrt(np.sum(W * res2) / np.sum(W)))
        else:
            g = np.linalg.solve((A * W[:, None]).T @ A + pen, (A * W[:, None]).T @ (np.log(res2 + 1e-8) + 1.27))
            s = np.sqrt(np.exp(np.clip(X1 @ g, np.log(0.3 ** 2), np.log(3.0 ** 2))))
        mean = np.exp(np.clip(m + s ** 2 / 2, -40, 0))
        ratio = recalibrate(df, fit, p * mean) if recal else p * mean
        return {"ratio": np.minimum(ratio, 1.0), "chance": p, "q10": np.exp(m - 1.2816 * s),
                "q90": np.exp(m + 1.2816 * s), "draw": ("lognormal", m, s)}
    return f


# ---------------------------------------------------------------- E: ensembles
def average(*members):
    def f(df, fit):
        return np.minimum(np.mean([m(df, fit) for m in members], axis=0), 1.0)
    return f


def row_dev(y, mu, p=POWER):
    mu = np.maximum(mu, 1e-12)
    return 2 * (y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p))


WEIGHTS_LOG = []


def stack(members, names, inner_k=4):
    """Stacking with weights learned only from the TRAINING storms: inner storm-held-out folds give each member's
    held-out predictions; non-negative weights summing to 1 minimize Tweedie deviance (in $M) on them."""
    def f(df, fit):
        e, y = df.bldg_value.to_numpy(float), df.prop_usd.to_numpy(float) / 1e6
        storms = np.unique(df.storm_id.to_numpy()[fit])
        rng = np.random.default_rng(SEED + len(storms))
        grp = dict(zip(storms, rng.permutation(len(storms)) % inner_k))
        g = df.storm_id.map(grp).to_numpy()
        P = np.zeros((len(df), len(members)))
        for k in range(inner_k):
            ifit, ihold = fit & (g != k), fit & (g == k)
            for j, m in enumerate(members):
                P[ihold, j] = m(df, ifit)[ihold]
        D = P[fit] * e[fit, None] / 1e6
        yy = y[fit]

        def loss(wt):
            return row_dev(yy, D @ wt).sum()
        n = len(members)
        res = minimize(loss, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda wt: wt.sum() - 1}])
        wt = np.clip(res.x, 0, None); wt /= wt.sum()
        WEIGHTS_LOG.append(dict(zip(names, wt.round(3))))
        return np.minimum(np.column_stack([m(df, fit) for m in members]) @ wt, 1.0)
    return f


MODELS = {
    "R0 K12 reference (current model)": k12,
    # Tweedie GLM
    "T1 Tweedie GLM p=1.5": tweedie(power=1.5),
    "T2 Tweedie GLM p=1.2": tweedie(power=1.2),
    "T3 Tweedie GLM p=1.8": tweedie(power=1.8),
    "T4 Tweedie GLM p=1.5 + recalibrated total": tweedie(power=1.5, recal=True),
    "T5 Tweedie GLM p=1.5, 7 capped inputs + recal": tweedie(capped(), power=1.5, recal=True),
    # gradient boosting
    "G1 LightGBM, dollar weights (old H1 + lat)": lgbm(weights="dollar"),
    "G2 LightGBM, storm weights": lgbm(),
    "G3 LightGBM, storm weights + recal": lgbm(recal=True),
    "G4 LightGBM, p=1.3, 15 leaves, 200 rows/leaf + recal": lgbm(power=1.3, leaves=15, min_rows=200, recal=True),
    "G5 LightGBM, rich inputs + recal": lgbm(RICH, recal=True),
    # hybrid
    "Y1 Hybrid: XGBoost (rich) x gamma size": hybrid("xgb", RICH),
    "Y2 Hybrid: random forest (rich) x gamma size": hybrid("rf", RICH),
    "Y3 Hybrid: XGBoost (4 inputs + lat) x gamma size": hybrid("xgb", LAT),
    "Y4 Hybrid Y1 averaged with by-type model": hybrid("xgb", RICH, with_type=True),
    # distributional
    "D0 Two-part gamma (K12 member 1) + distribution": dist_gamma(),
    "D1 Two-part lognormal, constant spread": dist_lognormal(spread="constant"),
    "D2 Two-part lognormal, modeled spread": dist_lognormal(spread="modeled"),
    "D3 Two-part lognormal, modeled spread + recal": dist_lognormal(spread="modeled", recal=True),
    # ensembles
    "E1 Equal average: K12 members + Tweedie + LightGBM": average(lat_member, by_type, tweedie(power=1.5, recal=True), lgbm(recal=True)),
    "E2 Equal average: K12 members + hybrid Y1": average(lat_member, by_type, hybrid("xgb", RICH)),
    "E3 Stacked (learned weights, nested CV)": stack([lat_member, by_type, tweedie(power=1.5, recal=True), lgbm(recal=True), hybrid("xgb", RICH)],
                                                     ["lat_member", "by_type", "tweedie", "lightgbm", "hybrid"]),
}
REFERENCE = "R0 K12 reference (current model)"


def slug(name):
    return name.split()[0]


# ---------------------------------------------------------------- fit stage
def run_fit(only=None):
    df = prep()
    OUT.mkdir(parents=True, exist_ok=True)
    e = df.bldg_value.to_numpy(float)
    for name, fn in MODELS.items():
        if only and slug(name) not in only:
            continue
        t, preds, extra = time.time(), [], {}
        WEIGHTS_LOG.clear()
        for r in range(3):
            fold, pred = df[f"fold_all_r{r}"].to_numpy(), np.zeros(len(df))
            q = {k: np.zeros(len(df)) for k in ("chance", "q10", "q90", "a", "b")}
            for k in range(5):
                held, fit = fold == k, fold != k
                out = fn(df, fit)
                if isinstance(out, dict):
                    for c in ("chance", "q10", "q90"):
                        q[c][held] = out[c][held]
                    kind, a, b = out["draw"]
                    q["a"][held] = a if np.isscalar(a) else a[held]
                    q["b"][held] = b[held]
                    extra["kind"] = kind
                    out = out["ratio"]
                pred[held] = (out * e)[held]
            preds.append(pred)
            if r == 0 and "kind" in extra:
                extra.update({c: q[c] for c in q})
        np.savez_compressed(OUT / f"{slug(name)}.npz", pred=np.mean(preds, axis=0), **{k: v for k, v in extra.items() if k != "kind"},
                            kind=extra.get("kind", ""))
        if WEIGHTS_LOG:
            pd.DataFrame(WEIGHTS_LOG).to_csv(HERE / "results" / "round8_stack_weights.csv", index=False)
        print(f"{name}: {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- score stage
def gini(actual, pred):
    def raw(a, p):
        a = np.asarray(a, float)[np.lexsort((np.arange(len(p)), -np.asarray(p, float)))]
        return (a.cumsum().sum() / a.sum() - (len(a) + 1) / 2) / len(a)
    return raw(actual, pred) / raw(actual, actual)


def pr_auc(y, score, w):
    """Average precision with row weights (bootstrap counts)."""
    keep = w > 0
    y, score, w = y[keep], score[keep], w[keep]
    o = np.argsort(-score, kind="mergesort")
    y, w = y[o], w[o]
    tp = np.cumsum(w * y); fp = np.cumsum(w * (1 - y))
    prec = tp / (tp + fp)
    return float(np.sum(prec * w * y) / np.sum(w * y))


def metrics(c, pred, ws):
    """c: shared arrays; pred: held-out $ per row; ws: weight per storm (1 = point estimate, counts = bootstrap)."""
    w = ws[c["s"]]
    out = {"d2": 1 - np.sum(w * row_dev(c["y"], pred / 1e6)) / np.sum(w * c["dev_null"]),
           "bias": np.sum(w * pred) / np.sum(w * c["rec"])}
    for t, nm in zip(range(3), ("bias_ts", "bias_c12", "bias_c3")):
        m = c["tier"] == t
        out[nm] = np.sum((w * pred)[m]) / np.sum((w * c["rec"])[m])
    pe = np.bincount(c["c"], weights=w * pred, minlength=c["nc"]) / YEARS
    re = np.bincount(c["c"], weights=w * c["rec"], minlength=c["nc"]) / YEARS
    out["mae_eal"] = np.mean(np.abs(pe - re)) / 1e6
    out["mae_eal_big"] = np.mean(np.abs(pe - re)[c["big"]]) / 1e6
    out["gini"] = gini(re, pe)
    out["pr_auc"] = pr_auc(c["any"], pred / c["e"], w)
    return out


def coverage(df, z, rng, n_sim=300):
    """Share of damaged rows inside the 10-90% band, and of storm totals inside a simulated 10-90% band (repeat 0)."""
    y, e = df.prop_usd.to_numpy(float), df.bldg_value.to_numpy(float)
    pos = y > 0
    r = y[pos] / e[pos]
    row_cov = np.mean((r >= z["q10"][pos]) & (r <= z["q90"][pos]))
    kind, a, b, p = str(z["kind"]), z["a"], z["b"], z["chance"]
    sid = df.storm_id.to_numpy()
    hits = []
    for s in np.unique(sid):
        m = sid == s
        hit = rng.random((n_sim, m.sum())) < p[m]
        if kind == "gamma":
            size = rng.gamma(a[m], b[m] / a[m], (n_sim, m.sum()))
        else:
            size = np.exp(rng.normal(a[m], b[m], (n_sim, m.sum())))
        tot = (hit * np.minimum(size, 1.0) * e[m]).sum(1)
        lo, hi = np.quantile(tot, [.1, .9])
        hits.append(lo <= y[m].sum() <= hi)
    return row_cov, float(np.mean(hits))


def run_score(budget=1e9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = prep()
    rec, e = df.prop_usd.to_numpy(float), df.bldg_value.to_numpy(float)
    s_codes, storms = pd.factorize(df.storm_id)
    c_codes, counties = pd.factorize(df.fips)
    # null model: one national loss ratio from the training storms (averaged over repeats, as in compare.py)
    null = np.zeros(len(df))
    for r in range(3):
        fold = df[f"fold_all_r{r}"].to_numpy()
        for k in range(5):
            held, fit = fold == k, fold != k
            null[held] += rec[fit].sum() / e[fit].sum() * e[held] / 3
    rec_eal = np.bincount(c_codes, weights=rec, minlength=len(counties)) / YEARS
    c = {"s": s_codes, "c": c_codes, "nc": len(counties), "rec": rec, "y": rec / 1e6, "e": e, "any": (rec > 0).astype(float),
         "dev_null": row_dev(rec / 1e6, null / 1e6), "tier": df.storm_strength.map({t: i for i, t in enumerate(TIERS)}).to_numpy(),
         "big": rec_eal >= 1e6}
    preds = {n: np.load(OUT / f"{slug(n)}.npz") for n in MODELS if (OUT / f"{slug(n)}.npz").exists()}
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, len(storms), len(storms)), minlength=len(storms)).astype(float) for _ in range(N_BOOT)]
    ones = np.ones(len(storms))
    boot, t0 = {}, time.time()
    for n, z in preds.items():                    # cached per model, so long runs can be split across calls
        f = OUT / f"boot_{slug(n)}_{N_BOOT}.csv"
        if f.exists():
            boot[n] = pd.read_csv(f)
            continue
        if time.time() - t0 > budget:
            print(f"time budget used; run again to continue (next: {n})"); return
        boot[n] = pd.DataFrame([metrics(c, z["pred"], w) for w in draws])
        boot[n].to_csv(f, index=False)
        print(f"  bootstrapped {n}", flush=True)
    keys = ["d2", "mae_eal", "gini", "pr_auc"]
    better = {"d2": 1, "mae_eal": -1, "gini": 1, "pr_auc": 1}
    rows = []
    for n, z in preds.items():
        pt, b, ref = metrics(c, z["pred"], ones), boot[n], boot[REFERENCE]
        row = {"family": slug(n)[0], "model": n}
        for k in ["d2", "mae_eal", "mae_eal_big", "gini", "pr_auc", "bias", "bias_ts", "bias_c12", "bias_c3"]:
            row[k] = pt[k]
            row[f"{k}_95ci"] = f"{b[k].quantile(.025):.3f} to {b[k].quantile(.975):.3f}"
        wins = 0
        for k in keys:
            diff = b[k] - ref[k]
            lo, hi = diff.quantile(.025), diff.quantile(.975)
            row[f"{k}_vs_ref_95ci"] = f"{lo:+.3f} to {hi:+.3f}"
            sig = (lo > 0 or hi < 0)
            row[f"{k}_vs_ref"] = ("better" if np.sign(diff.median()) == better[k] else "worse") if sig else "tie"
        if str(z["kind"]):
            row["row_cov_80"], row["storm_cov_80"] = coverage(df, z, np.random.default_rng(SEED))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(HERE / "results" / "round8_metrics.csv", index=False)

    # chart: paired differences vs the reference, one panel per metric
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.38 * len(out) + 1.6), sharey=True)
    names = out.model.tolist()
    for ax, k, lab in zip(axes, keys, ["Tweedie D² (higher better)", "MAE county EAL, $M/yr (lower better)",
                                       "Normalized Gini (higher better)", "PR-AUC (higher better)"]):
        for i, n in enumerate(names):
            d = boot[n][k] - boot[REFERENCE][k]
            lo, mid, hi = d.quantile([.025, .5, .975])
            colr = "#2a7f62" if ((lo > 0 and better[k] == 1) or (hi < 0 and better[k] == -1)) else \
                   "#b8423a" if ((hi < 0 and better[k] == 1) or (lo > 0 and better[k] == -1)) else "#888888"
            ax.plot([lo, hi], [i, i], color=colr, lw=2); ax.plot(mid, i, "o", color=colr, ms=4)
        ax.axvline(0, color="black", lw=0.8); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=8)
        ax.set_xlabel("difference vs current model (95% interval)", fontsize=8)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(names, fontsize=8); axes[0].invert_yaxis()
    fig.suptitle("Round 8: each model vs the current model (K12), all 194 storms, storm bootstrap", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "results" / "round8_summary.png", dpi=150)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 52)
    show = ["model", "d2", "mae_eal", "gini", "pr_auc", "bias", "bias_ts", "bias_c12", "bias_c3"] + [f"{k}_vs_ref" for k in keys]
    print(out[show].round(3).to_string(index=False))
    if "row_cov_80" in out:
        print(out.dropna(subset=["row_cov_80"])[["model", "row_cov_80", "storm_cov_80"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        run_fit(sys.argv[2:] or None)
    else:
        run_score(float(sys.argv[2]) if len(sys.argv) > 2 else 1e9)
