"""Sandbox library: model pieces, the experiment runner and the metrics. Plain numpy / pandas.

A model is described by a small dict ("spec"), e.g.
    {"features": BASE, "alpha": 1.0}                       two-part GLM
    {"features": BASE, "by_type": True, "rescale": "storm_strength"}
    {"kind": "table", "keys": ["wind_bin", "coast"]}       damage table
predict(df, fit, spec) fits on the rows in `fit` and returns a property damage RATIO for every row.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIERS = ["TD / TS", "Cat 1-2", "Cat 3+"]
BASE = ["wind_ms", "wind_excess", "log_wind_exposure", "log_dist_km", "coastal", "coastal_x_wind", "log_density"]
STORM = ["storm_peak_wind", "log_footprint"]
TRACK = ["right_of_track", "log_fwd_speed", "dvmax_24h", "coastal_x_right"]
EDGES = [0, 10, 17.5, 25, 33, 43, 50, 999]          # county wind bins, m/s
REST = ["log_wind_exposure", "log_dist_km", "coastal", "log_density"]
WIND_BINS = [f"wb{b}" for b in range(1, 7)]
J2 = ["wind_cap", "coastal_x_wind_cap", "log_dist_km", "coastal"]        # the chosen 4-input model (CHOICE.md)


def capped(cap=50):
    """Feature list with the wind response held flat above `cap` m/s."""
    tag = "" if cap == 50 else str(cap)
    return [f"wind_cap{tag}", f"wind_excess_cap{tag}", f"coastal_x_wind_cap{tag}"] + REST
POWER = 1.5                                         # Tweedie power for the deviance score


def load():
    df = pd.read_csv(HERE / "data" / "table.csv", dtype={"fips": str})
    df["wind_bin"] = pd.cut(df.wind_ms, EDGES, labels=False, right=False)
    df["coast"] = df.coastal.astype(int)
    # alternative shapes for the wind response (see experiments round 2)
    df["log_wind"] = np.log(df.wind_ms)                                   # power law: damage ~ wind^k
    df["log_wind_sq"] = df.log_wind ** 2
    for cap in (40, 45, 50, 55, 60):                                      # linear, but flat above the cap
        tag = "" if cap == 50 else str(cap)
        df[f"wind_cap{tag}"] = df.wind_ms.clip(upper=cap)
        df[f"wind_excess_cap{tag}"] = df.wind_excess.clip(upper=cap - 17.5)
        df[f"coastal_x_wind_cap{tag}"] = df.coastal * df[f"wind_excess_cap{tag}"]
    vn = np.maximum(df.wind_ms - 25.7, 0) / (74.7 - 25.7)                 # Emanuel (2011) damage curve
    df["log_emanuel"] = np.log(vn ** 3 / (1 + vn ** 3) + 1e-4)
    for b in range(1, len(EDGES) - 1):                                    # one yes/no column per wind bin
        df[f"wb{b}"] = (df.wind_bin == b).astype(float)
    df["coastal_x_ts"] = df.coastal * (df.wind_ms >= 17.5)                # on the coast AND tropical-storm wind
    add_round6_features(df)
    df["coastal_x_logwind"] = df.coastal * np.maximum(0, df.log_wind - np.log(17.5))
    return df


# ---------------------------------------------------------------- round 6 features
REGIONS = {"reg_tx_la": ["TX", "LA"], "reg_ms_al": ["MS", "AL"], "reg_fl": ["FL"], "reg_se_atl": ["GA", "SC", "NC"],
           "reg_mid_atl": ["VA", "MD", "DE", "DC", "NJ", "PA", "NY"], "reg_new_eng": ["CT", "RI", "MA", "NH", "VT", "ME"]}


def kmeans(X, k, seed=42, n_init=10, iters=100):
    """Plain k-means (several random starts, keep the tightest)."""
    rng, best = np.random.default_rng(seed), None
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def add_round6_features(df):
    root = HERE.parent
    co = pd.read_csv(root / "final" / "counties.csv", dtype={"fips": str}, usecols=["fips", "pop_lat", "pop_lon"])
    sc = pd.read_csv(root / "final" / "storm_county.csv", dtype={"fips": str}, usecols=["storm_id", "fips", "vmax_at_min_ms"])
    extra = df[["storm_id", "fips"]].merge(co, on="fips", how="left").merge(sc, on=["storm_id", "fips"], how="left")
    df["pop_lat"], df["vmax_at_min_ms"] = extra.pop_lat.to_numpy(), extra.vmax_at_min_ms.to_numpy()

    # (a) asymmetric wind: a moving storm's forward speed adds to the wind on its right side and subtracts on
    #     its left (standard approximation: about half the forward speed, fading with the wind field)
    fwd_ms = df.fwd_speed_kmh / 3.6
    fade = (df.wind_ms / df.vmax_at_min_ms).clip(0, 1)
    df["wind_asym"] = (df.wind_ms + 0.5 * fwd_ms * df.right_of_track * fade).clip(lower=0)
    df["wa_cap"] = df.wind_asym.clip(upper=50)
    df["coastal_x_wa_cap"] = df.coastal * (df.wind_asym - 17.5).clip(0, 32.5)

    # (b) region of the country (building practice and exposure differ by coast); shrunk by the ridge penalty
    for col, states in REGIONS.items():
        df[col] = df.state.isin(states).astype(float)

    # (c) county clusters (professor's idea): k-means on county traits, not on location or damage
    c = df.groupby("fips").agg(dens=("log_density", "mean"), val=("bldg_value", "mean"), people=("population", "mean"),
                               farm=("farm_timber_value", "mean"), coast=("coastal", "first"))
    X = np.column_stack([c.dens, np.log(c.val / c.people), np.log1p(1000 * c.farm / c.val), c.coast])
    lab = kmeans((X - X.mean(0)) / X.std(0), 6)
    lab = pd.Series(lab, index=c.index)
    for j in range(1, 6):
        df[f"cl{j}"] = (df.fips.map(lab) == j).astype(float)

    # (d) reporting era (labels get less complete over time)
    df["era_2010s"] = df.year.between(2010, 2019).astype(float)
    df["era_2020s"] = (df.year >= 2020).astype(float)


# ---------------------------------------------------------------- fitting
def fit_logistic(X1, y, pen, iters=50):
    """Logistic regression (Newton steps). X1 has the intercept column; pen = ridge penalty per column."""
    b, P = np.zeros(X1.shape[1]), np.diag(pen)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
        step = np.linalg.solve((X1 * (p * (1 - p))[:, None]).T @ X1 + P, X1.T @ (y - p) - P @ b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def fit_gamma(X1, y, w, pen, iters=200):
    """Gamma regression, log link, by repeated weighted least squares. w = row weights."""
    w, P = w / w.mean(), np.diag(pen)
    b = np.zeros(X1.shape[1]); b[0] = np.log(np.sum(w * y) / np.sum(w))

    def loss(b):
        mu = np.exp(np.clip(X1 @ b, -40, 10))
        return 2 * np.sum(w * ((y - mu) / mu - np.log(y / mu))) + b @ P @ b

    current = loss(b)
    for _ in range(iters):
        eta = np.clip(X1 @ b, -40, 10)
        z = eta + (y - np.exp(eta)) / np.exp(eta)
        step = np.linalg.solve((X1 * w[:, None]).T @ X1 + P, X1.T @ (w * z)) - b
        for _ in range(30):
            new = loss(b + step)
            if new <= current:
                break
            step /= 2
        b, previous, current = b + step, current, new
        if abs(previous - current) < 1e-9 * abs(previous):
            break
    return b


def fit_tweedie(X1, y, w, pen, power=POWER, iters=100):
    """One-equation Tweedie GLM (compound Poisson-gamma), log link, by IRLS. The standard insurance
    model for losses that are mostly zero with a long right tail."""
    w, P = w / w.mean(), np.diag(pen)
    b = np.zeros(X1.shape[1]); b[0] = np.log(np.sum(w * y) / np.sum(w))

    def loss(b):
        mu = np.exp(np.clip(X1 @ b, -50, 5))
        return np.sum(w * (-y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power))) + b @ P @ b / 2

    current = loss(b)
    for _ in range(iters):
        eta = np.clip(X1 @ b, -50, 5); mu = np.exp(eta)
        W = w * mu ** (2 - power)
        z = eta + (y - mu) / mu
        step = np.linalg.solve((X1 * W[:, None]).T @ X1 + P, (X1 * W[:, None]).T @ z) - b
        for _ in range(30):
            new = loss(b + step)
            if new <= current:
                break
            step /= 2
        b, previous, current = b + step, current, new
        if abs(previous - current) < 1e-10 * abs(previous):
            break
    return b


# ---------------------------------------------------------------- models
def two_part(df, fit, target, features=BASE, alpha=1.0, rows=None, suspect_wind=None, storm_penalty=None,
             exposure="bldg_value", cap=1.0, details=False, zero_features=None, exclude=None):
    """chance of any damage (logistic) x size if damaged (gamma on damage / exposure).

    alpha          weight of a damaged row in the size model = exposure / (its storm's exposure)^alpha
                   (1 = every storm counts equally, 0 = rows count by dollars at risk)
    rows           optional mask: the model only applies to these rows (others predict 0)
    suspect_wind   leave rows with wind >= this and NO damage record out of the chance model
                   (treat them as "unknown" rather than "$0")
    storm_penalty  add one adjustment per storm to the size model, shrunk toward 0 by this penalty
                   (a simple random effect); new storms get the average adjustment
    zero_features  features set to 0 when PREDICTING (fitted as usual), e.g. predict every year at the
                   2000s reporting level
    exclude        optional mask of rows left out of FITTING (still predicted), e.g. suspicious zeros
    """
    e = df[exposure].to_numpy(float)
    usable = e > 0 if rows is None else (e > 0) & rows
    ratio = np.where(usable, df[target].to_numpy(float) / np.where(e > 0, e, 1), 0.0)
    fit = fit & usable
    if exclude is not None:
        fit = fit & ~exclude
    X = df[features].to_numpy(float)
    mean, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    X1 = np.column_stack([np.ones(len(X)), (X - mean) / np.where(sd > 0, sd, 1)])
    Xp = X.copy()
    for f in (zero_features or []):
        Xp[:, list(features).index(f)] = 0.0
    X1p = np.column_stack([np.ones(len(X)), (Xp - mean) / np.where(sd > 0, sd, 1)])
    pen = np.r_[0.0, np.ones(len(features))]

    fit_chance = fit if suspect_wind is None else fit & ~((df.wind_ms.to_numpy() >= suspect_wind) & (ratio == 0))
    b_chance = fit_logistic(X1[fit_chance], (ratio[fit_chance] > 0).astype(float), pen)

    pos = fit & (ratio > 0)
    storm = df.storm_id.to_numpy()[pos]
    w = e[pos] / pd.Series(e[pos]).groupby(storm).transform("sum").to_numpy() ** alpha
    if storm_penalty is None:
        b_size, smear = fit_gamma(X1[pos], ratio[pos], w, pen), 1.0
    else:
        codes, names = pd.factorize(storm)
        D = np.zeros((len(codes), len(names))); D[np.arange(len(codes)), codes] = 1.0
        b_all = fit_gamma(np.column_stack([X1[pos], D]), ratio[pos], w, np.r_[pen, np.full(len(names), storm_penalty)])
        b_size, smear = b_all[:X1.shape[1]], float(np.mean(np.exp(b_all[X1.shape[1]:])))

    chance = 1 / (1 + np.exp(-np.clip(X1p @ b_chance, -30, 30)))
    pred = np.where(usable, np.minimum(chance * np.exp(np.clip(X1p @ b_size, -40, 10)) * smear, cap), 0.0)
    if details:
        return pred, pd.DataFrame({"feature": ["intercept"] + list(features), "chance": b_chance, "size": b_size})
    return pred


def table(df, fit, target, keys, exposure="bldg_value", min_storms=3):
    """Damage table: loss ratio per box = total damage / total exposure in the box (fit rows).
    A box with fewer than min_storms storms falls back to the same box without its last key."""
    d = df.loc[fit & (df[exposure] > 0).to_numpy()]
    out = pd.Series(np.nan, index=df.index)
    for n in range(len(keys), 0, -1):
        g = d.groupby(keys[:n])
        ratio = (g[target].sum() / g[exposure].sum()).where(g.storm_id.nunique() >= (min_storms if n > 1 else 1))
        look = df[keys[:n]].merge(ratio.rename("r").reset_index(), on=keys[:n], how="left").r.to_numpy()
        out = out.fillna(pd.Series(look, index=df.index))
    return np.where(df[exposure] > 0, out.fillna(0.0).to_numpy(), 0.0)


def tweedie_glm(df, fit, target, features, alpha=1.0, exposure="bldg_value", cap=1.0):
    """One-equation Tweedie GLM on the damage ratio. Row weight = exposure / (storm's exposure)^alpha."""
    e = df[exposure].to_numpy(float)
    usable = e > 0
    ratio = np.where(usable, df[target].to_numpy(float) / np.where(usable, e, 1), 0.0)
    fit = fit & usable
    X = df[features].to_numpy(float)
    mean, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    X1 = np.column_stack([np.ones(len(X)), (X - mean) / np.where(sd > 0, sd, 1)])
    w = e[fit] / pd.Series(e[fit]).groupby(df.storm_id.to_numpy()[fit]).transform("sum").to_numpy() ** alpha
    b = fit_tweedie(X1[fit], ratio[fit], w, np.r_[0.0, np.ones(len(features))])
    return np.where(usable, np.minimum(np.exp(np.clip(X1 @ b, -50, 5)), cap), 0.0)


def emanuel_fit(df, fit, target="prop_usd", exposure="bldg_value", floor=False):
    """Emanuel (2011) damage curve with its two wind thresholds FITTED (CLIMADA-style calibration),
    separately for coastal and inland counties. ratio = scale x vn^3 / (1 + vn^3),
    vn = max(0, wind - v_thresh) / (v_half - v_thresh). Thresholds picked on a small grid by Tweedie
    deviance on the fit rows; scale set so fit-row dollars match.
    floor=True adds a small constant loss ratio (the fit rows' average ratio below the threshold), because
    the pure curve predicts exactly $0 below its threshold although some of those counties do record damage."""
    e, y, w = df[exposure].to_numpy(float), df[target].to_numpy(float), df.wind_ms.to_numpy(float)
    out = np.zeros(len(df))
    for c in (0, 1):
        grp = (df.coastal.to_numpy() == c) & (e > 0)
        f = fit & grp
        best = None
        for vt in (15.0, 17.5, 20.0, 25.7):
            for vh in (40.0, 50.0, 60.0, 74.7, 90.0, 120.0):
                vn = np.maximum(w - vt, 0) / (vh - vt)
                curve = vn ** 3 / (1 + vn ** 3)
                below = f & (w <= vt)
                base = y[below].sum() / max(e[below].sum(), 1e-12) if floor else 0.0
                scale = max(y[f].sum() - base * e[f].sum(), 0) / max((curve * e)[f].sum(), 1e-12)
                mu = (scale * curve + base) * e
                dev = 2 * np.sum(y[f] ** (2 - POWER) / ((1 - POWER) * (2 - POWER)) - y[f] * np.maximum(mu[f], 1e-12) ** (1 - POWER) / (1 - POWER)
                                 + np.maximum(mu[f], 1e-12) ** (2 - POWER) / (2 - POWER))
                if best is None or dev < best[0]:
                    best = (dev, scale * curve + base, vt, vh)
        out[grp] = best[1][grp]
    return np.minimum(out, 1.0)


def predict(df, fit, spec):
    """Predicted property damage ratio for every row, from a model fitted on the `fit` rows."""
    if spec.get("kind") == "average":                # simple model averaging (forecast combination)
        return np.mean([predict(df, fit, m) for m in spec["members"]], axis=0)
    if spec.get("kind") == "table":
        ratio = table(df, fit, "prop_usd", spec["keys"])
    elif spec.get("kind") == "tweedie":
        ratio = tweedie_glm(df, fit, "prop_usd", spec["features"], alpha=spec.get("alpha", 1.0))
    elif spec.get("kind") == "emanuel":
        ratio = emanuel_fit(df, fit, floor=spec.get("floor", False))
    else:
        kw = {k: spec[k] for k in ("features", "alpha", "suspect_wind", "storm_penalty", "zero_features") if k in spec}
        if spec.get("exclude_col"):
            kw["exclude"] = df[spec["exclude_col"]].to_numpy(bool)
        sfx = "_imp" if spec.get("impute") else ""   # train on KNN-imputed labels (round 7)
        if spec.get("by_type"):                     # wind + surge (coastal counties only) + rain
            ratio = (two_part(df, fit, "prop_wind_usd" + sfx, **kw)
                     + two_part(df, fit, "prop_surge_usd", rows=(df.coastal == 1).to_numpy(), **kw)
                     + two_part(df, fit, "prop_rain_usd", **kw))
        else:
            ratio = two_part(df, fit, "prop_usd" + sfx, **kw)
    if spec.get("rescale"):                         # one correction factor per group, learned on the fit rows
        e, y = df.bldg_value.to_numpy(), df.prop_usd.to_numpy()
        shrink = spec.get("rescale_shrink", 0.0)    # 0 = full correction, 1 = none
        for value in df[spec["rescale"]].unique():
            m = (df[spec["rescale"]] == value).to_numpy()
            predicted = (ratio * e)[fit & m].sum()
            if predicted > 0 and y[fit & m].sum() > 0:
                ratio = np.where(m, ratio * (y[fit & m].sum() / predicted) ** (1 - shrink), ratio)
    return np.minimum(ratio, 1.0)


# ---------------------------------------------------------------- metrics
def roc_auc(y, score):
    y = np.asarray(y, bool)
    ranks = pd.Series(np.asarray(score, float)).rank().to_numpy()
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def spearman(a, b):
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))


def tweedie_deviance(y, mu):
    mu = np.maximum(mu, 1e-12)
    return 2 * np.sum(y ** (2 - POWER) / ((1 - POWER) * (2 - POWER)) - y * mu ** (1 - POWER) / (1 - POWER)
                      + mu ** (2 - POWER) / (2 - POWER))


def evaluate(d):
    """d: storm x county rows with storm_id, storm_strength, fips, pred, rec, null (all in $).

    d2            share of the Tweedie deviance explained vs. `null` (one national loss ratio x exposure)
    pred_rec      total predicted / total recorded; tier_* = the same per storm tier
    tier_miss     typical miss across the three tiers, as a factor (1.0 = all three tiers right)
    storm_miss    the typical storm's predicted vs recorded, as a factor (storms with damage)
    rho_storm     rank correlation of storm totals;  roc_auc = any damage, row level
    top10         share of recorded $ in the model's top 10% of counties;  within3x = counties within 3x
    """
    y, mu, null = d.rec.to_numpy() / 1e6, d.pred.to_numpy() / 1e6, d.null.to_numpy() / 1e6
    st = d.groupby("storm_id").agg(tier=("storm_strength", "first"), pred=("pred", "sum"), rec=("rec", "sum"))
    hit = st[st.rec > 0]
    tier = st.groupby("tier")[["pred", "rec"]].sum()
    tier_ratio = (tier.pred / tier.rec).reindex(TIERS)
    co = d.groupby("fips")[["pred", "rec"]].sum()
    big = co[co.rec >= 250_000]
    r = big.pred.clip(lower=1) / big.rec
    top = co.pred.nlargest(max(1, len(co) // 10)).index
    return {"d2": 1 - tweedie_deviance(y, mu) / tweedie_deviance(y, null),
            "pred_rec": st.pred.sum() / st.rec.sum(),
            "tier_ts": tier_ratio.iloc[0], "tier_c12": tier_ratio.iloc[1], "tier_c3": tier_ratio.iloc[2],
            "tier_miss": 10 ** np.mean(np.abs(np.log10(tier_ratio))),
            "storm_miss": 10 ** np.median(np.abs(np.log10((hit.pred + 1) / hit.rec))),
            "rho_storm": spearman(st.pred, st.rec), "roc_auc": roc_auc(d.rec > 0, d.pred / d.bldg_value),
            "top10": co.rec[top].sum() / co.rec.sum(), "within3x": ((r >= 1 / 3) & (r <= 3)).mean()}


def oof(df, spec, repeat, scope="dev", eval_col="prop_usd"):
    """Held-out predictions ($) under one 5-fold split; also the null model's.
    scope "dev" = the 134 dev storms (fold_r*), "all" = all 194 storms (fold_all_r*)."""
    dev = (df.set == "dev").to_numpy() if scope == "dev" else np.ones(len(df), bool)
    col = f"fold_r{repeat}" if scope == "dev" else f"fold_all_r{repeat}"
    fold, e, y = df[col].to_numpy(), df.bldg_value.to_numpy(), df[eval_col].to_numpy()
    pred, null = np.zeros(len(df)), np.zeros(len(df))
    for k in range(5):
        held, fit = dev & (fold == k), dev & (fold != k)
        pred[held] = (predict(df, fit, spec) * e)[held]
        null[held] = y[fit].sum() / e[fit].sum() * e[held]
    return df.loc[dev, ["storm_id", "storm_strength", "fips", "bldg_value"]].assign(pred=pred[dev], rec=y[dev], null=null[dev])


def cross_validate(df, spec, repeats=(0, 1, 2), scope="dev", eval_col="prop_usd"):
    """Average the metrics over repeated 5-fold splits (storm-held-out)."""
    runs = pd.DataFrame([evaluate(oof(df, spec, r, scope, eval_col)) for r in repeats])
    return runs.mean().to_dict(), runs
