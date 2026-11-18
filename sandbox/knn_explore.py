"""Round 7, part 1: county clusters and KNN neighbors. Does "similar counties" carry real information
about missing damage? Nothing here trains a model; it only checks whether the idea is trustworthy.

1. Clusters: k-means on county traits (density, building value per person, farm share, coastal), not on
   location or damage. Pick k by silhouette; profile each cluster, incl. how often strong-wind hits have
   ANY damage record (a recording rate).
2. KNN within a storm: for each county, its k most similar counties hit by the SAME storm (similar wind,
   distance, coast, density, value per person). Stats: share of neighbors with damage, their median loss ratio.
3. Checks
   A  hide each county's own damage: does the neighbor share predict whether it had damage? (ROC-AUC)
   B  hide each damaged county's amount: does the neighbors' median ratio track it? (rank corr, log error)
   C  natural experiment: use the OLD wind-only labels (hurricane + other wind + tornado records). Flag zeros
      that KNN finds suspicious. Did those counties turn out to have damage once surge / flood records were
      added? Compare with similar zeros that were NOT flagged.
Out: results/clusters_profile.csv, results/knn_checks.csv, results/knn_flag_sensitivity.csv, data/knn_stats.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lib import kmeans, load, roc_auc, spearman

HERE = Path(__file__).resolve().parent
K, FEATS = 10, ["wind_ms", "log_dist_km", "coastal", "log_density", "log_value_pc"]


def silhouette(X, lab):
    """Mean silhouette: how much closer each county is to its own cluster than to the next one (-1 to 1)."""
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        a = D[i, own].sum() / max(own.sum() - 1, 1)
        b = min(D[i, lab == j].mean() for j in np.unique(lab) if j != lab[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


def knn_stats(df, label, k=K):
    """For every row: its k most similar rows from the same storm (itself excluded)."""
    X = df[FEATS].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    ratio = (df[label] / df.bldg_value).to_numpy()
    share, med, mean = np.full(len(df), np.nan), np.full(len(df), np.nan), np.full(len(df), np.nan)
    for _, idx in df.groupby("storm_id").indices.items():
        if len(idx) < 3:
            continue
        D = ((X[idx][:, None, :] - X[idx][None]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        nb = np.argsort(D, axis=1)[:, :min(k, len(idx) - 1)]
        r = ratio[idx][nb]
        share[idx] = (r > 0).mean(1)
        mean[idx] = r.mean(1)
        pos = np.where(r > 0, r, np.nan)
        with np.errstate(all="ignore"):
            med[idx] = np.where(np.isfinite(pos).any(1), np.nanmedian(np.where(np.isfinite(pos).any(1)[:, None], pos, 0), axis=1), np.nan)
    return pd.DataFrame({"nb_share": share, "nb_median_pos": med, "nb_mean": mean}, index=df.index)


if __name__ == "__main__":
    df = load()
    df["log_value_pc"] = np.log(df.bldg_value / df.population)
    pd.set_option("display.width", 250)

    # ---- 1. clusters ----
    c = df.groupby("fips").agg(dens=("log_density", "mean"), val=("bldg_value", "mean"), people=("population", "mean"),
                               farm=("farm_timber_value", "mean"), coast=("coastal", "first"))
    Xc = np.column_stack([c.dens, np.log(c.val / c.people), np.log1p(1000 * c.farm / c.val), c.coast])
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    sil = {k: silhouette(Xc, kmeans(Xc, k)) for k in range(2, 9)}
    best_k = max(sil, key=sil.get)
    print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> k =", best_k)
    lab = pd.Series(kmeans(Xc, best_k), index=c.index)
    df["cluster"] = df.fips.map(lab)
    strong = df.wind_ms >= 25
    prof = c.assign(cluster=lab).groupby("cluster").agg(counties=("dens", "size"), density=("dens", lambda s: np.exp(s).median()),
                                                       value_per_person_k=("val", "median"), coastal_share=("coast", "mean"))
    prof["value_per_person_k"] = (c.val / c.people).groupby(lab).median() / 1e3
    prof["farm_share_pct"] = (100 * c.farm / c.val).groupby(lab).median()
    g = df[strong].groupby("cluster")
    prof["strong_wind_hits"] = g.size()
    prof["recording_rate"] = g.prop_usd.apply(lambda s: (s > 0).mean())
    prof["median_pct_lost_when_damaged"] = 100 * g.apply(lambda d: (d.prop_usd / d.bldg_value)[d.prop_usd > 0].median(), include_groups=False)
    prof["property_B"] = df.groupby("cluster").prop_usd.sum() / 1e9
    prof.round(4).to_csv(HERE / "results" / "clusters_profile.csv")
    print("\nclusters (strong wind = 25+ m/s at the county):"); print(prof.round(3).to_string())

    # ---- 2. KNN stats on the current labels and on the old wind-only labels ----
    now = knn_stats(df, "prop_usd")
    old = knn_stats(df, "prop_wind_usd").add_prefix("old_")
    stats = pd.concat([df[["storm_id", "fips"]], now, old], axis=1)
    stats.to_csv(HERE / "data" / "knn_stats.csv", index=False)

    # ---- 3A / 3B: hide what we know ----
    oofj2 = pd.read_csv(HERE / "results" / "j2_oof_all.csv", dtype={"fips": str}) if (HERE / "results" / "j2_oof_all.csv").exists() else None
    has = now.nb_share.notna().to_numpy()
    dmg = (df.prop_usd > 0).to_numpy()
    rows = [{"check": "A any damage: neighbor share", "score": roc_auc(dmg[has], now.nb_share[has]), "metric": "ROC-AUC"}]
    m = dmg & now.nb_median_pos.notna().to_numpy()
    actual = (df.prop_usd / df.bldg_value).to_numpy()[m]
    rows.append({"check": "B amount: neighbor median ratio", "score": spearman(now.nb_median_pos[m], actual), "metric": "rank corr"})
    rows.append({"check": "B amount: typical miss (x)", "score": 10 ** np.median(np.abs(np.log10(now.nb_median_pos[m] / actual))), "metric": "factor"})
    if oofj2 is not None:
        j = df[["storm_id", "fips"]].merge(oofj2, on=["storm_id", "fips"], how="left").pred_ratio.to_numpy()
        rows.append({"check": "A any damage: J2 model (held-out storms)", "score": roc_auc(dmg[has], j[has]), "metric": "ROC-AUC"})
        rows.append({"check": "B amount: J2 model (held-out storms)", "score": spearman(j[m], actual), "metric": "rank corr"})
        both = 0.5 * pd.Series(now.nb_share[has]).rank(pct=True).to_numpy() + 0.5 * pd.Series(j[has]).rank(pct=True).to_numpy()
        rows.append({"check": "A any damage: J2 + neighbors (rank average)", "score": roc_auc(dmg[has], both), "metric": "ROC-AUC"})
    # strong-wind rows only (where a missing record matters)
    sw = has & (df.wind_ms >= 25).to_numpy()
    rows.append({"check": "A any damage, 25+ m/s rows: neighbor share", "score": roc_auc(dmg[sw], now.nb_share[sw]), "metric": "ROC-AUC"})
    if oofj2 is not None:
        rows.append({"check": "A any damage, 25+ m/s rows: J2 model", "score": roc_auc(dmg[sw], j[sw]), "metric": "ROC-AUC"})

    # ---- 3C: natural experiment with the old wind-only labels ----
    found_later = ((df.prop_surge_usd + df.prop_rain_usd) > 0).to_numpy()
    old_zero = (df.prop_wind_usd == 0).to_numpy()
    sens = []
    for wmin in (17.5, 25.0, 33.0):
        for smin in (0.3, 0.5, 0.7):
            pool = old_zero & (df.wind_ms >= wmin).to_numpy() & old.old_nb_share.notna().to_numpy()
            flag = pool & (old.old_nb_share >= smin).to_numpy()
            rate_f, rate_n = found_later[flag].mean() if flag.any() else np.nan, found_later[pool & ~flag].mean()
            sens.append({"min_wind": wmin, "min_neighbor_share": smin, "zeros_in_pool": int(pool.sum()), "flagged": int(flag.sum()),
                         "flagged_found_later": rate_f, "not_flagged_found_later": rate_n, "lift": rate_f / rate_n if rate_n else np.nan,
                         "water_$B_in_flagged": (df.prop_surge_usd + df.prop_rain_usd)[flag].sum() / 1e9})
    sens = pd.DataFrame(sens)
    sens.round(4).to_csv(HERE / "results" / "knn_flag_sensitivity.csv", index=False)
    checks = pd.DataFrame(rows)
    checks.round(3).to_csv(HERE / "results" / "knn_checks.csv", index=False)
    print("\nchecks A and B (hide what we know):"); print(checks.round(3).to_string(index=False))
    print("\ncheck C (old wind-only labels): share of flagged zeros that later showed surge/flood damage")
    print(sens.round(3).to_string(index=False))
