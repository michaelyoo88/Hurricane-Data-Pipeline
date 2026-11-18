"""Shared scoring, so every model is judged the same way on the same held-out storms."""
import numpy as np
import pandas as pd

from glm import pr_auc, spearman

N_BOOT, SEED = 500, 42


def _storm_metrics(st):
    hit = st[st.rec > 0]
    return {"pred_over_recorded": st.pred.sum() / st.rec.sum(),
            "typical_storm_factor": 10 ** np.median(np.log10((hit.pred + 1) / hit.rec)),
            "rank_corr_storms": spearman(st.pred, st.rec)}


def score(d):
    """d: one row per storm x county with storm_id, storm_strength, fips, year,
    pred ($), rec ($ recorded), score (anything that ranks rows by risk).
    Returns (dict of metrics with 95% bootstrap intervals over storms, table by storm strength)."""
    st = d.groupby("storm_id").agg(strength=("storm_strength", "first"), pred=("pred", "sum"), rec=("rec", "sum"))
    rng = np.random.default_rng(SEED)
    boot = pd.DataFrame([_storm_metrics(st.sample(len(st), replace=True, random_state=int(s)))
                         for s in rng.integers(0, 1e9, N_BOOT)])
    cy = d.groupby(["fips", "year"])[["pred", "rec"]].sum()
    top3 = st.rec.nlargest(3).index
    out = {"recorded_B": st.rec.sum() / 1e9, "predicted_B": st.pred.sum() / 1e9,
           "pr_auc_any_damage": pr_auc(d.rec > 0, d.score), "rank_corr_county_year": spearman(cy.pred, cy.rec),
           "pred_over_recorded_without_top3": st.drop(top3).pred.sum() / st.drop(top3).rec.sum()}
    for k, v in _storm_metrics(st).items():
        lo, hi = boot[k].quantile([0.025, 0.975])
        out[k], out[k + "_95ci"] = v, f"{lo:.2f} to {hi:.2f}"
    by = st.groupby("strength")[["pred", "rec"]].apply(lambda g: pd.Series({
        "storms": len(g), "recorded_B": g.rec.sum() / 1e9, "predicted_B": g.pred.sum() / 1e9,
        "pred_over_recorded": g.pred.sum() / g.rec.sum() if g.rec.sum() else np.nan,
        "typical_storm_factor": 10 ** np.median(np.log10((g.pred[g.rec > 0] + 1) / g.rec[g.rec > 0])) if (g.rec > 0).any() else np.nan}))
    return out, by.reset_index()
