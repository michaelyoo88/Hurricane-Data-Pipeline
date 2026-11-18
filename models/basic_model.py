"""Basic model: expected annual hurricane loss (EAL) per county for property, crop and people.

Same two-part model for each category, on storm x county rows (2000-2024):
    chance of any damage   logistic regression on all rows
    size if damaged        gamma regression on the damage RATIO (damage / exposure), damaged rows only,
                           weighted so every storm counts equally (within a storm, bigger counties count more)
    predicted $ = chance x size x exposure     (ratio capped: can't lose more than 100% of buildings / people)

Honest test: 5 folds that hold out WHOLE storms. Every storm is predicted by a model that never saw it
("out-of-fold"). All metrics and the EAL below use only those predictions.

EAL per county = sum of predicted $ over all storms 2000-2024, priced on 2024 exposure, / 25 seasons.

Out (models/results/): cv_metrics.csv, cv_by_strength.csv, coefficients.csv, county_eal.csv,
eal_summary.csv, storm_pred_vs_recorded.png, calibration_by_strength.png, eal_vs_fema.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_table import FEATURES, VSL_2023
from glm import add_intercept, fit_gamma, fit_logistic, pr_auc, spearman

SEED, N_FOLDS, N_BOOT, N_SEASONS = 42, 5, 500, 25
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BLUE = "#2b6cb0"

# target, exposure that year, exposure in 2024, $ per unit of target, cap on the predicted ratio
CATEGORIES = {
    "property": ("prop_usd", "bldg_value", "bldg_value_2024", 1.0, 1.0),
    "crop": ("crop_usd", "farm_timber_value", "farm_timber_value_2024", 1.0, None),   # crop ratio can pass 1:
    "people": ("people_equiv", "population", "population_2024", VSL_2023, 1.0),       # sales are a yearly flow,
}                                                                                     # storms destroy standing timber

df = pd.read_csv(HERE / "data" / "model_table.csv", dtype={"fips": str})
rng = np.random.default_rng(SEED)
storms = np.sort(df.storm_id.unique())
fold = df.storm_id.map(dict(zip(rng.permutation(storms), np.arange(len(storms)) % N_FOLDS))).to_numpy()


def fit_predict(cat, fit):
    """Fit the two-part model on `fit` rows; return predicted ratio for every row + coefficients."""
    target, exposure, _, _, cap = CATEGORIES[cat]
    has_exposure = (df[exposure] > 0).to_numpy()                 # e.g. counties with no farms: predict 0
    ratio = np.where(has_exposure, df[target] / df[exposure].where(has_exposure, 1), 0.0)
    fit = fit & has_exposure
    damaged = ratio > 0

    X = df[FEATURES].to_numpy(dtype=float)
    mean, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    X = (X - mean) / np.where(sd > 0, sd, 1)                     # scale with the fit rows only

    b_chance = fit_logistic(X[fit], damaged[fit].astype(float))
    pos = fit & damaged
    share = df.loc[pos, exposure] / df.loc[pos].groupby("storm_id")[exposure].transform("sum")
    b_size = fit_gamma(X[pos], ratio[pos], share.to_numpy())     # each storm's weights add up to 1

    X1 = add_intercept(X)
    chance = 1 / (1 + np.exp(-np.clip(X1 @ b_chance, -30, 30)))
    size = np.exp(np.clip(X1 @ b_size, -40, 10))
    pred = chance * size
    pred = np.minimum(pred, cap if cap else ratio[pos].max())
    return np.where(has_exposure, pred, 0.0), chance, b_chance, b_size


# ---- 1. Out-of-fold predictions for every category ----
coefs = []
for cat, (target, exposure, exposure_now, usd, _) in CATEGORIES.items():
    ratio, chance = np.zeros(len(df)), np.zeros(len(df))
    for k in range(N_FOLDS):
        held = fold == k
        r, c, _, _ = fit_predict(cat, ~held)
        ratio[held], chance[held] = r[held], c[held]
    assert np.isfinite(ratio).all() and (ratio >= 0).all()
    df[f"{cat}_score"] = ratio                                   # for ranking "any damage"
    df[f"{cat}_pred"] = ratio * df[exposure] * usd               # $ on that year's exposure (compare with recorded)
    df[f"{cat}_pred_now"] = ratio * df[exposure_now] * usd       # $ on 2024 exposure (for EAL)
    df[f"{cat}_rec"] = df[target] * usd
    _, _, b_chance, b_size = fit_predict(cat, np.ones(len(df), bool))     # all storms, only to read the coefficients
    coefs.append(pd.DataFrame({"category": cat, "feature": ["intercept"] + FEATURES,
                               "chance_of_damage": b_chance, "size_if_damaged": b_size}))
df["total_pred"] = sum(df[f"{c}_pred"] for c in CATEGORIES)
df["total_pred_now"] = sum(df[f"{c}_pred_now"] for c in CATEGORIES)
df["total_rec"] = sum(df[f"{c}_rec"] for c in CATEGORIES)
df["total_score"] = df.total_pred
pd.concat(coefs).round(3).to_csv(OUT / "coefficients.csv", index=False)
# Save the held-out predictions (and the folds) so other models can be compared on the same storms
keep = ["storm_id", "fips"] + [f"{c}_{k}" for c in CATEGORIES for k in ["score", "pred", "pred_now"]]
df[keep].assign(fold=fold).to_csv(HERE / "data" / "oof_basic.csv", index=False)
ALL = list(CATEGORIES) + ["total"]


# ---- 2. Metrics (storm-held-out) ----
def storm_table(cat):
    return df.groupby("storm_id").agg(strength=("storm_strength", "first"), name=("name", "first"),
                                      year=("year", "first"), pred=(f"{cat}_pred", "sum"), rec=(f"{cat}_rec", "sum"))


def storm_metrics(st):
    hit = st[st.rec > 0]
    return {"pred_over_recorded": st.pred.sum() / st.rec.sum(),
            "typical_storm_factor": 10 ** np.median(np.log10((hit.pred + 1) / hit.rec)),
            "rank_corr_storms": spearman(st.pred, st.rec)}


rows, strength_rows = [], []
for cat in ALL:
    st = storm_table(cat)
    m = storm_metrics(st)
    boot = pd.DataFrame([storm_metrics(st.sample(len(st), replace=True, random_state=int(s)))
                         for s in rng.integers(0, 1e9, N_BOOT)])
    cy = df.groupby(["fips", "year"])[[f"{cat}_pred", f"{cat}_rec"]].sum()
    top3 = st.rec.nlargest(3)
    row = {"category": cat, "recorded_B": st.rec.sum() / 1e9, "predicted_B": st.pred.sum() / 1e9,
           "pr_auc_any_damage": pr_auc(df[f"{cat}_rec"] > 0, df[f"{cat}_score"]),
           "base_rate": (df[f"{cat}_rec"] > 0).mean(),
           "rank_corr_county_year": spearman(cy[f"{cat}_pred"], cy[f"{cat}_rec"])}
    for k, v in m.items():
        lo, hi = boot[k].quantile([0.025, 0.975])
        row[k], row[k + "_95ci"] = v, f"{lo:.2f} to {hi:.2f}"
    row["pred_over_recorded_without_top3"] = st.drop(top3.index).pred.sum() / st.drop(top3.index).rec.sum()
    row["top3_storms"] = ", ".join(f"{st.name[s].title()} {st.year[s]}" for s in top3.index)
    row["top3_share_of_recorded"] = top3.sum() / st.rec.sum()
    rows.append(row)
    for strength, g in st.groupby("strength"):
        hit = g[g.rec > 0]
        strength_rows.append({"category": cat, "storm_strength": strength, "storms": len(g),
                              "recorded_B": g.rec.sum() / 1e9, "predicted_B": g.pred.sum() / 1e9,
                              "pred_over_recorded": g.pred.sum() / g.rec.sum() if g.rec.sum() else np.nan,
                              "typical_storm_factor": 10 ** np.median(np.log10((hit.pred + 1) / hit.rec)) if len(hit) else np.nan})
metrics = pd.DataFrame(rows)
by_strength = pd.DataFrame(strength_rows)
metrics.round(3).to_csv(OUT / "cv_metrics.csv", index=False)
by_strength.round(3).to_csv(OUT / "cv_by_strength.csv", index=False)

# ---- 3. County EAL (2024 exposure, 25 seasons) vs recorded average vs FEMA ----
eal = df.groupby("fips").agg(**{f"eal_{c}": (f"{c}_pred_now", "sum") for c in ALL},
                             **{f"recorded_avg_{c}": (f"{c}_rec", "sum") for c in ALL}) / N_SEASONS
counties = pd.read_csv(HERE.parent / "final" / "counties.csv", dtype={"fips": str})
fema = counties.set_index("fips")[["state", "county_name", "nri_hrcn_eal_bldg_usd2023", "nri_hrcn_eal_ag_usd2023",
                                   "nri_hrcn_eal_pope_usd2023", "nri_hrcn_eal_total_usd2023"]]
fema.columns = ["state", "county_name", "fema_property", "fema_crop", "fema_people", "fema_total"]
eal = fema.join(eal).fillna(0)                 # counties never within 500 km of a storm: EAL 0
eal.round(0).to_csv(OUT / "county_eal.csv")

year_tot = df.groupby("year")[[f"{c}_pred_now" for c in ALL]].sum().reindex(range(2000, 2025), fill_value=0)
summary = []
for c in ALL:
    draws = [year_tot[f"{c}_pred_now"].sample(N_SEASONS, replace=True, random_state=int(s)).mean()
             for s in rng.integers(0, 1e9, N_BOOT)]
    both = (eal[f"eal_{c}"] > 0) & (eal[f"fema_{c}"] > 0)
    summary.append({"category": c, "our_eal_B_per_yr": eal[f"eal_{c}"].sum() / 1e9,
                    "our_eal_95ci_B": f"{np.quantile(draws, 0.025) / 1e9:.1f} to {np.quantile(draws, 0.975) / 1e9:.1f}",
                    "recorded_avg_B_per_yr": eal[f"recorded_avg_{c}"].sum() / 1e9,
                    "fema_eal_B_per_yr": eal[f"fema_{c}"].sum() / 1e9,
                    "rank_corr_ours_vs_recorded": spearman(eal[f"eal_{c}"], eal[f"recorded_avg_{c}"]),
                    "rank_corr_ours_vs_fema": spearman(eal[f"eal_{c}"], eal[f"fema_{c}"]),
                    "log_corr_ours_vs_fema": np.corrcoef(np.log10(eal.loc[both, f"eal_{c}"]),
                                                         np.log10(eal.loc[both, f"fema_{c}"]))[0, 1],
                    "counties_ours_gt0": int((eal[f"eal_{c}"] > 0).sum()), "counties_fema_gt0": int((eal[f"fema_{c}"] > 0).sum())})
summary = pd.DataFrame(summary)
summary.round(3).to_csv(OUT / "eal_summary.csv", index=False)

# ---- Checks ----
assert len(df) == 97_179 and df.storm_id.nunique() == 194
assert np.isclose(eal.eal_total.sum(), df.total_pred_now.sum() / N_SEASONS)
assert len(eal) == 3_109

# ---- Print ----
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
print(f"storm-held-out test: {len(df):,} rows, 194 storms, {N_FOLDS} folds, {N_BOOT} bootstrap draws\n")
print(metrics[["category", "recorded_B", "predicted_B", "pred_over_recorded", "pred_over_recorded_95ci",
               "pred_over_recorded_without_top3", "typical_storm_factor", "typical_storm_factor_95ci"]].round(2).to_string(index=False))
print()
print(metrics[["category", "base_rate", "pr_auc_any_damage", "rank_corr_storms", "rank_corr_storms_95ci",
               "rank_corr_county_year", "top3_share_of_recorded", "top3_storms"]].round(3).to_string(index=False))
print("\nby storm strength:"); print(by_strength.round(2).to_string(index=False))
print("\ncoefficients (standardized features; + = more damage):"); print(pd.concat(coefs).round(2).to_string(index=False))
print("\nEAL, $B per year (ours = 2024 exposure, 25 seasons; FEMA hurricane EAL is wind only):")
print(summary.round(2).to_string(index=False))
eal["ours_over_fema"] = eal.eal_total / eal.fema_total.where(eal.fema_total > 0)
big = eal[(eal.eal_total > 5e6) | (eal.fema_total > 5e6)]
cols = ["state", "county_name", "eal_total", "recorded_avg_total", "fema_total", "ours_over_fema"]
show = lambda d: d[cols].assign(**{c: (d[c] / 1e6).round(1) for c in cols[2:5]}).round(2).to_string()   # noqa: E731
print("\ntop 10 counties by our EAL ($M/yr):"); print(show(eal.nlargest(10, "eal_total")))
print("\nwe are far ABOVE FEMA (EAL > $5M somewhere):"); print(show(big.nlargest(8, "ours_over_fema")))
print("\nwe are far BELOW FEMA:"); print(show(big.nsmallest(8, "ours_over_fema")))

# ---- Charts ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, cat in zip(axes, CATEGORIES):
    st = storm_table(cat)
    st = st[(st.rec > 0) | (st.pred > 1e5)]
    x, y = st.rec.clip(lower=1e3) / 1e6, st.pred.clip(lower=1e3) / 1e6
    ax.scatter(x, y, color=BLUE, alpha=0.6, s=16)
    lim = [1e-3, max(x.max(), y.max()) * 3]
    ax.plot(lim, lim, color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    for s, r in st.nlargest(4, "rec").iterrows():
        ax.annotate(f"{r['name'].title()} {r.year}", (r.rec / 1e6, max(r.pred, 1e3) / 1e6), fontsize=7,
                    xytext=(3, -8), textcoords="offset points")
    ax.set_title(cat, fontsize=10)
    ax.set_xlabel("Recorded per storm ($M, log; $0 shown at 0.001)")
axes[0].set_ylabel("Predicted per storm ($M, log)")
fig.suptitle("Storm-held-out predictions vs. recorded damage, 2000-2024 (dashed = equal)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "storm_pred_vs_recorded.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
order = ["TD / TS", "Cat 1-2", "Cat 3+"]
for ax, cat in zip(axes, CATEGORIES):
    g = by_strength[by_strength.category == cat].set_index("storm_strength").reindex(order)
    ax.bar(order, g.pred_over_recorded, color=BLUE)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log"); ax.set_title(cat, fontsize=10); ax.set_xlabel("Storm strength (strongest county wind)")
axes[0].set_ylabel("Predicted $ / recorded $ (log; 1 = equal)")
fig.suptitle("Where the basic model over- and under-predicts", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "calibration_by_strength.png", dpi=150)

d = eal[(eal.eal_total > 0) & (eal.fema_total > 0)]
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(d.fema_total / 1e6, d.eal_total / 1e6, color=BLUE, alpha=0.35, s=8)
lim = [1e-5, max(d.fema_total.max(), d.eal_total.max()) / 1e6 * 3]
ax.plot(lim, lim, color="black", linestyle="--", linewidth=0.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("FEMA NRI hurricane EAL ($M/yr, log)"); ax.set_ylabel("Our EAL, all three categories ($M/yr, log)")
ax.set_title("County EAL: basic model vs. FEMA (dashed = equal)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "eal_vs_fema.png", dpi=150)
