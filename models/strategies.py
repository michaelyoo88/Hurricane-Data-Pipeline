"""Test each model idea on its own, with the same held-out storms and the same metrics.

  0  basic two-part         storm x county: chance of damage x size if damaged (every storm equal)
  1  damage table           loss ratio by wind bin x coast, no regression
  2  top-down               predict each STORM's total (two-part on storm-level features), then split it
                            across counties in proportion to the damage table
  3  by damage type         property = wind model + surge model (coastal counties only) + rain model
  4a weights halfway        basic model, big storms count more in the size model (alpha 0.5)
  4b weights by dollars     basic model, rows count by dollars at risk (alpha 0)
  4c tier rescale           basic model x one correction factor per storm-strength tier, learned on fit storms
  6  county-year two-part   one row per county per YEAR: chance of any damage that year (logistic) x
                            size of the year's loss if damaged (gamma). Held out by year.
  5  longer catalog         (EAL only) price all 1975-2025 storms with the damage table -> 51 seasons

Out (models/results/): strategies_storm_metrics.csv, strategies_by_strength.csv,
strategies_county_year.csv, strategies_eal.csv, catalog_eal.csv, strategies_by_strength.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_table import FEATURES, VSL_2023
from core import damage_table, table_lookup, two_part
from evaluate import score
from glm import add_intercept, fit_gamma, fit_logistic, pr_auc, roc_auc, spearman

SEED, N_SEASONS = 42, 25
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BLUE = "#2b6cb0"
# target, exposure that year, exposure in 2024, $ per unit, cap on the ratio
CATS = {"property": ("prop_usd", "bldg_value", "bldg_value_2024", 1.0, 1.0),
        "crop": ("crop_usd", "farm_timber_value", "farm_timber_value_2024", 1.0, None),
        "people": ("people_equiv", "population", "population_2024", VSL_2023, 1.0)}
ALL = list(CATS) + ["total"]
TIERS = ["TD / TS", "Cat 1-2", "Cat 3+"]

df = pd.read_csv(HERE / "data" / "model_table.csv", dtype={"fips": str})
folds = pd.read_csv(HERE / "data" / "oof_basic.csv", dtype={"fips": str}, usecols=["storm_id", "fips", "fold"])
df = df.merge(folds, on=["storm_id", "fips"], validate="one_to_one")     # same storm folds as basic_model.py
fold = df.fold.to_numpy()
N_FOLDS = fold.max() + 1
coastal = (df.coastal == 1).to_numpy()


def held_out(fn):
    """Run fn(fit_mask) -> {category: ratio for all rows} once per fold; keep only held-out rows."""
    out = {c: np.zeros(len(df)) for c in CATS}
    for k in range(N_FOLDS):
        held = fold == k
        r = fn(~held)
        for c in CATS:
            out[c][held] = r[c][held]
    return out


# ---------- strategy definitions ----------
def basic(fit, alpha=1.0):
    return {c: two_part(df, fit, t, e, FEATURES, cap=cap, alpha=alpha) for c, (t, e, _, _, cap) in CATS.items()}


def table(fit):
    return {c: np.where(df[e] > 0, table_lookup(damage_table(df, fit, t, e), df), 0) for c, (t, e, _, _, _) in CATS.items()}


def top_down(fit):
    out = {}
    fit_storms = set(df.storm_id[fit])
    for c, (t, e, _, _, _) in CATS.items():
        # storm-level table: how much exposure the storm put under strong wind, and its strongest wind
        st = pd.DataFrame({"storm_id": df.storm_id, "y": df[t], "e": df[e], "wind": df.wind_ms,
                           "e_ts": df[e] * (df.wind_ms >= 17.5), "e_hu": df[e] * (df.wind_ms >= 33)}) \
            .groupby("storm_id").agg(y=("y", "sum"), e=("e", "sum"), e_ts=("e_ts", "sum"), e_hu=("e_hu", "sum"),
                                     max_wind=("wind", "max"))
        X = np.column_stack([np.log1p(st.e_ts), np.log1p(st.e_hu), st.max_wind])
        f = st.index.isin(fit_storms)
        X = (X - X[f].mean(axis=0)) / np.where(X[f].std(axis=0) > 0, X[f].std(axis=0), 1)
        b_chance = fit_logistic(X[f], (st.y[f] > 0).to_numpy(dtype=float))
        pos = f & (st.y > 0).to_numpy()
        b_size = fit_gamma(X[pos], st.y[pos].to_numpy(), np.ones(pos.sum()))          # every storm equal
        X1 = add_intercept(X)
        total = 1 / (1 + np.exp(-np.clip(X1 @ b_chance, -30, 30))) * np.exp(np.clip(X1 @ b_size, -40, 40))
        st["total"] = np.minimum(total, st.e)                                        # can't exceed what's in the footprint
        # split the storm total across its counties in proportion to the damage table's prediction
        piece = np.where(df[e] > 0, table_lookup(damage_table(df, fit, t, e), df), 0) * df[e]
        share = piece / pd.Series(piece).groupby(df.storm_id).transform("sum").replace(0, np.nan)
        dollars = (share * df.storm_id.map(st.total)).fillna(0).to_numpy()
        out[c] = np.where(df[e] > 0, dollars / df[e].where(df[e] > 0, 1), 0)
    return out


def by_type(fit):
    out = basic(fit)                                                   # crop and people: unchanged
    wind = two_part(df, fit, "prop_wind_usd", "bldg_value", FEATURES)
    surge = two_part(df, fit, "prop_surge_usd", "bldg_value", FEATURES, rows=coastal)   # surge only exists on the coast
    rain = two_part(df, fit, "prop_rain_usd", "bldg_value", FEATURES)
    out["property"] = np.minimum(wind + surge + rain, 1.0)
    return out


def tier_rescale(fit):
    out = basic(fit)
    for c, (t, e, _, _, cap) in CATS.items():
        for tier in TIERS:
            m = (df.storm_strength == tier).to_numpy()
            pred = (out[c] * df[e])[fit & m].sum()
            factor = df[t][fit & m].sum() / pred if pred > 0 else 1.0   # learned from the FIT storms only
            out[c] = np.where(m, out[c] * factor, out[c])
        if cap:
            out[c] = np.minimum(out[c], cap)
    return out


STRATEGIES = {"0 basic two-part": basic, "1 damage table": table, "2 top-down": top_down,
              "3 by damage type": by_type, "4a weights halfway": lambda f: basic(f, 0.5),
              "4b weights by dollars": lambda f: basic(f, 0.0), "4c tier rescale": tier_rescale}

# ---------- run the storm x county strategies ----------
preds = {}       # name -> DataFrame with <cat>_pred, <cat>_now, <cat>_score
for c, (t, e, e_now, usd, _) in CATS.items():
    df[f"{c}_rec"] = df[t] * usd
df["total_rec"] = sum(df[f"{c}_rec"] for c in CATS)
for name, fn in STRATEGIES.items():
    ratio = held_out(fn)
    p = pd.DataFrame(index=df.index)
    for c, (t, e, e_now, usd, _) in CATS.items():
        assert np.isfinite(ratio[c]).all() and (ratio[c] >= 0).all(), (name, c)
        p[f"{c}_score"], p[f"{c}_pred"], p[f"{c}_now"] = ratio[c], ratio[c] * df[e] * usd, ratio[c] * df[e_now] * usd
    p["total_pred"] = sum(p[f"{c}_pred"] for c in CATS)
    p["total_now"] = sum(p[f"{c}_now"] for c in CATS)
    p["total_score"] = p.total_pred
    preds[name] = p
    print(f"ran {name}: total predicted ${p.total_pred.sum() / 1e9:.1f}B")
assert np.isclose(preds["0 basic two-part"].total_pred.sum() / 1e9, 237.20, atol=0.01), "basic model changed"

# ---------- strategy 6: county x year two-part ----------
cy = df.assign(ts=(df.wind_ms >= 17.5).astype(int)).groupby(["fips", "year"]).agg(
    wind_ms=("wind_ms", "max"), wind_exposure=("log_wind_exposure", lambda s: np.log1p(np.expm1(s).sum())),
    log_dist_km=("log_dist_km", "min"), n_ts_storms=("ts", "sum"), coastal=("coastal", "first"),
    log_density=("log_density", "first"),
    **{col: (col, "first") for _, e, e_now, _, _ in CATS.values() for col in (e, e_now)},
    **{t: (t, "sum") for t, _, _, _, _ in CATS.values()}).reset_index()
cy["wind_excess"] = np.maximum(0, cy.wind_ms - 17.5)
cy["coastal_x_wind"] = cy.coastal * cy.wind_excess
CY_FEATURES = ["wind_ms", "wind_excess", "wind_exposure", "log_dist_km", "n_ts_storms", "coastal", "coastal_x_wind", "log_density"]
rng = np.random.default_rng(SEED)
year_fold = cy.year.map(dict(zip(rng.permutation(np.arange(2000, 2025)), np.arange(25) % 5))).to_numpy()
for c, (t, e, e_now, usd, cap) in CATS.items():
    ratio = np.zeros(len(cy))
    for k in range(5):
        held = year_fold == k
        ratio[held] = two_part(cy, ~held, t, e, CY_FEATURES, cap=cap, group="year")[held]   # every YEAR counts equally
    cy[f"{c}_pred"], cy[f"{c}_now"], cy[f"{c}_rec"] = ratio * cy[e] * usd, ratio * cy[e_now] * usd, cy[t] * usd
for k in ["pred", "now", "rec"]:
    cy[f"total_{k}"] = sum(cy[f"{c}_{k}"] for c in CATS)
print(f"ran 6 county-year two-part: {len(cy):,} county-years, total predicted ${cy.total_pred.sum() / 1e9:.1f}B")

# ---------- save held-out predictions per county x year (for metrics_eval.py) ----------
cols = [f"{c}_{k}" for c in ALL for k in ("pred", "rec")]
frames = []
for name, p in preds.items():
    f = pd.concat([df[["fips", "year"] + [f"{c}_rec" for c in ALL]], p[[f"{c}_pred" for c in ALL]]], axis=1)
    frames.append(f.groupby(["fips", "year"])[cols].sum().reset_index().assign(strategy=name))
frames.append(cy[["fips", "year"] + cols].assign(strategy="6 county-year two-part"))
pd.concat(frames).to_csv(HERE / "data" / "county_year_preds.csv", index=False)

# ---------- metrics ----------
storm_rows, strength_rows = [], []
for name, p in preds.items():
    for c in ALL:
        d = df[["storm_id", "storm_strength", "fips", "year"]].assign(pred=p[f"{c}_pred"], rec=df[f"{c}_rec"], score=p[f"{c}_score"])
        m, by = score(d)
        storm_rows.append({"strategy": name, "category": c, **m})
        strength_rows.append(by.assign(strategy=name, category=c))
storm_metrics = pd.DataFrame(storm_rows)
by_strength = pd.concat(strength_rows)
storm_metrics.round(5).to_csv(OUT / "strategies_storm_metrics.csv", index=False)
by_strength.round(3).to_csv(OUT / "strategies_by_strength.csv", index=False)

fema = pd.read_csv(HERE.parent / "final" / "counties.csv", dtype={"fips": str}).set_index("fips")
FEMA_COL = {"property": "nri_hrcn_eal_bldg_usd2023", "crop": "nri_hrcn_eal_ag_usd2023",
            "people": "nri_hrcn_eal_pope_usd2023", "total": "nri_hrcn_eal_total_usd2023"}


def county_year_view(frame, c):
    """Annual view: frame has fips, year, <c>_pred, <c>_now, <c>_rec at county-year level."""
    pred, rec = frame[f"{c}_pred"], frame[f"{c}_rec"]
    yr = frame.groupby("year")[[f"{c}_pred", f"{c}_rec"]].sum().reindex(range(2000, 2025), fill_value=0)
    hit = yr[yr[f"{c}_rec"] > 0]
    ours = (frame.groupby("fips")[f"{c}_now"].sum() / N_SEASONS).reindex(fema.index).fillna(0)
    rec_avg = (frame.groupby("fips")[f"{c}_rec"].sum() / N_SEASONS).reindex(fema.index).fillna(0)
    return {"pred_over_recorded": pred.sum() / rec.sum(),
            "roc_auc_damage_that_year": roc_auc(rec > 0, pred),
            "pr_auc_damage_that_year": pr_auc(rec > 0, pred), "base_rate": (rec > 0).mean(),
            "rank_corr_county_year": spearman(pred, rec),
            "rank_corr_years": spearman(yr[f"{c}_pred"], yr[f"{c}_rec"]),
            "typical_year_factor": 10 ** np.median(np.log10((hit[f"{c}_pred"] + 1) / hit[f"{c}_rec"])),
            "eal_B_per_yr": ours.sum() / 1e9, "recorded_avg_B": rec_avg.sum() / 1e9, "fema_B": fema[FEMA_COL[c]].sum() / 1e9,
            "eal_rank_corr_vs_recorded": spearman(ours, rec_avg), "eal_rank_corr_vs_fema": spearman(ours, fema[FEMA_COL[c]].fillna(0))}


cy_rows = []
for name, p in preds.items():
    frame = pd.concat([df[["fips", "year"] + [f"{c}_rec" for c in ALL]], p], axis=1).groupby(["fips", "year"]).sum().reset_index()
    cy_rows += [{"strategy": name, "category": c, **county_year_view(frame, c)} for c in ALL]
cy_rows += [{"strategy": "6 county-year two-part", "category": c, **county_year_view(cy, c)} for c in ALL]
cy_metrics = pd.DataFrame(cy_rows)
cy_metrics.round(5).to_csv(OUT / "strategies_county_year.csv", index=False)

# ---------- strategy 5: longer catalog (EAL only), using the damage table ----------
catalog = pd.read_csv(HERE / "data" / "catalog_table.csv", dtype={"fips": str})
everything = np.ones(len(df), bool)
catalog["now"] = 0.0
for c, (t, e, e_now, usd, _) in CATS.items():
    catalog["now"] += np.where(catalog[e_now] > 0, table_lookup(damage_table(df, everything, t, e), catalog), 0) * catalog[e_now] * usd
table_oof = df[["storm_id", "fips"]].assign(oof=preds["1 damage table"].total_now)
catalog = catalog.merge(table_oof, on=["storm_id", "fips"], how="left")
catalog["now"] = catalog.oof.fillna(catalog.now)                       # held-out predictions where we have them
eal25 = (catalog[catalog.year.between(2000, 2024)].groupby("fips").now.sum() / 25).reindex(fema.index).fillna(0)
eal51 = (catalog.groupby("fips").now.sum() / 51).reindex(fema.index).fillna(0)
f_tot = fema.nri_hrcn_eal_total_usd2023.fillna(0)
cat_eal = pd.DataFrame({"state": fema.state, "county_name": fema.county_name, "eal_25_seasons": eal25,
                        "eal_51_seasons": eal51, "fema_total": f_tot})
cat_eal.round(0).to_csv(OUT / "catalog_eal.csv")
big = f_tot > 5e6
catalog_summary = pd.DataFrame([{
    "seasons": n, "eal_B_per_yr": s.sum() / 1e9, "rank_corr_vs_fema": spearman(s, f_tot),
    "log_corr_vs_fema": np.corrcoef(np.log10(s[(s > 0) & (f_tot > 0)]), np.log10(f_tot[(s > 0) & (f_tot > 0)]))[0, 1],
    "counties_fema_gt0_ours_0": int(((f_tot > 0) & (s == 0)).sum()),
    "big_fema_counties_we_are_under_20pct": int((big & (s < 0.2 * f_tot)).sum()),
    "big_fema_counties_we_are_over_5x": int((big & (s > 5 * f_tot)).sum())} for n, s in [(25, eal25), (51, eal51)]])
season_tot = catalog.groupby("year").now.sum().reindex(range(1975, 2026), fill_value=0) / 1e9

# ---------- print ----------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
tot = storm_metrics[storm_metrics.category == "total"]
print("\nTOTAL damage, storm-held-out (2000-2024, 194 storms):")
print(tot[["strategy", "pred_over_recorded", "pred_over_recorded_95ci", "pred_over_recorded_without_top3",
           "typical_storm_factor", "typical_storm_factor_95ci", "rank_corr_storms", "pr_auc_any_damage",
           "rank_corr_county_year"]].round(2).to_string(index=False))
print("\nTOTAL predicted / recorded by storm strength:")
print(by_strength[by_strength.category == "total"].pivot(index="strategy", columns="strength", values="pred_over_recorded")[TIERS].round(2).to_string())
for c in CATS:
    print(f"\n{c}: predicted / recorded | typical storm | storm ranking | PR-AUC")
    s = storm_metrics[storm_metrics.category == c]
    print(s[["strategy", "pred_over_recorded", "typical_storm_factor", "rank_corr_storms", "pr_auc_any_damage"]].round(2).to_string(index=False))
print("\nANNUAL view, total damage (county-year level; strategy 6 is held out by year, the rest by storm):")
print(cy_metrics[cy_metrics.category == "total"][["strategy", "pred_over_recorded", "pr_auc_damage_that_year", "rank_corr_county_year",
      "rank_corr_years", "typical_year_factor", "eal_B_per_yr", "eal_rank_corr_vs_recorded", "eal_rank_corr_vs_fema"]].round(2).to_string(index=False))
print(f"  (recorded average ${cy_metrics.recorded_avg_B[cy_metrics.category == 'total'].iloc[0]:.1f}B/yr, FEMA ${f_tot.sum() / 1e9:.1f}B/yr, "
      f"county-years with damage {cy_metrics.base_rate[cy_metrics.category == 'total'].iloc[0]:.1%})")
print("\nSTRATEGY 5, longer catalog (damage table, total, 2024 exposure):"); print(catalog_summary.round(2).to_string(index=False))
print("  season totals $B: 1975-1999 mean %.1f, 2000-2024 mean %.1f; biggest seasons: %s" % (
    season_tot.loc[1975:1999].mean(), season_tot.loc[2000:2024].mean(),
    ", ".join(f"{y} {v:.0f}" for y, v in season_tot.nlargest(5).items())))
show = cat_eal[big].assign(r25=lambda d: d.eal_25_seasons / d.fema_total, r51=lambda d: d.eal_51_seasons / d.fema_total)
print("  counties that were far below FEMA with 25 seasons (ours / FEMA, 25 -> 51 seasons):")
print(show.nsmallest(8, "r25")[["state", "county_name", "r25", "r51"]].round(2).to_string())

# ---------- chart ----------
names = list(preds)
fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names), 3.4), sharey=True)
for ax, name in zip(axes, names):
    g = by_strength[(by_strength.category == "total") & (by_strength.strategy == name)].set_index("strength").reindex(TIERS)
    ax.bar(TIERS, g.pred_over_recorded, color=BLUE)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_yscale("log"); ax.set_title(name, fontsize=8); ax.tick_params(axis="x", labelsize=7, labelrotation=30)
axes[0].set_ylabel("Predicted $ / recorded $ (log; 1 = equal)")
fig.suptitle("Total damage: over- and under-prediction by storm strength, each strategy (storm-held-out)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "strategies_by_strength.png", dpi=150)
