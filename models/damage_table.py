"""Model idea 1: a damage table. No regression, just averages.

Sort every storm x county row into a box: wind at the county (7 bins) x coastal or inland.
For each box, loss ratio = share of exposure destroyed, learned from the fit storms.
Predicted $ = the box's loss ratio x the county's exposure. Same structure as FEMA's
EAL (exposure x how often x historic loss ratio).

Two ways to average a box:
  table_dollar     total damage in the box / total exposure in the box  (big storms dominate)
  table_storm_avg  each storm's own box ratio, then a plain average over storms (every storm equal)
Boxes with fewer than 3 fit storms use the wind bin's ratio regardless of coast.

Scored exactly like basic_model.py: same 5 storm folds, same metrics, EAL on 2024 exposure.
Out (models/results/): damage_table.csv, table_vs_basic_metrics.csv, table_vs_basic_by_strength.csv,
table_vs_basic_eal.csv, damage_table_property.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_table import VSL_2023
from evaluate import score
from glm import spearman

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BLUE, N_SEASONS, MIN_STORMS = "#2b6cb0", 25, 3
EDGES = [0, 10, 17.5, 25, 33, 43, 50, 999]                      # m/s; 33 = hurricane force
BINS = ["<10", "10-17.5", "17.5-25 (TS)", "25-33 (strong TS)", "33-43 (Cat 1)", "43-50 (Cat 2)", "50+ (Cat 3+)"]
CATEGORIES = {"property": ("prop_usd", "bldg_value", "bldg_value_2024", 1.0),
              "crop": ("crop_usd", "farm_timber_value", "farm_timber_value_2024", 1.0),
              "people": ("people_equiv", "population", "population_2024", VSL_2023)}
ALL = list(CATEGORIES) + ["total"]

df = pd.read_csv(HERE / "data" / "model_table.csv", dtype={"fips": str})
oof = pd.read_csv(HERE / "data" / "oof_basic.csv", dtype={"fips": str})
df = df.merge(oof, on=["storm_id", "fips"], validate="one_to_one")      # brings the folds + basic model's predictions
df["wind_bin"] = pd.cut(df.wind_ms, EDGES, labels=False, right=False)
df["coast"] = df.coastal.astype(int)
GRID = pd.MultiIndex.from_product([range(len(BINS)), [0, 1]], names=["wind_bin", "coast"])


def box_ratio(d, keys, target, exposure, how):
    """Loss ratio per box (keys = the columns that define a box)."""
    if how == "dollar":
        g = d.groupby(keys)[[target, exposure]].sum()
        return g[target] / g[exposure]
    s = d.groupby(keys + ["storm_id"])[[target, exposure]].sum()          # one ratio per storm per box
    return (s[target] / s[exposure]).groupby(level=list(range(len(keys)))).mean()


def fit_table(fit, target, exposure, how):
    """Return the 14-box table as a Series indexed by (wind_bin, coast)."""
    d = df[fit & (df[exposure] > 0)]
    ratio = box_ratio(d, ["wind_bin", "coast"], target, exposure, how).reindex(GRID)
    n_storms = d.groupby(["wind_bin", "coast"]).storm_id.nunique().reindex(GRID).fillna(0)
    by_wind = box_ratio(d, ["wind_bin"], target, exposure, how).reindex(range(len(BINS))).ffill().fillna(0)
    fallback = pd.Series(by_wind.reindex(GRID.get_level_values("wind_bin")).to_numpy(), index=GRID)
    return ratio.where(n_storms >= MIN_STORMS, fallback)


def lookup(table):
    return table.reindex(pd.MultiIndex.from_arrays([df.wind_bin, df.coast])).to_numpy()


# ---- 1. Held-out predictions for both table versions ----
MODELS = {"basic_two_part": None, "table_dollar": "dollar", "table_storm_avg": "storm"}
for model, how in MODELS.items():
    for cat, (target, exposure, exposure_now, usd) in CATEGORIES.items():
        if how is None:                                           # basic model: already in the file
            ratio = df[f"{cat}_score"].to_numpy()
        else:
            ratio = np.zeros(len(df))
            for k in sorted(df.fold.unique()):
                held = (df.fold == k).to_numpy()
                ratio[held] = lookup(fit_table(~held, target, exposure, how))[held]
            ratio = np.where(df[exposure] > 0, ratio, 0)
        assert np.isfinite(ratio).all() and (ratio >= 0).all()
        df[f"{model}|{cat}|score"] = ratio
        df[f"{model}|{cat}|pred"] = ratio * df[exposure] * usd
        df[f"{model}|{cat}|now"] = ratio * df[exposure_now] * usd
        df[f"{cat}_rec"] = df[target] * usd
    for k in ["pred", "now"]:
        df[f"{model}|total|{k}"] = sum(df[f"{model}|{c}|{k}"] for c in CATEGORIES)
    df[f"{model}|total|score"] = df[f"{model}|total|pred"]
df["total_rec"] = sum(df[f"{c}_rec"] for c in CATEGORIES)

# ---- 2. Same metrics for all three models ----
metrics, strength = [], []
for model in MODELS:
    for cat in ALL:
        d = df[["storm_id", "storm_strength", "fips", "year"]].assign(
            pred=df[f"{model}|{cat}|pred"], rec=df[f"{cat}_rec"], score=df[f"{model}|{cat}|score"])
        m, by = score(d)
        metrics.append({"category": cat, "model": model, **m})
        strength.append(by.assign(category=cat, model=model))
metrics = pd.DataFrame(metrics).sort_values(["category", "model"])
strength = pd.concat(strength)
metrics.round(3).to_csv(OUT / "table_vs_basic_metrics.csv", index=False)
strength.round(3).to_csv(OUT / "table_vs_basic_by_strength.csv", index=False)

# ---- 3. County EAL for each model vs recorded average vs FEMA ----
fema = pd.read_csv(HERE.parent / "final" / "counties.csv", dtype={"fips": str}).set_index("fips")
FEMA_COL = {"property": "nri_hrcn_eal_bldg_usd2023", "crop": "nri_hrcn_eal_ag_usd2023",
            "people": "nri_hrcn_eal_pope_usd2023", "total": "nri_hrcn_eal_total_usd2023"}
eal_rows = []
for model in MODELS:
    for cat in ALL:
        ours = (df.groupby("fips")[f"{model}|{cat}|now"].sum() / N_SEASONS).reindex(fema.index).fillna(0)
        rec = (df.groupby("fips")[f"{cat}_rec"].sum() / N_SEASONS).reindex(fema.index).fillna(0)
        f = fema[FEMA_COL[cat]].fillna(0)
        eal_rows.append({"category": cat, "model": model, "our_eal_B": ours.sum() / 1e9,
                         "recorded_avg_B": rec.sum() / 1e9, "fema_B": f.sum() / 1e9,
                         "rank_corr_vs_recorded": spearman(ours, rec), "rank_corr_vs_fema": spearman(ours, f)})
eal = pd.DataFrame(eal_rows).sort_values(["category", "model"])
eal.round(3).to_csv(OUT / "table_vs_basic_eal.csv", index=False)

# ---- 4. The table itself, fit on all 194 storms (what you would show on a slide) ----
everything = np.ones(len(df), bool)
parts = []
for cat, (target, exposure, _, _) in CATEGORIES.items():
    d = df[df[exposure] > 0]
    g = d.groupby(["wind_bin", "coast"])
    t = pd.DataFrame({"rows": g.size(), "storms": g.storm_id.nunique(),
                      "pct_rows_with_damage": 100 * g[target].apply(lambda s: (s > 0).mean()),
                      "ratio_dollar": fit_table(everything, target, exposure, "dollar"),
                      "ratio_storm_avg": fit_table(everything, target, exposure, "storm")}).reindex(GRID)
    parts.append(t.reset_index().assign(category=cat))
table = pd.concat(parts)
table.insert(0, "wind_ms", table.wind_bin.map(dict(enumerate(BINS))))
table["where"] = table.coast.map({0: "inland", 1: "coastal"})
table.drop(columns=["wind_bin", "coast"]).to_csv(OUT / "damage_table.csv", index=False)

# ---- Checks ----
assert len(df) == 97_179 and df.storm_id.nunique() == 194
assert np.isclose(metrics[(metrics.model == "basic_two_part") & (metrics.category == "total")].predicted_B.iloc[0],
                  237.20, atol=0.01), "basic model predictions changed"

# ---- Print ----
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
p = table[table.category == "property"][["wind_ms", "where", "rows", "storms", "pct_rows_with_damage",
                                         "ratio_dollar", "ratio_storm_avg"]].copy()
for c in ["ratio_dollar", "ratio_storm_avg"]:
    p[c] = p[c].map(lambda v: f"{100 * v:.5f}%")
print("PROPERTY damage table (all 194 storms): % of building value lost per storm, by box")
print(p.round(1).to_string(index=False))
print("\nheld-out metrics, all models:")
print(metrics[["category", "model", "pred_over_recorded", "pred_over_recorded_95ci", "pred_over_recorded_without_top3",
               "typical_storm_factor", "rank_corr_storms", "rank_corr_storms_95ci", "pr_auc_any_damage",
               "rank_corr_county_year"]].round(2).to_string(index=False))
print("\nTOTAL damage by storm strength (predicted / recorded):")
print(strength[strength.category == "total"].pivot(index="strength", columns="model", values="pred_over_recorded")
      .reindex(["TD / TS", "Cat 1-2", "Cat 3+"]).round(2).to_string())
print("\nEAL ($B/yr; 2024 exposure, 25 seasons):"); print(eal.round(2).to_string(index=False))

# ---- Chart: the property table, inland and coastal ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, where in zip(axes, ["inland", "coastal"]):
    t = table[(table.category == "property") & (table["where"] == where)]
    ax.bar(t.wind_ms, 100 * t.ratio_dollar, color=BLUE)
    for x, (v, n) in enumerate(zip(100 * t.ratio_dollar, t.storms)):
        ax.text(x, v * 1.3, f"{int(n)} storms", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_title(where, fontsize=10)
    ax.set_xlabel("Wind at the county (m/s)"); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
axes[0].set_ylabel("% of building value lost per storm (log)")
fig.suptitle("Property damage table: loss ratio by wind and coast (total damage / total value, 2000-2024)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "damage_table_property.png", dpi=150)
