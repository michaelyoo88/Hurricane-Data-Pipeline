"""
Basic exploratory data analysis (EDA) of the final hurricane tables.

Run from the project folder:   python analysis/eda.py
Reads final/ (built by pipeline/run_all.py) and writes:
    analysis/eda_output/eda_summary.txt   all printed numbers
    analysis/eda_output/*.png             five charts

Nothing here changes the data. It only looks at it.
Rows used: storm x county pairs in labeled years (1996-2024), the same
table the models train on (load_data.model_table()).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import TRAIN_YEARS, TEST_YEARS   # noqa: E402
from load_data import load, model_table      # noqa: E402

OUT = ROOT / "analysis" / "eda_output"
OUT.mkdir(parents=True, exist_ok=True)

# chart style: one blue, quiet grey axes
BLUE, INK, GREY = "#2a78d6", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# everything printed also goes to the summary file
lines = []
def say(text=""):
    print(text)
    lines.append(str(text))

def section(title):
    say()
    say("=" * 70)
    say(title)
    say("=" * 70)

def money(x):
    return f"${x / 1e9:,.1f}B" if abs(x) >= 1e9 else f"${x / 1e6:,.1f}M"


# ---------------------------------------------------------------- load
df = model_table(labeled_only=True)
cy = load("county_year")
df["period"] = np.where(df["year"].between(*TRAIN_YEARS), "train",
               np.where(df["year"].between(*TEST_YEARS), "test", "other"))
df["any_prop"] = df["prop_dmg_usd2023"] > 0
df["any_crop"] = df["crop_dmg_usd2023"] > 0
df["any_death"] = df["deaths_direct"] > 0


# ---------------------------------------------------------------- 1. shape
section("1. What the table looks like")
say(f"storm x county rows (labeled years): {len(df):,}   columns: {df.shape[1]}")
say(f"storms: {df['storm_id'].nunique():,}   counties: {df['fips'].nunique():,}   "
    f"years: {df['year'].min()}-{df['year'].max()}")
say(f"county-year rows (all years 1975-2025): {len(cy):,}")
say()
say("Missing values (columns with any):")
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
for col, n in miss.items():
    say(f"  {col:<22} {n:>8,}  ({n / len(df):.1%})")
say("  (prop_ratio is blank before 2000 because yearly housing starts in 2000)")


# ---------------------------------------------------------------- 2. split
section("2. Train / test split and how rare damage is")
t = df.groupby("period").agg(
    rows=("fips", "size"), storms=("storm_id", "nunique"),
    prop_rows=("any_prop", "sum"), crop_rows=("any_crop", "sum"),
    death_rows=("any_death", "sum"), prop_usd=("prop_dmg_usd2023", "sum"),
    crop_usd=("crop_dmg_usd2023", "sum"), deaths=("deaths_direct", "sum"))
say(t.to_string())
say()
for col, name in [("any_prop", "property"), ("any_crop", "crop"), ("any_death", "death")]:
    say(f"share of rows with zero {name:<8}: {1 - df[col].mean():.1%}")
say("-> targets are mostly zeros: a model must handle zero-inflation (Tweedie/Poisson).")


# ---------------------------------------------------------------- 3. size
section("3. How big is damage when it happens (rows with damage > 0)")
qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
for col in ["prop_dmg_usd2023", "crop_dmg_usd2023", "prop_ratio", "crop_ratio", "deaths_direct"]:
    pos = df.loc[df[col] > 0, col]
    vals = "  ".join(f"p{int(q * 100)}={pos.quantile(q):.3g}" for q in qs)
    say(f"{col:<18} n={len(pos):>5,}  {vals}  max={pos.max():.3g}")
top_share = df["prop_dmg_usd2023"].nlargest(100).sum() / df["prop_dmg_usd2023"].sum()
say()
say(f"top 100 rows hold {top_share:.0%} of all property damage -> very heavy tail")
say(f"crop damage larger than county farm value: {int(df['crop_exceeds_value'].sum())} rows")


# ---------------------------------------------------------------- 4. features
section("4. Does each feature line up with damage?")
features = ["wind_est_ms", "min_dist_km", "vmax_at_min_ms", "rmax_km_at_min",
            "hours_ts_force", "hours_hurricane_force", "wind_exposure",
            "hours_within_100km", "density_per_sqmi", "population",
            "bldg_value_usd2023", "ag_value_usd2023", "coastal", "month"]
target = np.log1p(df["prop_dmg_usd2023"])
corr = {}
for f in features:
    # Spearman = Pearson on ranks (no scipy needed)
    corr[f] = df[f].rank().corr(target.rank())
corr = pd.Series(corr).sort_values(key=abs, ascending=False)
say("Spearman rank correlation with log(1 + property damage):")
for f, r in corr.items():
    say(f"  {f:<22} {r:+.2f}")

say()
say("Property damage by wind speed bin (m/s):")
wind_bins = [0, 10, 17.5, 25, 32.9, 42, 50, 58, 100]
df["wind_bin"] = pd.cut(df["wind_est_ms"], wind_bins, right=False)
wb = df.groupby("wind_bin", observed=True).agg(
    rows=("fips", "size"), share_damaged=("any_prop", "mean"),
    median_ratio=("prop_ratio", lambda s: s[s > 0].median()),
    total_usd=("prop_dmg_usd2023", "sum"))
say(wb.to_string(float_format=lambda x: f"{x:.4g}"))

say()
say("Property damage by distance to storm center (km):")
df["dist_bin"] = pd.cut(df["min_dist_km"], [0, 25, 50, 100, 200, 300, 500])
db = df.groupby("dist_bin", observed=True).agg(
    rows=("fips", "size"), share_damaged=("any_prop", "mean"),
    total_usd=("prop_dmg_usd2023", "sum"))
say(db.to_string(float_format=lambda x: f"{x:.4g}"))


# ---------------------------------------------------------------- 5. rural/urban
section("5. Rural vs urban, coastal vs inland")
df["density_bin"] = pd.qcut(df["density_per_sqmi"], 4,
                            labels=["rural (low 25%)", "2nd", "3rd", "urban (top 25%)"])
for col in ["density_bin", "coastal"]:
    g = df.groupby(col, observed=True).agg(
        rows=("fips", "size"), share_damaged=("any_prop", "mean"),
        share_crop=("any_crop", "mean"),
        median_ratio=("prop_ratio", lambda s: s[s > 0].median()),
        total_usd=("prop_dmg_usd2023", "sum"))
    say(g.to_string(float_format=lambda x: f"{x:.4g}"))
    say()


# ---------------------------------------------------------------- 6. time / storms / counties
section("6. Over time, biggest storms, biggest counties")
by_year = df.groupby("year")["prop_dmg_usd2023"].sum() + df.groupby("year")["crop_dmg_usd2023"].sum()
say("Recorded damage per year (property + crop):")
for y, v in by_year.items():
    say(f"  {y}  {money(v)}")

say()
storms = (df.groupby(["storm_id", "name", "year"])
            .agg(prop=("prop_dmg_usd2023", "sum"), counties=("any_prop", "sum"))
            .sort_values("prop", ascending=False).head(10))
say("Top 10 storms by property damage:")
for (sid, name, yr), r in storms.iterrows():
    say(f"  {name:<10} {yr}  {money(r['prop']):>9}  counties with damage: {int(r['counties'])}")

say()
counties = (cy.query("has_labels").groupby(["fips", "county_name", "state"])["total_dmg_usd2023"]
              .sum().sort_values(ascending=False).head(10))
say("Top 10 counties by total recorded damage (1996-2024):")
for (fips, name, st), v in counties.items():
    say(f"  {fips} {name}, {st}: {money(v)}")

say()
lab = cy[cy["has_labels"]]
say(f"county-years in labeled period: {len(lab):,}; with any damage: "
    f"{(lab['total_dmg_usd2023'] > 0).sum():,} ({(lab['total_dmg_usd2023'] > 0).mean():.1%})")
say(f"counties that never had recorded damage: "
    f"{(lab.groupby('fips')['total_dmg_usd2023'].sum() == 0).sum():,} of {lab['fips'].nunique():,}")


# ---------------------------------------------------------------- charts
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)

# A. distribution of property damage when > 0
pos = df.loc[df["prop_dmg_usd2023"] > 0, "prop_dmg_usd2023"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(pos), bins=40, color=BLUE, edgecolor="#fcfcfb", linewidth=1)
ax.set_xlabel("property damage per storm x county (log10 $, 2023)")
ax.set_ylabel("rows")
ax.set_title(f"Property damage is spread over 6+ orders of magnitude (n={len(pos):,})", loc="left")
save(fig, "a_prop_damage_hist.png")

# B. share of rows with damage by wind bin
fig, ax = plt.subplots(figsize=(7, 4))
labels = [f"{int(i.left)}-{int(i.right)}" for i in wb.index]
ax.bar(labels, wb["share_damaged"] * 100, color=BLUE, width=0.7)
ax.set_xlabel("estimated peak wind at county (m/s)")
ax.set_ylabel("% of storm x county rows with property damage")
ax.set_title("Chance of damage rises steeply with wind", loc="left")
save(fig, "b_damage_rate_by_wind.png")

# C. median damage ratio by wind bin
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(labels, wb["median_ratio"], color=BLUE, linewidth=2, marker="o", markersize=8)
ax.set_yscale("log")
ax.set_xlabel("estimated peak wind at county (m/s)")
ax.set_ylabel("median damage ÷ building value (when > 0)")
ax.set_title("When damage happens, its size also grows with wind", loc="left")
save(fig, "c_ratio_by_wind.png")

# D. damage per year with train/test line
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(by_year.index, by_year.values / 1e9, color=BLUE, width=0.75)
ax.axvline(TEST_YEARS[0] - 0.5, color=GREY, linestyle="--", linewidth=1)
ax.text(TEST_YEARS[0] - 0.3, ax.get_ylim()[1] * 0.92, "test →", color=GREY)
ax.text(TRAIN_YEARS[0] - 0.3, ax.get_ylim()[1] * 0.92, "train →", color=GREY)
ax.set_ylabel("recorded damage ($B, 2023)")
ax.set_title("A few years (big storms) dominate total damage", loc="left")
save(fig, "d_damage_by_year.png")

# E. feature correlations
fig, ax = plt.subplots(figsize=(7, 5))
c = corr.sort_values()
ax.barh(c.index, c.values, color=BLUE, height=0.65)
ax.axvline(0, color=GREY, linewidth=0.8)
ax.set_xlabel("Spearman correlation with log(1 + property damage)")
ax.set_title("Which features line up with damage", loc="left")
save(fig, "e_feature_correlation.png")

(OUT / "eda_summary.txt").write_text("\n".join(lines))
say()
say(f"Saved summary and 5 charts to {OUT.relative_to(ROOT)}/")
