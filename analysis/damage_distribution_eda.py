"""EDA: how is recorded hurricane damage distributed?

Uses final/storm_county.csv (labeled years 1996-2024 only) with all matched record types
(hurricane/TS/TD + surge, rain flood, other wind, tornado). Charts:
  1. by_category_and_source  property $, crop $, and people (deaths) - split by record type
  2. by_year                 property $, crop $, deaths per year, biggest storm labeled
  3. by_storm_strength       storms grouped by the strongest wind any county felt
  4. by_county_wind          per storm x county: chance of any damage and $ by local wind
  5. storm_footprint         storm size (counties with tropical-storm wind) vs total $
  6. concentration           share of all damage from the top N storms
People in dollars follows FEMA's NRI: (deaths + injuries / 10) x value of a statistical life
($13.7M in 2024 $, converted to 2023 $). Shown next to counts, never added silently.

Out: eda_output/damage_distribution/*.png, damage_distribution_summary.txt, *.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import VSL, cpi_factors   # noqa: E402

OUT = ROOT / "analysis" / "eda_output" / "damage_distribution"
OUT.mkdir(parents=True, exist_ok=True)
BLUE = "#2b6cb0"
VSL_2023 = VSL * cpi_factors()[2024]

sc = pd.read_csv(ROOT / "final" / "storm_county.csv", dtype={"fips": str})
sc = sc[sc.has_labels].copy()
storms = pd.read_csv(ROOT / "final" / "storms.csv").set_index("storm_id")
SOURCES = {"prop_dmg_surge_usd2023": "Storm surge / coastal flood", "prop_dmg_rain_usd2023": "Rain flood",
           "prop_dmg_other_wind_usd2023": "High / strong wind", "prop_dmg_tornado_usd2023": "Tornado"}
sc["prop_hurricane"] = sc.prop_dmg_usd2023 - sc[list(SOURCES)].sum(axis=1)
sc["people_usd"] = (sc.deaths_direct + sc.injuries_direct / 10) * VSL_2023
sc["deaths_hurricane"] = sc.deaths_direct - sc.deaths_direct_extra
sc["crop_hurricane"] = sc.crop_dmg_usd2023 - sc.crop_dmg_extra_usd2023
lines = []


def label_bars(ax, values, fmt):
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt(v), va="center", fontsize=8)


# ---- 1. Category x record type ----
prop = pd.Series({"Hurricane / tropical storm": sc.prop_hurricane.sum(),
                  **{name: sc[c].sum() for c, name in SOURCES.items()}}) / 1e9
crop = pd.Series({"Hurricane / tropical storm": sc.crop_hurricane.sum(),
                  "All other types": sc.crop_dmg_extra_usd2023.sum()}) / 1e9
deaths = pd.Series({"Hurricane / tropical storm": sc.deaths_hurricane.sum(),
                    "All other types": sc.deaths_direct_extra.sum()})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.4, 1, 1]})
for ax, s, xl, fmt in [(axes[0], prop, "Property damage ($B, 2023 $)", lambda v: f"${v:.1f}B"),
                       (axes[1], crop, "Crop damage ($B, 2023 $)", lambda v: f"${v:.1f}B"),
                       (axes[2], deaths, "Direct deaths", lambda v: f"{v:,.0f}")]:
    s = s.sort_values()
    ax.barh(s.index, s.values, color=BLUE)
    label_bars(ax, s.values, fmt)
    ax.set_xlabel(xl)
    ax.set_xlim(0, s.max() * 1.3)
fig.suptitle("Recorded hurricane damage 1996-2024 by category and NOAA record type", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "1_by_category_and_source.png", dpi=150)
tot = {"property_B": sc.prop_dmg_usd2023.sum() / 1e9, "crop_B": sc.crop_dmg_usd2023.sum() / 1e9,
       "people_equiv_B": sc.people_usd.sum() / 1e9, "deaths": sc.deaths_direct.sum(), "injuries": sc.injuries_direct.sum()}
lines += ["1996-2024 totals (labeled storm x county rows):",
          f"  property ${tot['property_B']:.1f}B | crop ${tot['crop_B']:.1f}B | people {tot['deaths']:,.0f} direct deaths, "
          f"{tot['injuries']:,.0f} injuries = ${tot['people_equiv_B']:.1f}B at FEMA's value of life (${VSL_2023 / 1e6:.2f}M, 2023 $)",
          f"  share of $ (property / crop / people): {tot['property_B'] / (tot['property_B'] + tot['crop_B'] + tot['people_equiv_B']):.0%} / "
          f"{tot['crop_B'] / (tot['property_B'] + tot['crop_B'] + tot['people_equiv_B']):.0%} / "
          f"{tot['people_equiv_B'] / (tot['property_B'] + tot['crop_B'] + tot['people_equiv_B']):.0%}",
          "  property by record type ($B): " + ", ".join(f"{k} {v:.1f}" for k, v in prop.sort_values(ascending=False).items()), ""]

# ---- 2. By year ----
yr = sc.groupby("year").agg(prop_B=("prop_dmg_usd2023", "sum"), crop_B=("crop_dmg_usd2023", "sum"),
                            deaths=("deaths_direct", "sum"), storms=("storm_id", "nunique"))
yr[["prop_B", "crop_B"]] /= 1e9
st_year = sc.groupby(["year", "storm_id"]).prop_dmg_usd2023.sum().reset_index()
biggest = st_year.loc[st_year.groupby("year").prop_dmg_usd2023.idxmax()].set_index("year").storm_id.map(storms.name)
yr["biggest_storm"] = biggest
yr.round(2).to_csv(OUT / "by_year.csv")
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for ax, col, yl in [(axes[0], "prop_B", "Property ($B)"), (axes[1], "crop_B", "Crop ($B)"), (axes[2], "deaths", "Direct deaths")]:
    ax.bar(yr.index, yr[col], color=BLUE)
    ax.set_ylabel(yl)
    for y in yr[col].nlargest(3).index:
        ax.text(y, yr.loc[y, col], str(yr.loc[y, "biggest_storm"]).title(), ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, yr[col].max() * 1.15)
axes[2].set_xlabel("Year")
axes[0].set_title("Recorded hurricane damage by year (label = biggest storm that year)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "2_by_year.png", dpi=150)
decades = sc.assign(d=pd.cut(sc.year, [1995, 2004, 2014, 2024], labels=["1996-2004", "2005-2014", "2015-2024"])) \
    .groupby("d", observed=True)[["prop_dmg_usd2023", "crop_dmg_usd2023", "deaths_direct"]].sum()
lines += ["By period (property $B, crop $B, deaths):",
          decades.assign(prop_dmg_usd2023=lambda d: (d.prop_dmg_usd2023 / 1e9).round(1),
                         crop_dmg_usd2023=lambda d: (d.crop_dmg_usd2023 / 1e9).round(1)).to_string(), ""]

# ---- 3. By storm strength: strongest wind any county felt (Saffir-Simpson style bins) ----
BINS = [0, 17.5, 33, 43, 50, 58, 70, 999]
NAMES = ["TD (<17.5)", "TS (17.5-33)", "Cat 1 (33-43)", "Cat 2 (43-50)", "Cat 3 (50-58)", "Cat 4 (58-70)", "Cat 5 (70+)"]
st = sc.groupby("storm_id").agg(year=("year", "first"), max_wind=("wind_est_ms", "max"),
                                 prop=("prop_dmg_usd2023", "sum"), crop=("crop_dmg_usd2023", "sum"),
                                 deaths=("deaths_direct", "sum"), ts_counties=("wind_est_ms", lambda w: (w >= 17.5).sum()))
st["name"] = st.index.map(storms.name)
st["strength"] = pd.cut(st.max_wind, BINS, labels=NAMES, right=False)
by_strength = st.groupby("strength", observed=False).agg(storms=("prop", "size"), prop_B=("prop", lambda s: s.sum() / 1e9),
                                                        median_storm_M=("prop", lambda s: s.median() / 1e6),
                                                        crop_B=("crop", lambda s: s.sum() / 1e9), deaths=("deaths", "sum"))
by_strength.round(2).to_csv(OUT / "by_storm_strength.csv")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, col, xl in [(axes[0], "storms", "Number of storms"), (axes[1], "prop_B", "Total property ($B)"),
                    (axes[2], "median_storm_M", "Median property per storm ($M)"), (axes[3], "deaths", "Direct deaths")]:
    ax.barh(by_strength.index.astype(str), by_strength[col], color=BLUE)
    ax.set_xlabel(xl)
axes[0].invert_yaxis()
fig.suptitle("Damage by storm strength (strongest sustained wind estimated at any county, m/s)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "3_by_storm_strength.png", dpi=150)
lines += ["By storm strength (strongest county wind):", by_strength.round(1).to_string(), ""]

# ---- 4. Per storm x county: local wind vs chance of damage and $ ----
sc["wind_bin"] = pd.cut(sc.wind_est_ms, [0, 10, 17.5, 25, 33, 43, 50, 999],
                        labels=["<10", "10-17.5", "17.5-25", "25-33", "33-43", "43-50", "50+"], right=False)
by_wind = sc.groupby("wind_bin", observed=False).agg(
    rows=("fips", "size"), pct_with_damage=("prop_dmg_usd2023", lambda s: 100 * (s > 0).mean()),
    prop_B=("prop_dmg_usd2023", lambda s: s.sum() / 1e9),
    median_when_damaged_K=("prop_dmg_usd2023", lambda s: s[s > 0].median() / 1e3))
by_wind.round(2).to_csv(OUT / "by_county_wind.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col, yl in [(axes[0], "pct_with_damage", "% of storm-county pairs with any damage"),
                    (axes[1], "median_when_damaged_K", "Median $ when damaged ($K, log)"),
                    (axes[2], "prop_B", "Total property ($B)")]:
    ax.bar(by_wind.index.astype(str), by_wind[col], color=BLUE)
    ax.set_xlabel("Wind estimated at the county (m/s)")
    ax.set_ylabel(yl, fontsize=8)
axes[1].set_yscale("log")
fig.suptitle("Per storm x county: local wind vs. recorded damage", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "4_by_county_wind.png", dpi=150)
lines += ["Per storm x county by local wind:", by_wind.round(2).to_string(), ""]

# ---- 5. Storm footprint vs total damage ----
d = st[st.prop > 0]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(d.ts_counties.clip(lower=0.8), d.prop / 1e9, color=BLUE, alpha=0.6, s=18)
ax.set_xscale("log"); ax.set_yscale("log")
for sid, r in d.nlargest(8, "prop").iterrows():
    ax.annotate(f"{str(r['name']).title()} {r.year}", (max(r.ts_counties, 0.8), r.prop / 1e9), fontsize=7,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Storm footprint: counties with tropical-storm-force wind (log)")
ax.set_ylabel("Recorded property damage ($B, log)")
ax.set_title("Bigger storms do more damage, but the spread is huge", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "5_storm_footprint.png", dpi=150)
rho = d.ts_counties.rank().corr((d.prop).rank())
lines += [f"Storm footprint vs property $ (storms with damage): rank correlation {rho:.2f}", ""]

# ---- 6. Concentration ----
order = st.prop.sort_values(ascending=False)
cum = order.cumsum() / order.sum()
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(cum) + 1), 100 * cum.values, color=BLUE)
ax.set_xscale("log")
ax.set_xlabel("Top N storms (log scale)")
ax.set_ylabel("% of all recorded property damage")
for n in [1, 3, 10]:
    ax.annotate(f"top {n}: {100 * cum.iloc[n - 1]:.0f}%", (n, 100 * cum.iloc[n - 1]), fontsize=8,
                xytext=(5, -12), textcoords="offset points")
ax.set_title(f"A few storms hold most of the damage ({len(order)} storms, 1996-2024)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "6_concentration.png", dpi=150)
top10 = st.nlargest(10, "prop")[["name", "year", "max_wind", "prop", "crop", "deaths"]]
top10["prop"] = (top10.prop / 1e9).round(1); top10["crop"] = (top10.crop / 1e9).round(2)
lines += [f"Concentration: top 1 storm {100 * cum.iloc[0]:.0f}%, top 3 {100 * cum.iloc[2]:.0f}%, "
          f"top 10 {100 * cum.iloc[9]:.0f}% of property $; storms with $0 recorded: {(order == 0).sum()} of {len(order)}",
          "Top 10 storms (property $B, crop $B, deaths):", top10.to_string(), ""]

text = "\n".join(lines)
(OUT / "damage_distribution_summary.txt").write_text(text)
print(text)
