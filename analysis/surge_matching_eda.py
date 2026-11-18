"""EDA: how sensitive is storm-surge / coastal-flood matching to the matching rule?

The pipeline (step 05) attaches a surge or coastal-flood record to a hurricane only if:
  - dates: the record's dates, widened 1 day before and 2 days after, overlap the
    storm's time within 500 km of the county   (window -1 / +2 days)
  - distance: the storm came within 200 km of the county
  - strength: the county had tropical-storm-force wind (>= 17.5 m/s) OR the storm
    passed within 100 km
This script reruns the matching with one setting changed at a time, and reports how
much surge damage (and how many records, crop $, deaths, injuries) would be matched.
Read-only: uses clean/ files, changes nothing in the pipeline.

Out: eda_output/surge_matching_summary.csv, eda_output/surge_matching.png,
     eda_output/surge_matching_summary.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "eda_output"
TS = 17.5      # tropical-storm-force wind, m/s

rec = pd.read_csv(ROOT / "clean" / "loss_records.csv", dtype={"fips": str})
rec = rec[rec.source == "surge"].copy()
rec["begin"] = pd.to_datetime(rec.begin_date)
rec["end"] = pd.to_datetime(rec.end_date)
sc = pd.read_csv(ROOT / "clean" / "storm_county.csv", dtype={"fips": str},
                 usecols=["storm_id", "fips", "min_dist_km", "wind_est_ms", "hours_within_100km",
                          "enter_500km_utc", "exit_500km_utc"],
                 parse_dates=["enter_500km_utc", "exit_500km_utc"])
names = pd.read_csv(ROOT / "final" / "storms.csv", usecols=["storm_id", "name"]).set_index("storm_id").name

# Every (record, storm) pair in the same county, computed once; each rule just filters it
pairs = rec[["record_id", "fips", "begin", "end"]].merge(sc, on="fips")
pairs["lead_days"] = (pairs.begin - pairs.exit_500km_utc).dt.total_seconds() / 86400   # record starts after storm left
pairs["lag_days"] = (pairs.enter_500km_utc - pairs.end).dt.total_seconds() / 86400     # storm arrived after record ended

TOTALS = {"prop_B": rec.prop_dmg_2023.sum() / 1e9, "records": len(rec)}


def match(max_km=200, min_wind=TS, or_within_100=True, before=1, after=2):
    """Records matched under one rule; closest qualifying storm wins (as in step 05)."""
    p = pairs[(pairs.lead_days <= before) & (pairs.lag_days <= after)]   # dates overlap the widened window
    p = p[p.min_dist_km <= max_km]
    strong = p.wind_est_ms >= min_wind if min_wind is not None else pd.Series(True, index=p.index)
    if or_within_100 and min_wind is not None:
        strong = strong | (p.hours_within_100km > 0)
    p = p[strong]
    n_storms = p.groupby("record_id").storm_id.nunique()
    best = p.sort_values("min_dist_km").drop_duplicates("record_id")[["record_id", "storm_id"]]
    m = rec.merge(best, on="record_id")
    return m, (n_storms > 1).sum()


SCENARIOS = [
    # (group, label, settings)
    ("baseline", "baseline: 200 km, TS wind or <100 km, -1/+2 d", {}),
    ("distance", "100 km", {"max_km": 100}),
    ("distance", "150 km", {"max_km": 150}),
    ("distance", "200 km (baseline)", {}),
    ("distance", "300 km", {"max_km": 300}),
    ("distance", "400 km", {"max_km": 400}),
    ("distance", "500 km", {"max_km": 500}),
    ("wind", "wind >= 25.7 m/s or <100 km", {"min_wind": 25.7}),
    ("wind", "wind >= 17.5 m/s or <100 km (baseline)", {}),
    ("wind", "wind >= 10 m/s or <100 km", {"min_wind": 10}),
    ("wind", "wind >= 17.5 m/s only (no <100 km rule)", {"or_within_100": False}),
    ("wind", "no wind / proximity rule", {"min_wind": None}),
    ("window", "-0 / +1 days", {"before": 0, "after": 1}),
    ("window", "-1 / +2 days (baseline)", {}),
    ("window", "-2 / +3 days", {"before": 2, "after": 3}),
    ("window", "-3 / +4 days", {"before": 3, "after": 4}),
    ("loosest", "500 km, no wind rule, -2/+3 d", {"max_km": 500, "min_wind": None, "before": 2, "after": 3}),
]

base, _ = match()
base_by_storm = base.groupby("storm_id").prop_dmg_2023.sum()
rows = []
for group, label, kw in SCENARIOS:
    m, n_ambiguous = match(**kw)
    by_storm = m.groupby("storm_id").prop_dmg_2023.sum()
    gain = by_storm.sub(base_by_storm, fill_value=0).sort_values(ascending=False)
    top_gain = ", ".join(f"{names.get(s, s)} {s[-4:]} +${v / 1e9:.1f}B" for s, v in gain.head(3).items() if v > 5e7)
    rows.append({
        "group": group, "rule": label,
        "records_matched": len(m), "pct_records": round(100 * len(m) / TOTALS["records"], 1),
        "prop_B": round(m.prop_dmg_2023.sum() / 1e9, 1),
        "pct_prop_dollars": round(100 * m.prop_dmg_2023.sum() / 1e9 / TOTALS["prop_B"], 1),
        "change_vs_baseline_B": round((m.prop_dmg_2023.sum() - base.prop_dmg_2023.sum()) / 1e9, 1),
        "crop_M": round(m.crop_dmg_2023.sum() / 1e6, 1),
        "deaths_direct": int(m.deaths_direct.sum()), "injuries_direct": int(m.injuries_direct.sum()),
        "storms_with_surge": m.storm_id.nunique(),
        "pct_ambiguous": round(100 * n_ambiguous / max(len(m), 1), 1),
        "pct_off_season": round(100 * m.begin.dt.month.isin([12, 1, 2, 3, 4, 5]).mean(), 1),
        "biggest_gains_vs_baseline": top_gain,
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "surge_matching_summary.csv", index=False)

# ---- Side question: crop, deaths, injuries in surge records ----
by_type = rec.groupby("event_type").agg(records=("record_id", "size"),
                                        prop_B=("prop_dmg_2023", lambda s: s.sum() / 1e9),
                                        crop_M=("crop_dmg_2023", lambda s: s.sum() / 1e6),
                                        deaths_direct=("deaths_direct", "sum"),
                                        injuries_direct=("injuries_direct", "sum"))
by_type.loc["all surge records"] = by_type.sum()
base_tot = base.groupby("event_type")[["prop_dmg_2023", "crop_dmg_2023", "deaths_direct", "injuries_direct"]].sum()
base_tot.loc["matched (baseline)"] = base_tot.sum()
season = rec.begin.dt.month.between(6, 11)
deadly = base[base.deaths_direct > 0].groupby("storm_id").deaths_direct.sum().sort_values(ascending=False)

# ---- Text summary ----
pd.set_option("display.width", 220)
lines = [f"Surge-type records in loss_records: {len(rec):,} ({rec.year.min()}-{rec.year.max()}), "
         f"property ${TOTALS['prop_B']:.1f}B (2023 $)",
         f"In hurricane season (Jun-Nov): {season.mean():.0%} of records, "
         f"{rec.prop_dmg_2023[season].sum() / 1e9 / TOTALS['prop_B']:.0%} of property $",
         "", "Match rate under each rule (one setting changed from baseline):",
         summary.drop(columns=["biggest_gains_vs_baseline"]).to_string(index=False), "",
         "Where the extra $ goes when loosening:",
         summary[["rule", "biggest_gains_vs_baseline"]].to_string(index=False), "",
         "Side question - what surge records contain (all, 1996-2025):",
         by_type.round(1).to_string(), "",
         "Matched under the baseline rule:", base_tot.assign(prop_dmg_2023=lambda d: (d.prop_dmg_2023 / 1e9).round(1),
                                                         crop_dmg_2023=lambda d: (d.crop_dmg_2023 / 1e6).round(1))
         .rename(columns={"prop_dmg_2023": "prop_B", "crop_dmg_2023": "crop_M"}).to_string(), "",
         "Storms with direct surge deaths (baseline match): " +
         ", ".join(f"{names.get(s, s)} {s[-4:]} {int(v)}" for s, v in deadly.items())]
text = "\n".join(lines)
(OUT / "surge_matching_summary.txt").write_text(text)
print(text)

# ---- Chart: % of surge property $ matched, one panel per setting ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (group, title) in zip(axes, [("distance", "Max storm distance"),
                                     ("wind", "Wind / proximity rule"),
                                     ("window", "Date window (days before / after)")]):
    g = summary[summary.group == group]
    ax.barh(g.rule.str.replace(" (baseline)", " *"), g.pct_prop_dollars, color="#2b6cb0")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("% of surge property $ matched")
axes[0].invert_yaxis()
fig.suptitle("Storm surge / coastal flood matching: sensitivity to the rule (* = baseline)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "surge_matching.png", dpi=150)
