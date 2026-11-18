"""EDA: other NOAA event types that may hold hurricane damage - how much would match?

Candidates (records our labels do NOT use yet): Flood, Flash Flood, High/Strong Wind,
Tornado, Heavy Rain. For each, rerun the storm matching with one setting changed at a
time from the current surge rule (baseline: storm within 300 km; tropical-storm wind at
the county or storm within 100 km; record dates widened -1 / +2 days overlap the storm's
passage). Reports records and $ matched (property, crop), deaths, injuries.

Unlike surge, these event types happen year-round everywhere, so some matches will be
coincidence (a summer thunderstorm flood the same week as a far-off storm). Warning signs:
  pct_long_records  share of matched $ from records lasting > 3 days (long floods/droughts
                    overlap storms by chance)
  pct_ambiguous     records that qualify for 2+ storms
  pct_weak_wind     share of matched $ in counties with < 17.5 m/s wind (only matched
                    through the "within 100 km" rule or a looser rule)
Read-only. Out: eda_output/candidate_matching_summary.csv, candidate_matching_top_storms.csv,
candidate_matching.png, candidate_matching_summary.txt
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import SOURCE, CLEAN, DROP_STATES, cpi_factors   # noqa: E402

OUT = ROOT / "analysis" / "eda_output"
CACHE = ROOT / "analysis" / "cache" / "candidate_records.csv"
TS = 17.5
CANDIDATES = {"Flood": "Flood", "Flash Flood": "Flash Flood", "High Wind": "High/Strong Wind",
              "Strong Wind": "High/Strong Wind", "Tornado": "Tornado", "Heavy Rain": "Heavy Rain"}

# ---- 1. Pull candidate records with any damage or casualties (Housing split, as the pipeline) ----
if not CACHE.exists() or CACHE.stat().st_size < 1000:
    wanted = ";".join(CANDIDATES)
    cmd = (f"awk -F',' -v w='{wanted}' 'BEGIN {{n = split(w, a, \";\"); for (i = 1; i <= n; i++) keep[a[i]] = 1}} "
           f"NR==1 || ($3==\"Housing\" && ($7 in keep) && ($10+$12+$15+$17) > 0)' "
           f"'{SOURCE / 'nyc_fed_losses.csv'}' > '{CACHE}'")
    subprocess.run(cmd, shell=True, check=True)
rec = pd.read_csv(CACHE, dtype={"fips": str, "event_id": str},
                  usecols=["event_id", "fips", "begin_date", "end_date", "event_type", "damages_property",
                           "damages_crops", "injuries_direct", "fatalities_direct"])
rec["fips"] = rec.fips.str.zfill(5)
rec["year"] = rec.begin_date.str[:4].astype(int)
rec = rec[~rec.fips.str[:2].isin(DROP_STATES) & rec.year.between(1996, 2024)].copy()
f = rec.year.map(cpi_factors())
rec["prop"], rec["crop"] = rec.damages_property * f, rec.damages_crops * f
rec["begin"], rec["end"] = pd.to_datetime(rec.begin_date), pd.to_datetime(rec.end_date)
rec["long"] = (rec.end - rec.begin).dt.days > 3
rec["group"] = rec.event_type.map(CANDIDATES)
rec["record_id"] = rec.event_id + "_" + rec.fips
US = rec.groupby("group").agg(us_records=("record_id", "size"), us_prop_B=("prop", lambda s: s.sum() / 1e9))

# ---- 2. Candidate (record, storm) pairs in the same county; only records near any storm's dates ----
sc = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str},
                 usecols=["storm_id", "fips", "min_dist_km", "wind_est_ms", "hours_within_100km",
                          "enter_500km_utc", "exit_500km_utc"],
                 parse_dates=["enter_500km_utc", "exit_500km_utc"])
spans = sc.groupby("storm_id").agg(s=("enter_500km_utc", "min"), e=("exit_500km_utc", "max"))
near = pd.Series(False, index=rec.index)
for s, e in zip(spans.s - pd.Timedelta(days=4), spans.e + pd.Timedelta(days=4)):
    near |= (rec.begin <= e) & (rec.end >= s)
pairs = rec[near][["record_id", "fips", "begin", "end"]].merge(sc, on="fips")
pairs["lead"] = (pairs.begin - pairs.exit_500km_utc).dt.total_seconds() / 86400
pairs["lag"] = (pairs.enter_500km_utc - pairs.end).dt.total_seconds() / 86400


def match(max_km=300, min_wind=TS, or_within_100=True, before=1, after=2):
    p = pairs[(pairs.lead <= before) & (pairs.lag <= after) & (pairs.min_dist_km <= max_km)]
    if min_wind is not None:
        ok = p.wind_est_ms >= min_wind
        if or_within_100:
            ok |= p.hours_within_100km > 0
        p = p[ok]
    n = p.groupby("record_id").storm_id.nunique()
    best = p.sort_values("min_dist_km").drop_duplicates("record_id")[["record_id", "storm_id", "wind_est_ms"]]
    m = rec.merge(best, on="record_id")
    m["ambiguous"] = m.record_id.map(n) > 1
    return m


SCENARIOS = [
    ("baseline", "baseline (300 km, TS wind or <100 km, -1/+2 d)", {}),
    ("distance", "100 km", {"max_km": 100}), ("distance", "200 km", {"max_km": 200}),
    ("distance", "300 km *", {}), ("distance", "500 km", {"max_km": 500}),
    ("wind", "wind >= 25.7 or <100 km", {"min_wind": 25.7}), ("wind", "wind >= 17.5 or <100 km *", {}),
    ("wind", "wind >= 17.5 only", {"or_within_100": False}), ("wind", "wind >= 10 or <100 km", {"min_wind": 10}),
    ("wind", "no wind rule", {"min_wind": None}),
    ("window", "-0 / +1 d", {"before": 0, "after": 1}), ("window", "-1 / +2 d *", {}),
    ("window", "-2 / +3 d", {"before": 2, "after": 3}), ("window", "-3 / +4 d", {"before": 3, "after": 4}),
]

rows, top = [], []
for setting, label, kw in SCENARIOS:
    m = match(**kw)
    for g, d in m.groupby("group"):
        prop = d.prop.sum()
        rows.append({"candidate": g, "setting": setting, "rule": label,
                     "records": len(d), "pct_of_us_records": round(100 * len(d) / US.us_records[g], 2),
                     "prop_B": round(prop / 1e9, 2), "crop_B": round(d.crop.sum() / 1e9, 2),
                     "deaths_direct": int(d.fatalities_direct.sum()), "injuries_direct": int(d.injuries_direct.sum()),
                     "storms": d.storm_id.nunique(),
                     "pct_long_records": round(100 * d.prop[d.long].sum() / prop, 1) if prop else 0,
                     "pct_ambiguous": round(100 * d.ambiguous.mean(), 1),
                     "pct_weak_wind": round(100 * d.prop[d.wind_est_ms < TS].sum() / prop, 1) if prop else 0})
        if setting == "baseline":
            names = pd.read_csv(ROOT / "final" / "storms.csv", usecols=["storm_id", "name"]).set_index("storm_id").name
            t = d.groupby("storm_id").agg(prop_B=("prop", lambda s: s.sum() / 1e9), crop_B=("crop", lambda s: s.sum() / 1e9),
                                          deaths=("fatalities_direct", "sum")).nlargest(5, "prop_B")
            for sid, r in t.iterrows():
                top.append({"candidate": g, "storm": f"{names.get(sid, sid)} {sid[-4:]}", **r.round(2).to_dict()})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "candidate_matching_summary.csv", index=False)
top = pd.DataFrame(top)
top.to_csv(OUT / "candidate_matching_top_storms.csv", index=False)

# Current labels, for scale
labels = pd.read_csv(ROOT / "final" / "storm_county.csv", usecols=["year", "prop_dmg_usd2023", "crop_dmg_usd2023", "deaths_direct"])
labels = labels[labels.year.between(1996, 2024)]

pd.set_option("display.width", 220)
base = summary[summary.setting == "baseline"].set_index("candidate")
lines = [f"Current labels 1996-2024 (hurricane + surge records): property ${labels.prop_dmg_usd2023.sum() / 1e9:.1f}B, "
         f"crop ${labels.crop_dmg_usd2023.sum() / 1e9:.1f}B, direct deaths {labels.deaths_direct.sum():.0f}", "",
         "US totals of each candidate type 1996-2024 (any cause):", US.round(1).to_string(), "",
         "Baseline match by candidate:", base.drop(columns=["setting", "rule"]).to_string(), ""]
for g in sorted(summary.candidate.unique()):
    lines += [f"--- {g}: one setting changed at a time (* = baseline) ---",
              summary[summary.candidate == g].drop(columns=["candidate"]).to_string(index=False), ""]
lines += ["Top storms per candidate at baseline:", top.to_string(index=False)]
text = "\n".join(lines)
(OUT / "candidate_matching_summary.txt").write_text(text)
print(text)

# ---- Chart: property $ matched by distance, one small panel per candidate ----
cands = sorted(summary.candidate.unique())
fig, axes = plt.subplots(1, len(cands), figsize=(3 * len(cands), 3.2))
for ax, g in zip(axes, cands):
    d = summary[(summary.candidate == g) & (summary.setting == "distance")]
    ax.bar(d.rule.str.replace(" *", ""), d.prop_B, color="#2b6cb0")
    ax.set_title(g, fontsize=9)
    ax.set_xlabel("Max storm distance")
    ax.tick_params(axis="x", labelsize=7)
axes[0].set_ylabel("Property $ matched ($B, 2023 $)")
fig.suptitle("Candidate event types: property damage matched to hurricanes vs. distance rule", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "candidate_matching.png", dpi=150)
