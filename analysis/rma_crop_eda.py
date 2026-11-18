"""EDA: USDA RMA crop-insurance losses vs. NY Fed (NOAA) crop damage for hurricanes.

RMA "Summary of Business with Month of Loss" files (source/rma_cause_of_loss/colsom_YYYY.zip,
crop years 2000-2024): one row per county x crop x insurance plan x cause of loss x month of
loss, with liability (insured value) and indemnity (what insurance paid).

Questions:
  1. How big are hurricane-coded crop insurance losses, and where / when / which crops?
  2. Do they line up with our storms (same county, year, month as a storm's passage)?
  3. How do they compare with NY Fed crop damage, by year and by storm?
     NY Fed / RMA < 1 suggests NOAA under-records crop losses (RMA itself understates
     total loss: insured crops only, deductibles; timber mostly uninsured).

Causes: main = 92 Hurricane/Tropical Depression, 67 Storm Surge, 63 Cyclone.
Sensitivity = 61 Wind, 51 Flood, 31 Excess Moisture/Rain, 64 Tornado, counted ONLY when the
county / year / month matches a storm's passage (same footprint rule as our damage labels).
Read-only. Out: eda_output/rma/*.csv, *.png, rma_summary.txt; cache in analysis/cache/.
"""
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import CLEAN, DROP_STATES, EXTRA_MAX_KM, TROPICAL_STORM_MS, cpi_factors   # noqa: E402

RAW = ROOT / "source" / "rma_cause_of_loss"
OUT = ROOT / "analysis" / "eda_output" / "rma"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "analysis" / "cache" / "rma_indemnities_2000_2024.csv.gz"
BLUE = "#2b6cb0"
MAIN = {"92", "67", "63"}
SENSITIVITY = {"61", "51", "31", "64"}
# columns of the 30-field layout we keep (0-based) -> names (see COL_Summary_of_Business_with_Month_All_Years.pdf)
COLS = {0: "commodity_year", 1: "state_code", 2: "state", 3: "county_code", 4: "county_name", 6: "commodity",
        11: "cause_code", 12: "cause", 13: "month_loss", 15: "year_loss", 20: "liability", 28: "indemnity"}

# ---- 1. Read all years once (only rows that paid something) ----
if not CACHE.exists():
    parts = []
    for y in range(2000, 2025):
        with zipfile.ZipFile(RAW / f"colsom_{y}.zip") as z:
            with z.open(z.namelist()[0]) as f:
                d = pd.read_csv(f, sep="|", header=None, usecols=list(COLS), dtype=str, encoding="latin-1")
        d.columns = [COLS[i] for i in sorted(COLS)]
        d["indemnity"] = pd.to_numeric(d.indemnity, errors="coerce")
        parts.append(d[d.indemnity.fillna(0) != 0])
    raw = pd.concat(parts, ignore_index=True)
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].str.strip()
    raw.to_csv(CACHE, index=False, compression="gzip")
rma = pd.read_csv(CACHE, dtype={"state_code": str, "county_code": str, "cause_code": str, "month_loss": str})
rma["fips"] = rma.state_code.str.zfill(2) + rma.county_code.str.zfill(3)
rma["commodity"] = rma.commodity.str.upper().str.strip()   # RMA mixes "Cotton" / "COTTON" across years
rma["year_loss"] = pd.to_numeric(rma.year_loss, errors="coerce")
rma["month"] = pd.to_numeric(rma.month_loss, errors="coerce")
cpi = cpi_factors()
rma["indem_2023"] = rma.indemnity * rma.year_loss.map(cpi).fillna(rma.commodity_year.map(cpi))
lines = [f"RMA rows with an indemnity, crop years 2000-2024: {len(rma):,}; "
         f"total indemnity ${rma.indem_2023.sum() / 1e9:.1f}B (2023 $)"]

# ---- 2. National picture: causes ----
cause = rma.groupby(["cause_code", "cause"]).indem_2023.sum().div(1e9).sort_values(ascending=False)
cause_share = (cause / cause.sum()).round(3)
lines += ["", "Top causes, all US, all years ($B, share):",
          pd.DataFrame({"B": cause.round(2), "share": cause_share}).head(15).to_string()]

# ---- 3. Hurricane-coded losses (main causes) ----
hur = rma[rma.cause_code.isin(MAIN)].copy()
cont = ~hur.fips.str[:2].isin(DROP_STATES) & (hur.county_code != "999")
lines += ["", f"Hurricane-coded (codes 92/67/63): ${hur.indem_2023.sum() / 1e9:.2f}B "
              f"({hur.indem_2023.sum() / rma.indem_2023.sum():.1%} of all US indemnity); "
              f"in contiguous-US named counties ${hur[cont].indem_2023.sum() / 1e9:.2f}B; "
              f"county code 999 ('all other counties') ${hur[hur.county_code == '999'].indem_2023.sum() / 1e6:.0f}M"]
hur = hur[cont]
by_year = hur.groupby("year_loss").indem_2023.sum().div(1e9)
by_state = hur.groupby("state").indem_2023.sum().div(1e9).sort_values(ascending=False)
by_crop = hur.groupby("commodity").indem_2023.sum().div(1e9).sort_values(ascending=False)
by_month = hur.groupby("month").indem_2023.sum().div(1e9)
lines += ["", "Hurricane-coded by state ($B, top 10):", by_state.head(10).round(2).to_string(),
          "", "by crop ($B, top 12):", by_crop.head(12).round(2).to_string(),
          "", "by month of loss ($B):", by_month.round(2).to_string()]

# ---- 4. Match to storms: same county, year of loss, and month the storm passed ----
sc = pd.read_csv(CLEAN / "storm_county.csv", dtype={"fips": str},
                 usecols=["storm_id", "fips", "year", "min_dist_km", "wind_est_ms", "hours_within_100km",
                          "enter_500km_utc", "exit_500km_utc"],
                 parse_dates=["enter_500km_utc", "exit_500km_utc"])
sc = sc[(sc.min_dist_km <= EXTRA_MAX_KM) & ((sc.wind_est_ms >= TROPICAL_STORM_MS) | (sc.hours_within_100km > 0))]
sc_m = pd.concat([sc.assign(month=sc.enter_500km_utc.dt.month), sc.assign(month=sc.exit_500km_utc.dt.month)]) \
    .drop_duplicates(["storm_id", "fips", "month"])
cand = rma[rma.cause_code.isin(MAIN | SENSITIVITY) & ~rma.fips.str[:2].isin(DROP_STATES)].copy()
cand["row"] = np.arange(len(cand))
m = cand.merge(sc_m[["storm_id", "fips", "year", "month", "min_dist_km"]],
               left_on=["fips", "year_loss", "month"], right_on=["fips", "year", "month"])
n_storms = m.groupby("row").storm_id.nunique()
m = m.sort_values("min_dist_km").drop_duplicates("row")
m["ambiguous"] = m.row.map(n_storms) > 1
m["group"] = np.where(m.cause_code.isin(MAIN), "main", "sensitivity")
matched_main = m[m.group == "main"].indem_2023.sum()
lines += ["", f"Hurricane-coded $ that match a storm (county + year + month): "
              f"{matched_main / hur.indem_2023.sum():.0%} (${matched_main / 1e9:.2f}B); "
              f"ambiguous (2+ storms that month): {m[m.group == 'main'].ambiguous.mean():.1%} of matched rows",
          f"Sensitivity causes (wind/flood/rain/tornado) inside storm footprints: ${m[m.group == 'sensitivity'].indem_2023.sum() / 1e9:.2f}B",
          "  by cause: " + ", ".join(f"{c} ${v / 1e9:.2f}B" for c, v in
                                    m[m.group == "sensitivity"].groupby("cause").indem_2023.sum().sort_values(ascending=False).items())]

# ---- 5. Compare with NY Fed crop damage ----
fed = pd.read_csv(ROOT / "final" / "storm_county.csv", dtype={"fips": str},
                  usecols=["storm_id", "fips", "year", "has_labels", "crop_dmg_usd2023"])
fed = fed[fed.has_labels & fed.year.between(2000, 2024)]
names = pd.read_csv(ROOT / "final" / "storms.csv", usecols=["storm_id", "name"]).set_index("storm_id").name
st = pd.DataFrame({
    "nyfed_crop_M": fed.groupby("storm_id").crop_dmg_usd2023.sum() / 1e6,
    "rma_main_M": m[m.group == "main"].groupby("storm_id").indem_2023.sum() / 1e6,
    "rma_plus_sens_M": m.groupby("storm_id").indem_2023.sum() / 1e6}).fillna(0)
st = st[(st.nyfed_crop_M > 0) | (st.rma_main_M > 0)]
st.insert(0, "name", st.index.map(names))
st["year"] = st.index.str[-4:].astype(int)
st["nyfed_over_rma"] = (st.nyfed_crop_M / st.rma_main_M).replace(np.inf, np.nan)
st = st.sort_values("rma_main_M", ascending=False)
st.round(2).to_csv(OUT / "rma_vs_nyfed_by_storm.csv")

yr = pd.DataFrame({"nyfed_crop_B": fed.groupby("year").crop_dmg_usd2023.sum() / 1e9,
                   "rma_hurricane_coded_B": by_year.reindex(range(2000, 2025)),
                   "rma_matched_main_B": m[m.group == "main"].groupby("year_loss").indem_2023.sum() / 1e9}).fillna(0)
yr.round(3).to_csv(OUT / "rma_vs_nyfed_by_year.csv")

# County-level agreement: storm x county pairs where either source has crop $
fc = fed[fed.crop_dmg_usd2023 > 0].groupby(["storm_id", "fips"]).crop_dmg_usd2023.sum()
rc = m[m.group == "main"].groupby(["storm_id", "fips"]).indem_2023.sum()
both = pd.concat([fc.rename("fed"), rc.rename("rma")], axis=1).fillna(0)
agree = {"both": ((both.fed > 0) & (both.rma > 0)).sum(), "NY Fed only": ((both.fed > 0) & (both.rma == 0)).sum(),
         "RMA only": ((both.fed == 0) & (both.rma > 0)).sum()}
tot_f, tot_r = st.nyfed_crop_M.sum(), st.rma_main_M.sum()
med = st[(st.nyfed_crop_M > 0) & (st.rma_main_M > 0)].nyfed_over_rma.median()
lines += ["", f"2000-2024 totals: NY Fed crop ${tot_f / 1e3:.1f}B vs RMA hurricane-coded (matched) ${tot_r / 1e3:.1f}B "
              f"-> NY Fed / RMA = {tot_f / tot_r:.2f} overall; median storm ratio {med:.2f}",
          f"Storms with crop $ in: both {((st.nyfed_crop_M > 0) & (st.rma_main_M > 0)).sum()}, "
          f"NY Fed only {((st.nyfed_crop_M > 0) & (st.rma_main_M == 0)).sum()}, "
          f"RMA only {((st.nyfed_crop_M == 0) & (st.rma_main_M > 0)).sum()}",
          f"Storm x county pairs with crop $ in: " + ", ".join(f"{k} {v:,}" for k, v in agree.items()),
          "", "By year ($B):", yr.round(2).to_string(),
          "", "Top 15 storms by RMA hurricane indemnity ($M):", st.head(15).round(1).to_string()]

text = "\n".join(lines)
(OUT / "rma_summary.txt").write_text(text)
print(text)

# ---- Charts ----
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, sharey=True)
for ax, col, title in [(axes[0], "nyfed_crop_B", "NY Fed (NOAA) recorded crop damage"),
                       (axes[1], "rma_matched_main_B", "USDA RMA hurricane crop-insurance indemnity (matched to storms)")]:
    ax.bar(yr.index, yr[col], color=BLUE)
    ax.set_ylabel("$B (2023 $)")
    ax.set_title(title, fontsize=9, loc="left")
axes[1].set_xlabel("Year")
fig.tight_layout(); fig.savefig(OUT / "1_by_year.png", dpi=150)

d = st[(st.nyfed_crop_M > 0) | (st.rma_main_M > 0)]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
x, y = d.rma_main_M.clip(lower=0.01), d.nyfed_crop_M.clip(lower=0.01)
ax.scatter(x, y, color=BLUE, alpha=0.6, s=20)
lim = [0.01, max(x.max(), y.max()) * 2]
ax.plot(lim, lim, color="black", linewidth=0.8, linestyle="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
for sid, r in d.nlargest(10, "rma_main_M").iterrows():
    ax.annotate(f"{str(r['name']).title()} {r.year}", (max(r.rma_main_M, .01), max(r.nyfed_crop_M, .01)),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("RMA hurricane indemnity ($M, log; 0 plotted at 0.01)")
ax.set_ylabel("NY Fed crop damage ($M, log)")
ax.set_title("Per storm: NY Fed vs. crop insurance (dashed = equal)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "2_storm_scatter.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
c = by_crop.head(10).sort_values()
axes[0].barh(c.index, c.values, color=BLUE); axes[0].set_xlabel("Hurricane indemnity ($B, 2023 $)")
axes[0].set_title("By crop", fontsize=9)
s = by_state.head(10).sort_values()
axes[1].barh(s.index, s.values, color=BLUE); axes[1].set_xlabel("Hurricane indemnity ($B, 2023 $)")
axes[1].set_title("By state", fontsize=9)
fig.suptitle("USDA RMA hurricane-coded crop insurance losses, 2000-2024", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "3_crop_and_state.png", dpi=150)
