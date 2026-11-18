"""Data checkpoint: structure of every table in final/.

For each table: rows, what one row is, and per column: % filled (not blank),
% non-zero (numbers), min / max, and the data-dictionary description.
Out: eda_output/data_checkpoint.csv, eda_output/data_checkpoint.txt
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "final"
OUT = ROOT / "analysis" / "eda_output"
GRAIN = {
    "storm_county": "one storm x one county within 500 km of its track (1975-2025)",
    "storms": "one storm (HURDAT2, 1975-2025)",
    "counties": "one county (contiguous US + DC; static, mostly 2020 / FEMA NRI)",
    "storm_tracks": "one storm x one hour of its track",
    "county_year": "one county x one year (1975-2025): storm hits + damage summed over that year's storms",
    "county_stats_by_year": "one county x one year (2000-2024): exposure stats",
}
dd = pd.read_csv(FINAL / "data_dictionary.csv")
dd_cols = [c for c in dd.columns if c.lower() in ("table", "column", "description", "unit", "units", "source")]
dd = dd[dd_cols]

rows, text = [], []
for t, grain in GRAIN.items():
    df = pd.read_csv(FINAL / f"{t}.csv", dtype={"fips": str}, low_memory=False)
    text.append(f"\n=== {t}.csv: {len(df):,} rows x {df.shape[1]} columns; one row = {grain}")
    if "has_labels" in df.columns:
        text.append(f"    rows in labeled years (damage known): {df.has_labels.mean():.0%}")
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rows.append({
            "table": t, "column": c, "type": "number" if num else ("yes/no" if pd.api.types.is_bool_dtype(s) else "text/date"),
            "pct_filled": round(100 * s.notna().mean(), 1),
            "pct_nonzero": round(100 * (s.fillna(0) != 0).mean(), 1) if num else None,
            "min": (round(s.min(), 3) if num else s.dropna().astype(str).min()) if s.notna().any() else None,
            "max": (round(s.max(), 3) if num else s.dropna().astype(str).max()) if s.notna().any() else None,
            "n_unique": s.nunique(),
        })
prof = pd.DataFrame(rows)
if {"table", "column"} <= set(dd.columns):
    prof = prof.merge(dd, on=["table", "column"], how="left")
prof.to_csv(OUT / "data_checkpoint.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
out = []
for t in GRAIN:
    out.append([l for l in text if l.startswith(f"\n=== {t}.")][0])
    extra = [l for l in text if l.startswith("    rows in labeled") and text.index(l) == text.index([x for x in text if x.startswith(f"\n=== {t}.")][0]) + 1]
    out += extra
    out.append(prof[prof.table == t].drop(columns=["table"] + [c for c in ["description", "source", "unit", "units"] if c in prof]).to_string(index=False))
(OUT / "data_checkpoint.txt").write_text("\n".join(out))
print("\n".join(out))
