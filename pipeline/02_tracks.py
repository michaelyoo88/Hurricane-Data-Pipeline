"""Step 02: parse HURDAT2 storm tracks.

In : source/hurdat2_atlantic.txt
Out: clean/tracks_6h.csv     (every best-track row, all fields parsed)
     clean/tracks_hourly.csv (linear interpolation to whole hours)
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("02_tracks")
QUADS = ["ne", "se", "sw", "nw"]
RADII = [f"r{k}_{q}_nm" for k in (34, 50, 64) for q in QUADS]


def num(x):
    v = float(x) if x not in ("", None) else np.nan
    return np.nan if v < 0 else v   # HURDAT2 codes missing values as -99 / -999


rows = []
n_storms_all = 0
with open(SRC / "hurdat2_atlantic.txt") as fh:
    for line in fh:
        parts = [p.strip() for p in line.rstrip("\n").split(",")]
        if line[:2].isalpha():                       # header line
            sid, name = parts[0], parts[1]
            year = int(sid[4:8])
            n_storms_all += 1
            continue
        if not (STORM_YEAR_MIN <= year <= STORM_YEAR_MAX):
            continue
        date, hhmm, rec, status, lat, lon = parts[:6]
        lat_v = float(lat[:-1]) * (1 if lat[-1] == "N" else -1)
        lon_v = float(lon[:-1]) * (-1 if lon[-1] == "W" else 1)
        row = {"storm_id": sid, "name": name, "year": year,
               "time_utc": pd.Timestamp(f"{date} {hhmm[:2]}:{hhmm[2:]}"),
               "record_id": rec, "status": status, "lat": lat_v, "lon": lon_v,
               "vmax_kt": num(parts[6]), "pmin_mb": num(parts[7])}
        row.update({k: num(v) for k, v in zip(RADII, parts[8:20])})
        row["rmw_nm"] = num(parts[20]) if len(parts) > 20 else np.nan
        rows.append(row)

t6 = pd.DataFrame(rows)
t6["vmax_ms"] = t6["vmax_kt"] * KT_TO_MS
log.write(f"Storms in file: {n_storms_all}; kept {STORM_YEAR_MIN}-{STORM_YEAR_MAX}: "
          f"{t6.storm_id.nunique()} storms, {len(t6)} best-track rows")
dups = t6.duplicated(["storm_id", "time_utc"]).sum()
t6 = t6.drop_duplicates(["storm_id", "time_utc"]).sort_values(["storm_id", "time_utc"])
log.write(f"Duplicate storm-time rows removed: {dups}")
log.write(f"Rows missing wind: {t6.vmax_ms.isna().sum()}")
log.write(f"Rows missing pressure: {t6.pmin_mb.isna().mean():.1%}; "
          f"missing radius of max wind: {t6.rmw_nm.isna().mean():.1%} (recorded from 2021 only)")
t6.to_csv(CLEAN / "tracks_6h.csv", index=False)

# Hourly interpolation of position, wind and pressure
hourly = []
for sid, g in t6.groupby("storm_id", sort=False):
    h0 = g.time_utc.min().ceil("h")
    h1 = g.time_utc.max().floor("h")
    grid = pd.date_range(h0, h1, freq="h")
    x = g.time_utc.astype("int64").to_numpy()
    xi = grid.astype("int64").to_numpy()
    out = pd.DataFrame({"storm_id": sid, "name": g.name.iat[0],
                        "year": g.year.iat[0], "time_utc": grid})
    for col in ("lat", "lon"):
        out[col] = np.interp(xi, x, g[col].to_numpy())
    for col in ("vmax_ms", "pmin_mb"):          # interpolate over valid values only
        ok = g[col].notna().to_numpy()
        out[col] = np.interp(xi, x[ok], g[col].to_numpy()[ok]) if ok.sum() >= 2 else np.nan
    # status = the most recent best-track status at or before each hour
    pos = np.searchsorted(x, xi, side="right") - 1
    out["status"] = g.status.to_numpy()[np.clip(pos, 0, None)]
    hourly.append(out)
th = pd.concat(hourly, ignore_index=True)

log.check(th.year.nunique() == STORM_YEAR_MAX - STORM_YEAR_MIN + 1,
          f"{th.year.nunique()} storm seasons present (expect {STORM_YEAR_MAX - STORM_YEAR_MIN + 1})")
log.check(th.vmax_ms.dropna().between(0, 90).all(),
          f"wind between 0 and 90 m/s (max {th.vmax_ms.max():.1f})")
log.check(th.lat.between(0, 75).all() and th.lon.between(-140, 20).all(),
          "positions plausible (a few storms, e.g. Bonnie 2022, cross into the Pacific)")
log.write(f"Hourly rows out: {len(th)}")
th.to_csv(CLEAN / "tracks_hourly.csv", index=False)
log.close()
