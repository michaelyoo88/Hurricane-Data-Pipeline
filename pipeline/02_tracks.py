"""Step 02 - Storm tracks from HURDAT2.

HURDAT2 is a text file: each storm has a header line (id, name) followed by one
line per 6-hourly reading (time, position, max wind, pressure, ...).
We keep storms from 1975-2025, then fill in hourly positions by straight-line
interpolation between readings, so a fast storm can't "skip over" a county.

Out: clean/tracks_6h.csv      (one row per storm per 6-hour reading)
     clean/tracks_hourly.csv  (one row per storm per hour, with storm size)
"""
import numpy as np
import pandas as pd
from common import *


def value(text):
    """HURDAT2 marks missing numbers as -99 or -999."""
    v = float(text)
    return np.nan if v < 0 else v


rows = []
with open(SOURCE / "hurdat2_atlantic.txt") as f:
    for line in f:
        parts = [p.strip() for p in line.split(",")]
        if line[:2].isalpha():                      # header line, e.g. "AL092022, IAN, 52,"
            storm_id, name, year = parts[0], parts[1], int(parts[0][4:8])
            continue
        if not STORM_YEARS[0] <= year <= STORM_YEARS[1]:
            continue
        date, hhmm, _, status, lat, lon, wind_kt, pressure = parts[:8]
        rows.append({
            "storm_id": storm_id, "name": name, "year": year,
            "time_utc": pd.Timestamp(f"{date} {hhmm[:2]}:{hhmm[2:]}"),
            "status": status,                                   # HU, TS, TD, EX, ...
            "lat": float(lat[:-1]) * (1 if lat.endswith("N") else -1),
            "lon": float(lon[:-1]) * (-1 if lon.endswith("W") else 1),
            "vmax_ms": value(wind_kt) * KT_TO_MS,               # knots -> m/s
            "pmin_mb": value(pressure),
        })

tracks = pd.DataFrame(rows).sort_values(["storm_id", "time_utc"])
tracks.to_csv(CLEAN / "tracks_6h.csv", index=False)


def to_hourly(storm):
    """Interpolate one storm's readings to every whole hour."""
    hours = pd.date_range(storm.time_utc.min().ceil("h"), storm.time_utc.max().floor("h"), freq="h")
    t, th = storm.time_utc.astype("int64"), hours.astype("int64")
    out = pd.DataFrame({"time_utc": hours})
    for col in ["lat", "lon", "vmax_ms", "pmin_mb"]:
        ok = storm[col].notna()
        out[col] = np.interp(th, t[ok], storm[col][ok]) if ok.sum() >= 2 else np.nan
    # storm type = the most recent reading at or before each hour
    out = pd.merge_asof(out, storm[["time_utc", "status"]], on="time_utc")
    out.insert(0, "storm_id", storm.storm_id.iloc[0])
    return out


hourly = pd.concat([to_hourly(s) for _, s in tracks.groupby("storm_id")], ignore_index=True)
hourly["rmax_km"] = willoughby_rmax_km(hourly.vmax_ms, hourly.lat).round(1)   # storm size
hourly.to_csv(CLEAN / "tracks_hourly.csv", index=False)
print(f"02 tracks: {tracks.storm_id.nunique()} storms, {len(tracks):,} readings, "
      f"{len(hourly):,} hourly rows")
