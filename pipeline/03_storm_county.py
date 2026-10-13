"""Step 03: closest approach of every storm to every county (no wind formula).

In : clean/tracks_hourly.csv, clean/counties.csv
Out: clean/storm_county.csv  (one row per storm x county within MATCH_RADIUS_KM)
"""
import numpy as np
import pandas as pd
from common import *

log = StepLog("03_storm_county")
cty = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str},
                  usecols=["fips", "pop_lat", "pop_lon"])
tr = pd.read_csv(CLEAN / "tracks_hourly.csv", parse_dates=["time_utc"])
clat = cty.pop_lat.to_numpy()[None, :]
clon = cty.pop_lon.to_numpy()[None, :]

frames, n_hit = [], 0
for sid, g in tr.groupby("storm_id", sort=False):
    d = haversine_km(g.lat.to_numpy()[:, None], g.lon.to_numpy()[:, None], clat, clon)
    near = d.min(axis=0) <= MATCH_RADIUS_KM
    if not near.any():
        continue
    n_hit += 1
    d = d[:, near]
    times = g.time_utc.to_numpy()
    vmax = g.vmax_ms.to_numpy()
    i_min = d.argmin(axis=0)
    inside = d <= MATCH_RADIUS_KM
    first = inside.argmax(axis=0)
    last = inside.shape[0] - 1 - inside[::-1].argmax(axis=0)
    close = d <= NEAR_KM
    peak_close = np.where(close, vmax[:, None], np.nan)
    frames.append(pd.DataFrame({
        "storm_id": sid,
        "fips": cty.fips.to_numpy()[near],
        "year": g.year.iat[0],
        "min_dist_km": d.min(axis=0).round(1),
        "vmax_at_min_ms": vmax[i_min].round(2),
        "time_at_min": times[i_min],
        "hours_within_100km": close.sum(axis=0),
        "vmax_peak_within_100km_ms": np.round(
            np.nanmax(np.where(close.any(axis=0), peak_close, -1), axis=0), 2),
        "t_enter_500km": times[first],
        "t_exit_500km": times[last],
    }))
sc = pd.concat(frames, ignore_index=True)
sc.loc[sc.hours_within_100km == 0, "vmax_peak_within_100km_ms"] = np.nan
sc["month"] = pd.to_datetime(sc.time_at_min).dt.month

log.write(f"Storms passing within {MATCH_RADIUS_KM} km of a county: "
          f"{n_hit} of {tr.storm_id.nunique()}")
log.check(not sc.duplicated(["storm_id", "fips"]).any(), "no duplicate storm x county pairs")
log.write(f"Rows out: {len(sc)}; counties ever reached: {sc.fips.nunique()} of {len(cty)}")
log.write(f"Rows with eye within 100 km: {(sc.hours_within_100km > 0).sum()}")
sc.to_csv(CLEAN / "storm_county.csv", index=False)
log.close()
