"""Step 03 - Which counties each storm reached, and how hard.

For every storm, measure the distance from each hourly position to every county
center (numpy does all counties at once). Keep counties the storm passed within
500 km of, and record:
  - how close it came and how strong it was then
  - estimated peak wind at the county: the strongest wind over all hours, using a
    Rankine vortex sized with the Willoughby formula (see common.py)
  - how long the county was battered: hours at tropical-storm force (17.5 m/s),
    hours at hurricane force (32.9 m/s), and total wind above 17.5 m/s summed
    over hours (wind_exposure, in m/s-hours: strength and duration in one number)
  - hours spent within 100 km, and when it entered/left the 500 km range
    (the time window is used in step 05 to match damage records to storms)

Out: clean/storm_county.csv  (one row per storm x county within 500 km)
"""
import numpy as np
import pandas as pd
from common import *

counties = pd.read_csv(CLEAN / "counties.csv", dtype={"fips": str})
tracks = pd.read_csv(CLEAN / "tracks_hourly.csv", parse_dates=["time_utc"])
county_lat = counties.pop_lat.to_numpy()[None, :]    # shape (1, n_counties)
county_lon = counties.pop_lon.to_numpy()[None, :]

pieces = []
for storm_id, s in tracks.groupby("storm_id"):
    # distance matrix: one row per hour, one column per county
    dist = haversine_km(s.lat.to_numpy()[:, None], s.lon.to_numpy()[:, None],
                        county_lat, county_lon)
    reached = dist.min(axis=0) <= MATCH_RADIUS_KM
    if not reached.any():
        continue                                   # storm stayed out at sea
    dist = dist[:, reached]
    vmax = s.vmax_ms.to_numpy()[:, None]
    rmax = s.rmax_km.to_numpy()[:, None]
    times = s.time_utc.to_numpy()

    closest_hour = dist.argmin(axis=0)
    # estimated wind at each county, every hour (rows = hours, columns = counties)
    hourly_wind = np.nan_to_num(rankine_wind_ms(vmax, rmax, dist))
    wind = hourly_wind.max(axis=0)
    # how long the county was battered, not just how hard
    hours_ts = (hourly_wind >= TROPICAL_STORM_MS).sum(axis=0)
    hours_hu = (hourly_wind >= HURRICANE_MS).sum(axis=0)
    exposure = np.clip(hourly_wind - TROPICAL_STORM_MS, 0, None).sum(axis=0)  # m/s-hours above 17.5
    inside = dist <= MATCH_RADIUS_KM
    first_hour = inside.argmax(axis=0)                             # first True
    last_hour = len(inside) - 1 - inside[::-1].argmax(axis=0)       # last True

    pieces.append(pd.DataFrame({
        "storm_id": storm_id,
        "fips": counties.fips.to_numpy()[reached],
        "year": int(storm_id[4:8]),
        "month": pd.DatetimeIndex(times[closest_hour]).month,
        "min_dist_km": dist.min(axis=0).round(1),
        "vmax_at_min_ms": s.vmax_ms.to_numpy()[closest_hour].round(2),
        "rmax_km_at_min": s.rmax_km.to_numpy()[closest_hour],
        "wind_est_ms": wind.round(2),
        "hours_ts_force": hours_ts,
        "hours_hurricane_force": hours_hu,
        "wind_exposure": exposure.round(1),
        "hours_within_100km": (dist <= NEAR_KM).sum(axis=0),
        "time_at_min_utc": times[closest_hour],
        "enter_500km_utc": times[first_hour],
        "exit_500km_utc": times[last_hour],
    }))

storm_county = pd.concat(pieces, ignore_index=True)
storm_county.to_csv(CLEAN / "storm_county.csv", index=False)
print(f"03 storm_county: {storm_county.storm_id.nunique()} storms reached land, "
      f"{len(storm_county):,} storm-county pairs")
