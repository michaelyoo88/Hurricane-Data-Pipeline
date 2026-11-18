"""Sandbox step 1: the experiment table + the splits every experiment uses.

Starts from models/data/model_table.csv (storm x county, 2000-2024) and adds
  track features   right_of_track (+1 = county directly right of the storm's motion, -1 = left),
                   fwd_speed_kmh (storm forward speed at closest approach),
                   dvmax_24h (change in storm intensity over the previous 24 h; negative = weakening)
  storm features   storm_peak_wind (strongest wind any county felt), log_footprint
                   (log of building value under tropical-storm wind) - same value for every row of a storm
  splits           set = "dev" (70% of storms) or "test" (30%, LOCKED: scored once, at the very end)
                   fold_r0..fold_r2 = three different 5-fold splits of the dev storms
All splits are by whole storm and stratified: within each strength tier, storms are sorted by
recorded damage and dealt out in blocks, so every part gets its share of big and small storms.

Out: sandbox/data/table.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import haversine_km   # noqa: E402

SEED, TEST_SHARE, N_FOLDS, N_REPEATS = 42, 0.30, 5, 3

df = pd.read_csv(ROOT / "models" / "data" / "model_table.csv", dtype={"fips": str})
sc = pd.read_csv(ROOT / "final" / "storm_county.csv", dtype={"fips": str},
                 usecols=["storm_id", "fips", "time_at_min_utc", "min_dist_km", "hours_hurricane_force"],
                 parse_dates=["time_at_min_utc"])
df = df.merge(sc, on=["storm_id", "fips"], how="left", validate="one_to_one")
counties = pd.read_csv(ROOT / "final" / "counties.csv", dtype={"fips": str}, usecols=["fips", "pop_lat", "pop_lon"])
df = df.merge(counties, on="fips", how="left")
tracks = pd.read_csv(ROOT / "final" / "storm_tracks.csv", usecols=["storm_id", "time_utc", "lat", "lon", "vmax_ms"],
                     parse_dates=["time_utc"])


def track_at(offset_hours, suffix):
    """Storm position / intensity at (closest approach + offset). Missing where the track doesn't reach."""
    t = tracks.rename(columns={"lat": f"lat{suffix}", "lon": f"lon{suffix}", "vmax_ms": f"vmax{suffix}"})
    key = df[["storm_id"]].assign(time_utc=df.time_at_min_utc.dt.round("h") + pd.Timedelta(hours=offset_hours))
    return key.merge(t, on=["storm_id", "time_utc"], how="left").drop(columns=["storm_id", "time_utc"]).set_index(df.index)


now, before, after, day_ago = track_at(0, "0"), track_at(-3, "a"), track_at(3, "b"), track_at(-24, "d")
p = pd.concat([now, before, after, day_ago], axis=1)
assert p.lat0.notna().all(), "closest-approach time not found on the hourly track"
# at the ends of a track use the point we do have
for s in ["a", "b"]:
    p[f"lat{s}"], p[f"lon{s}"] = p[f"lat{s}"].fillna(p.lat0), p[f"lon{s}"].fillna(p.lon0)
hours = np.where(before.lata.notna(), 3, 0) + np.where(after.latb.notna(), 3, 0)
moved_km = haversine_km(p.lata, p.lona, p.latb, p.lonb)
df["fwd_speed_kmh"] = np.where(hours > 0, moved_km / np.maximum(hours, 1), np.nan)
df["fwd_speed_kmh"] = df.fwd_speed_kmh.fillna(df.fwd_speed_kmh.median())


def bearing(lat1, lon1, lat2, lon2):
    """Compass direction (radians) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(x, y)


heading = bearing(p.lata, p.lona, p.latb, p.lonb)
to_county = bearing(p.lat0, p.lon0, df.pop_lat, df.pop_lon)
df["right_of_track"] = np.where(moved_km > 1, np.sin(to_county - heading), 0.0)     # 0 when the storm is stalled
df["dvmax_24h"] = (p.vmax0 - p.vmaxd).fillna(0.0)
df["log_fwd_speed"] = np.log1p(df.fwd_speed_kmh)
df["coastal_x_right"] = df.coastal * df.right_of_track * (df.wind_ms >= 17.5)

# storm-level features (from wind and exposure only, never from damage)
g = df.groupby("storm_id")
df["storm_peak_wind"] = g.wind_ms.transform("max")
df["log_footprint"] = np.log1p((df.bldg_value * (df.wind_ms >= 17.5)).groupby(df.storm_id).transform("sum") / 1e9)

# ---- splits, by whole storm, stratified by tier and recorded damage ----
rng = np.random.default_rng(SEED)
st = g.agg(tier=("storm_strength", "first"), dmg=("prop_usd", "sum")).reset_index()
st["dmg"] += df.groupby("storm_id").crop_usd.sum().to_numpy()


def deal(storms_sorted, n_groups, rng):
    """Deal storms (already sorted by damage) into n_groups, shuffling within each block of n_groups."""
    out = {}
    for start in range(0, len(storms_sorted), n_groups):
        block = storms_sorted[start:start + n_groups]
        for s, k in zip(block, rng.permutation(n_groups)[:len(block)]):
            out[s] = int(k)
    return out


is_test = {}
for tier, t in st.groupby("tier"):
    order = t.sort_values("dmg", ascending=False).storm_id.to_list()
    slot = deal(order, 10, rng)                       # 10 slots per block; slots 0-2 -> test (30%)
    is_test.update({s: slot[s] < round(10 * TEST_SHARE) for s in order})
st["set"] = st.storm_id.map(is_test).map({True: "test", False: "dev"})
for r in range(N_REPEATS):
    fold = {}
    for tier, t in st[st.set == "dev"].groupby("tier"):
        fold.update(deal(t.sort_values("dmg", ascending=False).storm_id.to_list(), N_FOLDS, rng))
    st[f"fold_r{r}"] = st.storm_id.map(fold).fillna(-1).astype(int)
# all-storm folds (added for round 6): the same recipe over all 194 storms, for a fresh comparison of
# models fixed in advance. Drawn AFTER the splits above, so those stay exactly as they were.
for r in range(N_REPEATS):
    fold = {}
    for tier, t in st.groupby("tier"):
        fold.update(deal(t.sort_values("dmg", ascending=False).storm_id.to_list(), N_FOLDS, rng))
    st[f"fold_all_r{r}"] = st.storm_id.map(fold).astype(int)
df = df.merge(st.drop(columns=["tier", "dmg"]), on="storm_id")

drop = ["time_at_min_utc", "pop_lat", "pop_lon"]
df.drop(columns=drop).to_csv(HERE / "data" / "table.csv", index=False)

# ---- report ----
assert len(df) == 97_179 and df.drop(columns=drop).notna().all().all()
s2 = df.groupby("storm_id").agg(set=("set", "first"), tier=("storm_strength", "first"), prop=("prop_usd", "sum"),
                                name=("name", "first"), year=("year", "first"), f0=("fold_r0", "first"))
order = ["TD / TS", "Cat 1-2", "Cat 3+"]
print(f"table: {len(df):,} rows, {len(s2)} storms")
print("storms by set and tier:"); print(pd.crosstab(s2.set, s2.tier)[order].to_string())
print("recorded property $B by set and tier:")
print((s2.pivot_table(index="set", columns="tier", values="prop", aggfunc="sum") / 1e9).round(1)[order].to_string())
print("locked test Cat 3+:", ", ".join(f"{r['name'].title()} {r.year}" for _, r in s2[(s2.set == "test") & (s2.tier == "Cat 3+")].iterrows()))
print("dev folds (repeat 0), Cat 3+ storms and property $B per fold:")
d = s2[s2.set == "dev"]
print(pd.DataFrame({"cat3": d[d.tier == "Cat 3+"].groupby("f0").size(), "prop_B": (d.groupby("f0").prop.sum() / 1e9).round(1)}).T.to_string())
print("new features:"); print(df[["right_of_track", "fwd_speed_kmh", "dvmax_24h", "storm_peak_wind", "log_footprint"]].describe().round(2).loc[["mean", "min", "50%", "max"]].to_string())
