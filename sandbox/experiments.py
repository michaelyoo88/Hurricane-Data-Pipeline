"""Run a named round of experiments on the DEV storms and append the results to results/experiments.csv.

    python3 sandbox/experiments.py round1a
Every model is scored the same way: 3 repeats of stratified 5-fold, whole storms held out.
The locked test storms are never touched here.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from lib import BASE, J2, REGIONS, REST, STORM, TRACK, WIND_BINS, capped, cross_validate, load

HERE = Path(__file__).resolve().parent
ROUNDS = {
    "round1a": {
        "A0 two-part (base)": {"features": BASE},
        "A1 table wind x coast": {"kind": "table", "keys": ["wind_bin", "coast"]},
        "A2 table tier x wind x coast": {"kind": "table", "keys": ["wind_bin", "storm_strength", "coast"]},
        "A3 by damage type": {"features": BASE, "by_type": True},
        "A4 base + tier correction": {"features": BASE, "rescale": "storm_strength"},
        "A5 by type + tier correction": {"features": BASE, "by_type": True, "rescale": "storm_strength"},
        "A6 weights halfway": {"features": BASE, "alpha": 0.5},
        "A7 weights by dollars": {"features": BASE, "alpha": 0.0},
    },
    "round1b": {
        "B1 base + storm features": {"features": BASE + STORM},
        "B2 base + track features": {"features": BASE + TRACK},
        "B3 base + storm + track": {"features": BASE + STORM + TRACK},
        "B4 by type + storm features": {"features": BASE + STORM, "by_type": True},
        "B5 suspect zeros >= 33 m/s": {"features": BASE, "suspect_wind": 33.0},
        "B6 suspect zeros >= 25 m/s": {"features": BASE, "suspect_wind": 25.0},
        "B7 storm effect (penalty 1)": {"features": BASE, "storm_penalty": 1.0},
        "B8 storm effect (penalty 5)": {"features": BASE, "storm_penalty": 5.0},
    },
    # Round 2: the base model's wind curve grows without limit (predicted $414B vs $93B recorded in the
    # 50+ m/s bin). Try other shapes for the wind response; everything else unchanged.
    "round2": {
        "C1 log wind (power law)": {"features": ["log_wind", "coastal_x_logwind"] + REST},
        "C2 log wind, no coast x wind": {"features": ["log_wind"] + REST},
        "C3 wind bins": {"features": WIND_BINS + ["coastal_x_ts"] + REST},
        "C4 wind capped at 50 m/s": {"features": ["wind_cap", "wind_excess_cap", "coastal_x_wind_cap"] + REST},
        "C5 Emanuel curve": {"features": ["log_emanuel", "coastal_x_ts"] + REST},
        "C6 log wind + squared": {"features": ["log_wind", "log_wind_sq", "coastal_x_logwind"] + REST},
        "C7 log wind + Emanuel": {"features": ["log_wind", "log_emanuel", "coastal_x_logwind"] + REST},
    },
    # Round 3: capped wind (C4) fixed the runaway curve. Re-test the planned add-ons on top of it.
    "round3a": {
        "D0 cap 40": {"features": capped(40)},
        "D1 cap 45": {"features": capped(45)},
        "D2 cap 55": {"features": capped(55)},
        "D3 cap 60": {"features": capped(60)},
        "D4 cap50 + storm size": {"features": capped() + ["log_footprint"]},
        "D5 cap50 + storm size + peak": {"features": capped() + STORM},
        "D6 cap50 + track features": {"features": capped() + TRACK},
        "D7 cap50 + right of track only": {"features": capped() + ["right_of_track", "coastal_x_right"]},
    },
    "round3b": {
        "E1 cap50 by damage type": {"features": capped(), "by_type": True},
        "E2 cap50 + tier correction": {"features": capped(), "rescale": "storm_strength"},
        "E3 cap50 weights halfway": {"features": capped(), "alpha": 0.5},
        "E4 cap50 weights by dollars": {"features": capped(), "alpha": 0.0},
        "E5 cap50 suspect zeros >= 33": {"features": capped(), "suspect_wind": 33.0},
        "E6 cap50 storm effect (pen 5)": {"features": capped(), "storm_penalty": 5.0},
        "E7 cap50 by type + storm size": {"features": capped() + ["log_footprint"], "by_type": True},
        "E8 wind bins + storm size": {"features": WIND_BINS + ["coastal_x_ts", "log_footprint"] + REST},
    },
    # Round 4: simplify for explainability. One wind term instead of two overlapping ones:
    # wind_excess_cap = wind above tropical-storm force (17.5 m/s), flat above 50 m/s.
    "round4": {
        "G1 simple: one wind term": {"features": ["wind_excess_cap", "coastal_x_wind_cap"] + REST},
        "G2 simple, raw wind term": {"features": ["wind_cap", "coastal_x_wind_cap"] + REST},
        "G3 simple - density": {"features": ["wind_excess_cap", "coastal_x_wind_cap", "log_wind_exposure", "log_dist_km", "coastal"]},
        "G4 simple - distance": {"features": ["wind_excess_cap", "coastal_x_wind_cap", "log_wind_exposure", "coastal", "log_density"]},
        "G5 simple - wind hours": {"features": ["wind_excess_cap", "coastal_x_wind_cap", "log_dist_km", "coastal", "log_density"]},
        "G6 simple - coast x wind": {"features": ["wind_excess_cap"] + REST},
        "G7 simple + right of track": {"features": ["wind_excess_cap", "coastal_x_wind_cap", "right_of_track", "coastal_x_right"] + REST},
        "G8 simple by damage type": {"features": ["wind_excess_cap", "coastal_x_wind_cap"] + REST, "by_type": True},
        "G9 simple + right, by type": {"features": ["wind_excess_cap", "coastal_x_wind_cap", "right_of_track", "coastal_x_right"] + REST, "by_type": True},
    },
    # Round 5: can the 6-feature model (G2) lose more features without losing skill?
    "round5": {
        "J1 G2 - density (5 features)": {"features": ["wind_cap", "coastal_x_wind_cap", "log_wind_exposure", "log_dist_km", "coastal"]},
        "J2 G2 - density - wind hours (4)": {"features": ["wind_cap", "coastal_x_wind_cap", "log_dist_km", "coastal"]},
        "J3 G2 - density - coast x wind (4)": {"features": ["wind_cap", "log_wind_exposure", "log_dist_km", "coastal"]},
        "J4 wind, distance, coastal (3)": {"features": ["wind_cap", "log_dist_km", "coastal"]},
    },
    # Round 6: ideas from the hurricane-loss literature and standard practice, each on top of J2.
    "round6": {
        "K0 J2 reference (4 inputs)": {"features": J2},
        "K1 asymmetric wind (forward speed)": {"features": ["wa_cap", "coastal_x_wa_cap", "log_dist_km", "coastal"]},
        "K2 Tweedie GLM, one equation": {"kind": "tweedie", "features": J2},
        "K3 Tweedie GLM, dollar weights": {"kind": "tweedie", "features": J2, "alpha": 0.0},
        "K4 Emanuel curve, fitted thresholds": {"kind": "emanuel"},
        "K5 J2 + region (pooled)": {"features": J2 + list(REGIONS)},
        "K6 J2 + latitude": {"features": J2 + ["pop_lat"]},
        "K7 J2 + county clusters": {"features": J2 + [f"cl{j}" for j in range(1, 6)]},
        "K8 average: J2 + damage table": {"kind": "average", "members": [{"features": J2}, {"kind": "table", "keys": ["wind_bin", "coast"]}]},
        "K9 average: J2 + by damage type": {"kind": "average", "members": [{"features": J2}, {"features": capped(), "by_type": True}]},
        "K10 J2 + era (own era)": {"features": J2 + ["era_2010s", "era_2020s"]},
        "K11 J2 + era (predict at 2000s level)": {"features": J2 + ["era_2010s", "era_2020s"], "zero_features": ["era_2010s", "era_2020s"]},
    },
    # Round 6b: give the Emanuel curve a fair floor; combine the two ideas that helped (latitude, averaging).
    "round6b": {
        "K4b Emanuel curve + floor": {"kind": "emanuel", "floor": True},
        "K12 average: J2+lat + by type": {"kind": "average", "members": [{"features": J2 + ["pop_lat"]}, {"features": capped(), "by_type": True}]},
    },
}

if __name__ == "__main__":
    name = sys.argv[1]
    df = load()
    rows = []
    for label, spec in ROUNDS[name].items():
        t = time.time()
        mean, runs = cross_validate(df, spec)
        rows.append({"round": name, "model": label, **mean, "d2_min": runs.d2.min(), "d2_max": runs.d2.max(),
                     "seconds": round(time.time() - t, 1)})
        print(f"  {label}: d2 {mean['d2']:.3f} ({time.time() - t:.0f}s)", flush=True)
    new = pd.DataFrame(rows)
    path = HERE / "results" / "experiments.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True)
    new.round(4).to_csv(path, index=False)
    pd.set_option("display.width", 250)
    show = new[new["round"] == name].drop(columns=["round", "seconds"]).set_index("model")
    print(show.round(2).to_string())
