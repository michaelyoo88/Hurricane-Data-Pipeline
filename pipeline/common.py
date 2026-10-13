"""Shared settings and helpers for the hurricane data pipeline.

Every setting that encodes a project decision lives here, so a change of
mind (radius, years, dollar year) is a one-line edit followed by run_all.py.
"""
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source"          # raw downloads, never edited
CLEAN = ROOT / "clean"         # rebuilt by the pipeline
LOGS = CLEAN / "logs"

# --- Decisions locked Oct 11-12 ---------------------------------------------
STORM_YEAR_MIN, STORM_YEAR_MAX = 1975, 2025   # storms kept from HURDAT2
LABEL_YEAR_MIN, LABEL_YEAR_MAX = 1996, 2025   # years with NY Fed loss labels
BASE_DOLLAR_YEAR = 2023                       # all money converted to this year
NRI_DOLLAR_YEAR = 2023  # TODO: confirm in the NRI technical documentation
MATCH_RADIUS_KM = 500   # storm must pass within this distance to count
NEAR_KM = 100           # used for hours_within_100km
DATE_PAD_DAYS = 1       # slack on loss-record dates (local time vs UTC)

# Scope: contiguous 48 states + DC. Drop AK, HI and all territories.
EXCLUDED_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# Connecticut: old counties (NY Fed, Census 2020) vs planning regions (NRI)
CT_OLD_COUNTIES = {"09001", "09003", "09005", "09007",
                   "09009", "09011", "09013", "09015"}
CT_CROSSWALK = CLEAN / "ct_crosswalk.csv"     # built by 00_ct_crosswalk.py

# NY Fed weighting schemes chosen
WEIGHT_PROPERTY = "Housing"     # property and crop damage
WEIGHT_PEOPLE = "Population"    # deaths and injuries
WEIGHT_SENSITIVITY = "Area"     # crop damage, sensitivity check only

KT_TO_MS = 0.514444


class StepLog:
    """Prints progress and saves the same lines to clean/logs/<step>.txt."""

    def __init__(self, name):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.path = LOGS / f"{name}.txt"
        self.lines = []
        self.write(f"{name}  (run {dt.datetime.now():%Y-%m-%d %H:%M})")

    def write(self, msg=""):
        print(f"  [{self.name}] {msg}")
        self.lines.append(str(msg))

    def check(self, ok, msg):
        self.write(("PASS  " if ok else "WARN  ") + msg)
        return ok

    def close(self):
        self.path.write_text("\n".join(self.lines) + "\n")


def state_of(fips):
    return fips.str[:2]


def cpi_factors():
    """Factor that converts a year's dollars to BASE_DOLLAR_YEAR dollars.
    Annual average of monthly CPI-U; incomplete years are excluded."""
    cpi = pd.read_csv(SRC / "cpi_monthly.csv")
    cpi["year"] = cpi["observation_date"].str[:4].astype(int)
    g = cpi.groupby("year")["CPIAUCSL"]
    annual = g.mean()[g.size() == 12]
    return annual[BASE_DOLLAR_YEAR] / annual


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; inputs in degrees, numpy-broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def snake(text):
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")
