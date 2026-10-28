"""Settings and small helper functions shared by every pipeline step.

Every number that encodes a project decision lives here. To change a decision,
edit one line and re-run run_all.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Folders ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"   # raw downloads, never edited
CLEAN = ROOT / "clean"     # in-between tables, rebuilt every run
FINAL = ROOT / "final"     # tables we model on, rebuilt every run
CLEAN.mkdir(exist_ok=True)
FINAL.mkdir(exist_ok=True)

# ---- Years -------------------------------------------------------------------
STORM_YEARS = (1975, 2025)   # storms kept from HURDAT2 (satellite era onward)
LABEL_YEARS = (1996, 2024)   # years with damage records (2025 is incomplete in our download: 11 records, $0)
DOLLAR_YEAR = 2023           # every dollar amount is converted to this year
NRI_DOLLAR_YEAR = 2023       # TODO: confirm the year in FEMA's NRI documentation

# ---- Distances ---------------------------------------------------------------
MATCH_RADIUS_KM = 500   # a storm "reaches" a county if its center passes this close
NEAR_KM = 100           # used to count hours a storm spent very close

# ---- Wind --------------------------------------------------------------------
KT_TO_MS = 0.514444
WIND_DECAY = 0.5        # how fast wind fades outside the eyewall (standard Rankine value)
HURRICANE_MS = 32.9     # 64 knots = hurricane-force wind
TROPICAL_STORM_MS = 17.5  # 34 knots = tropical-storm-force wind

# ---- Money -------------------------------------------------------------------
VSL = 13.7e6   # FEMA NRI v1.20 value of a statistical life; 10 injuries = 1 death

# ---- Scope ---------------------------------------------------------------------
# Contiguous 48 states + DC only: drop Alaska, Hawaii and the territories.
DROP_STATES = {"02", "15", "60", "66", "69", "72", "78"}
# Connecticut's 8 old county codes (used by the NY Fed and Census 2020)
CT_OLD_COUNTIES = {"09001", "09003", "09005", "09007", "09009", "09011", "09013", "09015"}

# NY Fed splits each zone's damage across counties several ways; we use:
PROPERTY_WEIGHT = "Housing"     # property and crop damage follow where houses are
PEOPLE_WEIGHT = "Population"    # deaths and injuries follow where people are
CROP_CHECK_WEIGHT = "Area"      # crop damage by land area, kept as a sensitivity check

# ---- Modeling ---------------------------------------------------------------------
SEED = 42                  # pass to every random step so results are reproducible
TRAIN_YEARS = (2000, 2019) # yearly property/farm values start in 2000
TEST_YEARS = (2020, 2024)  # held out: never used for training or tuning


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between points on the Earth's surface (inputs in degrees).
    Works on numpy arrays, so one call can measure many storm-county pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def cpi_factors():
    """Multiply a year's dollars by factor[year] to get DOLLAR_YEAR dollars.
    Uses the yearly average of monthly CPI; incomplete years are skipped."""
    cpi = pd.read_csv(SOURCE / "cpi_monthly.csv")
    cpi["year"] = cpi["observation_date"].str[:4].astype(int)
    by_year = cpi.groupby("year")["CPIAUCSL"]
    yearly = by_year.mean()[by_year.size() == 12]
    return yearly[DOLLAR_YEAR] / yearly


def willoughby_rmax_km(vmax_ms, lat):
    """Radius of maximum wind (km): how far from the eye the strongest winds are.

    Willoughby, Darling & Rahn (2006), "Parametric representation of the primary
    hurricane vortex. Part II", Monthly Weather Review 134: 1102-1120, fitted to
    decades of aircraft measurements:
        Rmax = 46.4 * exp(-0.0155 * Vmax + 0.0169 * latitude)
    Stronger storms are tighter; storms farther north are larger.
    We use it for every storm so all years are treated the same way.
    """
    return 46.4 * np.exp(-0.0155 * vmax_ms + 0.0169 * lat)


def rankine_wind_ms(vmax_ms, rmax_km, dist_km):
    """Wind at a distance from the eye (modified Rankine vortex):
    full Vmax inside Rmax, then fading as (Rmax / distance) ** WIND_DECAY."""
    return vmax_ms * np.minimum(1.0, rmax_km / dist_km) ** WIND_DECAY
