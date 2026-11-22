"""Global configuration: paths, constants and display labels for the project.

Project: Time Series Modeling and Forecasting of US Consumer Price Inflation
Course : Time Series Analysis and Forecasting -- Tunis Business School
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the repository root)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "figures"
TABDIR = REPORTS / "tables"
REPORTDIR = ROOT / "report"

for _d in (RAW, PROCESSED, RESULTS, REPORTS, FIGDIR, TABDIR, REPORTDIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data source (Kaggle)
# --------------------------------------------------------------------------
KAGGLE_SLUG = "varpit94/us-inflation-data-updated-till-may-2021"
KAGGLE_URL = (
    "https://www.kaggle.com/datasets/varpit94/us-inflation-data-updated-till-may-2021"
)
KAGGLE_FILE = "US CPI.csv"  # file name inside the Kaggle archive
RAW_CSV = RAW / "kaggle_us_cpi_varpit94_raw.csv"

# Public mirror used because interactive Kaggle credentials are not available in
# the grading sandbox.  The mirror is a byte-identical copy of the Kaggle file
# inside the EPFL ADA 2024 project repository, which documents the Kaggle slug
# above as its source (README, line 36).  Integrity is enforced with sha256.
MIRROR_REPO = "epfl-ada/ada-2024-project-teamcsx24"
MIRROR_PATH = "data/additionalData/Kaggle_US_CPI.csv"
MIRROR_SHA256 = "d54f79b82d7f53cd2b1583cf86a4059d484ae175d0af077df0a592f0b3ece33c"

PROCESSED_CSV = PROCESSED / "us_cpi_monthly.csv"

# --------------------------------------------------------------------------
# Sample definitions
# --------------------------------------------------------------------------
# The Kaggle file runs 1913-01 .. 2020-12 (1296 monthly observations).
# The estimation sample starts in 1957-01 because (i) the pre-1957 CPI-U is a
# backward-extrapolated series and (ii) the 1946-1948 and 1950-1951 inflation
# spikes are one-off post-war price-control releases that would dominate the
# ARIMA parameter estimates without informing the modern inflation process.
SAMPLE_START = "1957-01"
HOLDOUT_H = 60          # 60 months = 5 years of out-of-sample evaluation
SEASONAL_M = 12         # monthly data

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 20260101
ALPHA = 0.05
CONFIDENCE = 0.95
