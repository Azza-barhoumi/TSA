"""Step 2 -- Clean the CPI series and build the working variables.

Transformations used (syllabus Ch. 2: difference and backshift operators,
variance-stabilising transformations)
--------------------------------------------------------------------------
P_t     CPI-U index (1982-84 = 100)
y_t     100 * ln(P_t)                        -- log transform stabilises the
                                                growing variance of a price index
pi_t    y_t - y_(t-1)  = 100 * d ln P_t       -- monthly inflation, %
pi12_t  y_t - y_(t-12) = 100 * d12 ln P_t     -- annual inflation, %

The log transform is the variance-stabilising choice here; the Box-Cox profile
likelihood that justifies it is computed in step 3 (Ch. 2 also covers Box-Cox).

Sample: the Kaggle file covers 1913-01..2020-12.  The estimation sample starts
in 1957-01 because the pre-1957 CPI-U is a backward extrapolation and the
1946-51 observations are one-off post-war price-control releases.

Outputs: data/processed/us_cpi_monthly.csv, results/02_data_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def load_raw() -> pd.DataFrame:
    """Read the Kaggle file and return a clean monthly CPI series."""
    raw = pd.read_csv(C.RAW_CSV)
    # the Kaggle `Yearmon` field is dd-mm-yyyy (01-02-1913 = February 1913)
    dates = pd.to_datetime(raw["Yearmon"], format="%d-%m-%Y")
    df = pd.DataFrame({"date": dates, "cpi": pd.to_numeric(raw["CPI"], errors="coerce")})
    df = df.dropna(subset=["date", "cpi"]).drop_duplicates("date").sort_values("date")
    df = df.set_index(pd.PeriodIndex(df["date"], freq="M")).drop(columns="date")
    return df


def build(verbose: bool = True) -> pd.DataFrame:
    full = load_raw()

    # ---- completeness check on the whole file ---------------------------
    expected = pd.period_range(full.index.min(), full.index.max(), freq="M")
    missing = expected.difference(full.index)
    assert len(missing) == 0, f"missing months in the raw file: {missing}"

    audit: dict = {
        "full_sample": {"start": str(full.index.min()), "end": str(full.index.max()),
                        "n_obs": int(len(full))},
        "missing_months_full_sample": int(len(missing)),
    }

    # ---- estimation sample ----------------------------------------------
    df = full.loc[pd.Period(C.SAMPLE_START, "M"):].copy()
    df["calendar_month"] = df.index.month
    df["year"] = df.index.year
    df["y"] = 100 * np.log(df["cpi"])      # log index, scaled by 100
    df["pi"] = df["y"].diff()              # monthly log-change (%)
    df["pi12"] = df["y"].diff(12)          # annual log-change (%)

    audit["estimation_sample"] = {
        "start": str(df.index.min()), "end": str(df.index.max()), "n_obs": int(len(df)),
        "excluded_obs": int(len(full) - len(df)),
        "why_start": ("pre-1957 CPI-U is a backward extrapolation; the 1946-51 "
                      "observations are one-off post-war price-control releases"),
    }

    # ---- descriptive statistics -----------------------------------------
    desc = {}
    for col, label in [("cpi", "CPI level P_t"), ("y", "Log index y_t = 100 ln P_t"),
                       ("pi", "Monthly inflation pi_t"), ("pi12", "Annual inflation pi12_t")]:
        s = df[col].dropna()
        desc[label] = {
            "n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std()), "min": float(s.min()), "max": float(s.max()),
            "skew": float(s.skew()), "excess_kurtosis": float(s.kurtosis()),
        }
    audit["descriptives"] = desc

    # ---- inflation by decade --------------------------------------------
    dec = df.groupby((df.index.year // 10) * 10)["pi12"].agg(["mean", "std", "min", "max", "count"])
    audit["annual_inflation_by_decade"] = {
        f"{int(k)}s": {kk: (None if pd.isna(v) else float(v)) for kk, v in row.items()}
        for k, row in dec.iterrows()
    }

    # ---- calendar-month profile (sanity check: the series is SA) --------
    by_month = df.groupby("calendar_month")["pi"].agg(["mean", "std", "count"])
    audit["monthly_profile_of_pi"] = {int(k): {kk: float(v) for kk, v in row.items()}
                                      for k, row in by_month.iterrows()}

    # ---- data quality flags ---------------------------------------------
    z = (df["pi"] - df["pi"].mean()) / df["pi"].std()
    audit["large_monthly_moves_abs_z_gt3"] = {
        str(k): {"pi": float(df.loc[k, "pi"]), "z": float(z.loc[k])}
        for k in z[z.abs() > 3].index
    }
    audit["n_nonpositive_cpi"] = int((df["cpi"] <= 0).sum())

    # ---- save ------------------------------------------------------------
    out = df[["cpi", "y", "pi", "pi12", "calendar_month", "year"]].copy()
    out.index = out.index.astype(str)
    out.index.name = "period"
    out.to_csv(C.PROCESSED_CSV, index=True)
    audit["processed_path"] = str(C.PROCESSED_CSV.relative_to(C.ROOT))
    C.RESULTS.joinpath("02_data_audit.json").write_text(json.dumps(audit, indent=2))

    if verbose:
        print(f"processed -> {C.PROCESSED_CSV.relative_to(C.ROOT)}  ({len(df)} rows)")
        print(pd.DataFrame(desc).T[["n", "mean", "std", "min", "max"]].round(3))
    return df


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(C.PROCESSED_CSV)
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    df["calendar_month"] = df["calendar_month"].astype(int)
    return df.set_index("period")


if __name__ == "__main__":
    build()
