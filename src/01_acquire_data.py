"""Step 1 -- Acquire and audit the Kaggle inflation dataset.

The project uses the Kaggle dataset
    "US Inflation Data (updated till May 2021)"  --  slug varpit94/us-inflation-data-updated-till-may-2021
which contains the monthly US Consumer Price Index for All Urban Consumers
(CPI-U, seasonally adjusted), series CPIAUCSL, 1913-01 .. 2020-12.

Acquisition strategy
--------------------
* If ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (or ``~/.kaggle/kaggle.json``) are
  configured, the file is pulled straight from Kaggle with the official modern
  client (``kagglehub``).
* Otherwise the byte-identical copy of the same Kaggle file that is archived in
  the EPFL ADA-2024 teaching repository is downloaded, and its sha256 digest is
  checked against ``config.MIRROR_SHA256``.  That repository documents the
  Kaggle slug as the source of the file, so provenance is preserved.

Both routes produce ``data/raw/kaggle_us_cpi_varpit94_raw.csv`` and a JSON audit
record in ``results/01_data_audit.json``.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_kagglehub() -> tuple[bool, str]:
    """Download the dataset from Kaggle if credentials are available."""
    try:
        import kagglehub  # type: ignore
    except ImportError:
        return False, "kagglehub not installed"
    if not (os.environ.get("KAGGLE_USERNAME") or Path.home().joinpath(".kaggle/kaggle.json").exists()):
        return False, "no Kaggle credentials in the environment"
    try:
        path = Path(kagglehub.dataset_download(C.KAGGLE_SLUG))
    except Exception as exc:  # pragma: no cover - network dependent
        return False, f"kagglehub failed: {exc}"
    csvs = list(path.rglob("*.csv"))
    if not csvs:
        return False, "no csv in Kaggle archive"
    # prefer the documented file name
    csvs.sort(key=lambda p: (p.name.lower() != C.KAGGLE_FILE.lower().replace(" ", "_"), p.name))
    shutil.copyfile(csvs[0], C.RAW_CSV)
    return True, f"kagglehub -> {csvs[0]}"


def _download_mirror() -> str:
    """Fetch the archived Kaggle file from its public git mirror (blobless clone)."""
    import subprocess
    import tempfile

    url = f"https://github.com/{C.MIRROR_REPO}.git"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(tmp / "r")],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "--no-cone", "/" + C.MIRROR_PATH],
            cwd=tmp / "r", check=True, capture_output=True,
        )
        src = tmp / "r" / C.MIRROR_PATH
        shutil.copyfile(src, C.RAW_CSV)
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp / "r",
                             capture_output=True, text=True, check=True).stdout.strip()
    return f"git mirror {C.MIRROR_REPO}@{rev}:{C.MIRROR_PATH}"


# --------------------------------------------------------------------------
# main acquisition
# --------------------------------------------------------------------------
def acquire(force: bool = False) -> dict:
    audit: dict = {
        "kaggle_dataset": C.KAGGLE_SLUG,
        "kaggle_url": C.KAGGLE_URL,
        "file_in_archive": C.KAGGLE_FILE,
        "raw_path": str(C.RAW_CSV.relative_to(C.ROOT)),
    }

    if C.RAW_CSV.exists() and not force:
        audit["acquisition"] = "cached local copy"
    else:
        ok, msg = _try_kagglehub()
        if ok:
            audit["acquisition"] = f"kagglehub ({msg})"
        else:
            audit["kagglehub_note"] = msg
            audit["acquisition"] = _download_mirror()

    digest = sha256_of(C.RAW_CSV)
    audit["sha256"] = digest
    audit["sha256_verified_against_mirror"] = digest == C.MIRROR_SHA256
    if digest != C.MIRROR_SHA256:
        raise RuntimeError(
            "Checksum mismatch on the raw CPI file: the downloaded data are not the "
            f"expected Kaggle extract (got {digest})."
        )

    # ---- structural audit of the raw file -------------------------------
    raw = pd.read_csv(C.RAW_CSV)
    audit["raw_columns"] = list(raw.columns)
    audit["raw_rows"] = int(len(raw))
    audit["raw_head"] = raw.head(3).to_dict(orient="records")
    audit["raw_tail"] = raw.tail(3).to_dict(orient="records")

    # The Kaggle file stores dates as dd-mm-yyyy (e.g. 01-02-1913 = Feb 1913).
    dates = pd.to_datetime(raw["Yearmon"], format="%d-%m-%Y")
    audit["n_missing_dates"] = int(dates.isna().sum())
    audit["n_missing_cpi"] = int(raw["CPI"].isna().sum())
    audit["n_duplicate_months"] = int(dates.duplicated().sum())
    gaps = dates.diff().dt.days.dropna()
    audit["max_day_gap"] = int(gaps.max())
    audit["min_day_gap"] = int(gaps.min())

    # verify the inside-Kaggle signature: `Yearmon` + `CPI`, monthly, starts 1913-01
    audit["is_monthly"] = bool(dates.dt.day.nunique() == 1)
    audit["first_month"] = str(dates.min().date())
    audit["last_month"] = str(dates.max().date())
    audit["n_months"] = int(dates.nunique())

    C.RESULTS.joinpath("01_data_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    import pprint

    pprint.pprint(acquire(force="--force" in os.sys.argv))
