"""Run the whole pipeline in order and build the report.

    python run_all.py            # everything, including the PDF
    python run_all.py --no-pdf   # analysis only
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(Path(sys.executable))

STAGES = [
    ("01_acquire_data.py", "Acquire and audit the Kaggle dataset"),
    ("02_prepare_data.py", "Clean, transform, descriptive statistics"),
    ("03_components_stationarity.py", "Components, Box-Cox, ACF/PACF, stationarity"),
    ("04_identify_estimate.py", "Identification and estimation (MoM / CSS / MLE)"),
    ("05_forecasting.py", "MMSE forecasts, intervals, hold-out evaluation"),
    ("06_diagnostics.py", "Residual diagnostics and parsimony checks"),
]


def run(script: str, label: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {script}  --  {label}")
    print("=" * 78, flush=True)
    t0 = time.time()
    p = subprocess.run([PY, "-W", "ignore", str(ROOT / "src" / script)], cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit(f"stage {script} failed with exit code {p.returncode}")
    print(f"  done in {time.time() - t0:.1f}s")


def main() -> None:
    print(f"python: {PY}")
    for script, label in STAGES:
        run(script, label)
    if "--no-pdf" not in sys.argv:
        run("07_build_report.py", "Compile the LaTeX report to PDF")
    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
