# Time Series Analysis and Forecasting of US Consumer Price Inflation

A complete Box–Jenkins analysis of US CPI-U inflation, submitted as the term project
of *Time Series Analysis and Forecasting* (Tunis Business School).

**The deliverables are:**
- [`report/main.pdf`](report/main.pdf) — a 34-page typeset report (source: [`report/main.tex`](report/main.tex))
- [`slides/main.pdf`](slides/main.pdf) — a 19-slide presentation deck

## The question

> Given the history of US consumer prices to December 2020, what inflation rate should
> a Tunisian bank, treasurer or importer budget for over the next one to five years,
> and how wide is the interval within which the realised rate should fall?

## The answer, in one paragraph

A multiplicative **ARIMA(1,1,3)(1,0,1)[12] with a constant** is fitted to the log CPI
over 1957-01 to 2020-12 (*n* = 768). It implies long-run annual inflation of **2.9%**,
a half-life for inflation shocks of **16.5 months**, and a 95% forecast band of
**±1.8 points at one year** widening to **±8.6 points at five**. On a 60-month hold-out
(2016–2020) it attains an RMSE of **1.79** on the log index against **6.23** for a random
walk — a **71% reduction** in forecast error.

## Data

| | |
|---|---|
| Source | [Kaggle: *US Inflation Data*](https://www.kaggle.com/datasets/varpit94/us-inflation-data-updated-till-may-2021) (CPI-U, US city average, all items) |
| Coverage | 1,296 monthly observations, 1913-01 to 2020-12, no gaps |
| Estimation sample | 1957-01 to 2020-12 (*n* = 768) |
| Hold-out | 2016-01 to 2020-12 (*n* = 60) |
| Integrity | SHA-256 of the raw CSV verified by `src/01_acquire_data.py` |

## Reproducing everything

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py
```

`run_all.py` runs the eight stages in order (~9 minutes) and rebuilds both
`report/main.pdf` and `slides/main.pdf`. Use `--no-pdf` to skip both.

## Pipeline

| Stage | Script | Produces |
|---|---|---|
| 1 | `src/01_acquire_data.py` | download + SHA-256 audit → `results/01_data_audit.json` |
| 2 | `src/02_prepare_data.py` | log index, Δ, Δ₁₂, descriptive statistics |
| 3 | `src/03_components_stationarity.py` | STL, Box–Cox, ACF/PACF, ADF/KPSS |
| 4 | `src/04_identify_estimate.py` | 63-model SARIMA grid, MoM/CSS/HR/MLE, final model |
| 5 | `src/05_forecasting.py` | MMSE forecasts, ψ-weight intervals, hold-out accuracy |
| 6 | `src/06_diagnostics.py` | residual diagnostics, parsimony, sensitivity |
| 7 | `src/07_build_report.py` | compile `report/main.tex` → `report/main.pdf` |
| 8 | `src/08_build_slides.py` | build `slides/main.pdf` (numbers injected from `results/`) |

## Scope

The analysis stays inside Chapters 1–7 of Cryer & Chan (2008): components, stationarity,
ACF/PACF and the Yule–Walker equations, backshift and difference operators, Box–Cox,
AR/MA/ARMA/ARIMA models, the three estimation methods, MMSE forecasting with intervals,
the three-stage Box–Jenkins cycle, and multiplicative seasonal SARMA. No out-of-scope
econometrics is used; ARCH effects and non-normality are *reported as limitations*, not
modelled.

## Layout

```
data/raw/            the Kaggle CSV, untouched
data/processed/      the cleaned analysis series
src/                 one standalone script per stage
results/             every number the report cites, as JSON
reports/figures/     18 figures (PDF/PNG)
reports/tables/      13 booktabs tables
report/              LaTeX source + compiled PDF
slides/              presentation deck (Typst) + compiled PDF
```

## Note on the PDF build

No TeX distribution is available in this environment, so `src/07_build_report.py`
compiles the *same* LaTeX source with **pandoc → Typst**. If `latexmk`, `pdflatex` or
`tectonic` is installed, the script detects it and uses it instead, producing the PDF
exactly as LaTeX would.
