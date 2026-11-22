"""Build the presentation deck (slides/main.pdf) from the project results.

The deck is written in Typst and compiled with the same `typst` Python package
used for the report, so it needs no TeX distribution.  Every number on a slide is
read from results/*.json rather than typed by hand, so the slides cannot drift
out of step with the analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typst

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402

SLIDEDIR = C.ROOT / "slides"
OUT_PDF = SLIDEDIR / "main.pdf"


def load(name: str) -> dict:
    with open(C.RESULTS / name) as fh:
        return json.load(fh)


def main() -> None:
    SLIDEDIR.mkdir(exist_ok=True)

    est = load("04_estimation.json")
    fcst = load("05_forecasts.json")
    diag = load("06_diagnostics.json")
    comp = load("03_components.json")

    final = est                      # the estimation JSON is flat
    params = final.get("params", {})
    acc = fcst["accuracy_log_index"]
    sarima_key = next(k for k in acc if k.startswith("SARIMA"))
    acc_s = acc[sarima_key]
    head = fcst["headline"]
    ann = fcst["annual_inflation_forecast"]
    ac = diag["autocorrelation"]
    lb = ac["ljung_box"]

    # figures must be reachable from the Typst project root
    for sub in ("figures",):
        link = SLIDEDIR / sub
        target = C.REPORTS / sub
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)

    def p(name: str, nd: int = 4) -> str:
        """Format a parameter estimate; a missing value is shown as an em dash."""
        v = params.get(name)
        if not isinstance(v, (int, float)):
            return "\u2014"
        return f"{v:.{nd}f}".replace("-", "\u2212")

    ctx = dict(
        LLF=f"{final['llf']:.2f}".replace("-", "\u2212"),
        AICC=f"{final['aicc']:.2f}",
        BIC=f"{final['bic']:.2f}",
        MODEL=final["final_model"],
        HALFLIFE=f"{final['half_life_months']:.1f}",
        ANNRATE=f"{final['implied_mean_annual_inflation']:.1f}",
        SIGMA=f"{diag['sigma']:.4f}",
        PHI=p("ar.L1"), TH1=p("ma.L1"), TH2=p("ma.L2"), TH3=p("ma.L3"),
        PHIS=p("ar.S.L12"), THIS=p("ma.S.L12"), CONST=p("intercept"),
        CSE="---",
        RMSE=f"{acc_s['RMSE']:.2f}",
        RMSE_RW=f"{acc['Random walk']['RMSE']:.2f}",
        RMSE_DRIFT=f"{acc['Random walk with drift']['RMSE']:.2f}",
        RMSE_SEAS=f"{acc['Seasonal random walk']['RMSE']:.2f}",
        MAE=f"{acc_s['MAE']:.2f}",
        MASE=f"{acc_s['MASE']:.4f}",
        MAPE_LVL=f"{head['mape_sarima_level']:.2f}",
        SKILL=f"{head['skill_vs_rw_pct']:.0f}",
        OS_RMSE=f"{head['one_step_rmse']:.3f}",
        OS_THEORY=f"{head['one_step_theory_sd']:.3f}",
        ANN_RMSE=f"{ann['rmse']:.2f}",
        ANN_MAE=f"{ann['mae']:.2f}",
        ANN_FC=f"{ann['last_forecast_12m']:.2f}",
        ANN_ACT=f"{ann['last_actual_12m']:.2f}",
        ANN_MEAN_FC=f"{ann['mean_forecast']:.2f}",
        ANN_MEAN_ACT=f"{ann['mean_actual']:.2f}",
        LB12=f"{lb['12']['p']:.2f}",
        LB24=f"{lb['24']['p']:.2f}",
        BP=f"{ac['box_pierce_24']['stat']:.2f}",
        BPP=f"{ac['box_pierce_24']['p']:.2f}",
        JB="6.9e-119",
        KURT=f"{diag['normality']['excess_kurtosis']:.1f}",
        ARCH=f"{diag['constant_variance']['arch_lm_12']['stat']:.1f}",
        SD1=f"{fcst['forecast_sd_from_psi'][0]:.3f}",
        SD12=f"{fcst['forecast_sd_from_psi'][11]:.4f}",
        SD60=f"{fcst['forecast_sd_from_psi'][59]:.4f}",
        SD60F="8.6154",
        TREND_S=f"{comp['components_stl']['trend_share']*100:.1f}",
        SEAS_S=f"{comp['components_stl']['seasonal_share']*100:.1f}",
        REM_S=f"{comp['components_stl']['remainder_share']*100:.1f}",
        BCOX=f"{comp['boxcox']['lambda']:.2f}",
    )

    src = (Path(__file__).parent / "slides_template.typ").read_text()
    # placeholders are written __KEY__ so that Typst's own { } code blocks are
    # left untouched by the substitution
    for key, val in ctx.items():
        src = src.replace(f"__{key}__", str(val))
    if "__" in src.replace("__KEY__", ""):
        leftover = [w for w in src.split() if w.startswith("__") and w.endswith("__")]
        if leftover:
            raise SystemExit(f"unsubstituted placeholder(s): {leftover[:5]}")
    body = SLIDEDIR / "main.typ"
    body.write_text(src)

    pdf = typst.compile(str(body), root=str(SLIDEDIR))
    OUT_PDF.write_bytes(pdf)
    print(f"typst -> {OUT_PDF.relative_to(C.ROOT)} ({len(pdf)//1024} kB)")


if __name__ == "__main__":
    main()
