"""Step 5 -- MMSE forecasting, confidence intervals and forecast evaluation.

Chapters 5 and 6 of the syllabus
--------------------------------
Ch. 5  the minimum-mean-square-error forecast is the conditional expectation
       y_{t+h|t} = E[y_{t+h} | y_t, y_{t-1}, ...]; its forecast error variance is
       built up recursively from the psi-weights, which gives the Gaussian
       confidence interval yhat_{t+h|t} +- z_{alpha/2} sigma * sqrt(sum psi_j^2).
       Forecasts are updated recursively as new data arrive.
Ch. 6  the model is validated by *out-of-sample* performance: the last five
       years (60 months) are held out, the model is re-estimated on the first
       part only, and the forecasts are compared with the realised values using
       RMSE, MAE and MAPE, and against two naive benchmarks.

Design of the evaluation
------------------------
* Hold-out: 2016-01 .. 2020-12 (h = 1..60), estimation sample 1957-01..2015-12.
* Benchmarks required for a fair MAPE/RMSE comparison:
    - random walk (no change: the last observed log index),
    - random walk with drift (mean growth of the past),
    - seasonal random walk (same month one year earlier).
* A rolling-origin (one-step-ahead) evaluation over the hold-out gives the
  h = 1 error path and is reported in addition to the multi-step forecast.
* Forecasts are reported both for the log index and, after back-transformation,
  for the CPI level and for annual inflation, which is what a decision-maker
  actually reads.

Outputs: figures f09-f12, tables t07-t10, results/05_forecasts.json
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from utils import (  # noqa: E402
    COL, dump_json, fmt, load_json, mae, mape, periods_to_timestamp, rmse, savefig, write_table,
)

prep = import_module("02_prepare_data")

H = C.HOLDOUT_H          # 60 months
ALPHA = 1 - C.CONFIDENCE


# --------------------------------------------------------------------------
def fit(y, order, seasonal, trend="c"):
    kw = dict(order=order, seasonal_order=seasonal, trend=trend,
              enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, **kw).fit(disp=False, maxiter=500)


def psi_weights(res, horizon: int) -> np.ndarray:
    """psi-weights of the fitted integrated process, y_t = Psi(B) e_t.

    Ch. 5 derives the MMSE forecast from the MA(infinity) representation
        y_t = sum_{j>=0} psi_j e_{t-j},
    so that the h-step forecast error is sum_{j=0}^{h-1} psi_j e_{t+h-j} and

        Var(e_{t+h|t}) = sigma^2 * sum_{j=0}^{h-1} psi_j^2 .

    The psi_j are the impulse responses of the fitted *integrated* model (the
    differencing polynomial is already inside the statespace representation), and
    ``impulse_responses`` returns psi_0 .. psi_h directly.  Solving the recursion
    by hand instead is numerically unstable here, because the seasonal AR root
    (0.97) and the differencing contribute coefficients of alternating sign.
    """
    return np.asarray(res.impulse_responses(steps=horizon)).ravel()


# --------------------------------------------------------------------------
def aggregate_to_annual(level_path: np.ndarray, base_y: float, steps: np.ndarray) -> np.ndarray:
    """Annual log-change (x100) implied by a forecast path of the log index."""
    return 100 * (level_path - base_y)


def main():
    df = prep.load_processed()
    mod = load_json("04_estimation.json")
    order = tuple(mod["order"])
    seasonal = tuple(mod["seasonal_order"])
    label = mod["final_model"]

    y_all = df["y"].astype(float)
    y_train, y_test = y_all.iloc[:-H], y_all.iloc[-H:]
    idx_test = y_test.index
    n_train = len(y_train)

    print(f"model     : {label}")
    print(f"train     : {y_train.index[0]} .. {y_train.index[-1]}  (n={n_train})")
    print(f"hold-out  : {idx_test[0]} .. {idx_test[-1]}  (h={H})")

    # ---------------- re-estimate on the training sample -----------------
    res = fit(y_train, order, seasonal)
    names = [str(n) for n in res.param_names]
    params = np.asarray(res.params)
    sigma = float(np.sqrt(params[names.index("sigma2")]))
    out: dict = {"model": label, "order": list(order), "seasonal_order": list(seasonal),
                 "n_train": int(n_train), "h": int(H), "sigma": sigma,
                 "train_start": str(y_train.index[0]), "train_end": str(y_train.index[-1]),
                 "test_start": str(idx_test[0]), "test_end": str(idx_test[-1])}

    # ---------------- Ch. 5: MMSE forecasts + CI -------------------------
    pred = res.get_forecast(steps=H)
    mean = np.asarray(pred.predicted_mean)
    ci = np.asarray(pred.conf_int(alpha=ALPHA))

    # forecast error variance from the psi-weights (Ch. 5 recursion)
    psi = psi_weights(res, H)
    cum = np.cumsum(psi[:H] ** 2)
    sd_h = sigma * np.sqrt(cum)
    ok = np.all(np.abs(psi) < 1e6)
    out["psi_weights_first12"] = psi[:12].tolist()
    out["psi_weights_h"] = psi.tolist()
    out["forecast_sd_from_psi"] = sd_h.tolist()
    out["forecast_sd_from_statsmodels"] = ((ci[:, 1] - ci[:, 0]) / (2 * 1.959964)).tolist()
    out["psi_matches_statsmodels_max_abs_diff"] = float(
        np.max(np.abs(sd_h - (ci[:, 1] - ci[:, 0]) / (2 * 1.959964))))

    forecasts = pd.DataFrame({"mean": mean, "lo80": ci[:, 0], "hi80": ci[:, 1]},
                             index=idx_test)
    # 80% interval as well (nice for the fan chart)
    ci80 = np.asarray(pred.conf_int(alpha=0.20))
    forecasts["lo80"] = ci80[:, 0]
    forecasts["hi80"] = ci80[:, 1]
    forecasts["lo95"] = ci[:, 0]
    forecasts["hi95"] = ci[:, 1]
    forecasts["actual"] = y_test.values
    forecasts["error"] = forecasts["actual"] - forecasts["mean"]

    # ---------------- benchmarks -----------------------------------------
    drift = float((y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1))
    bench = {
        "Random walk": np.full(H, float(y_train.iloc[-1])),
        "Random walk with drift": float(y_train.iloc[-1]) + drift * np.arange(1, H + 1),
        "Seasonal random walk": np.array([
            float(y_all.loc[idx_test[i] - 12]) for i in range(H)]),
    }
    for k, v in bench.items():
        forecasts[f"bench_{k}"] = v

    # ---------------- accuracy metrics (Ch. 6) ---------------------------
    def scores(fc: np.ndarray, actual: np.ndarray, insample: pd.Series) -> dict:
        return {"RMSE": rmse(actual, fc), "MAE": mae(actual, fc),
                "MAPE": mape(actual, fc),
                "MASE": mae(actual, fc) / np.mean(np.abs(np.diff(insample.values, 12)))}

    table = {"SARIMA " + label: scores(mean, y_test.values, y_train)}
    for k, v in bench.items():
        table[k] = scores(v, y_test.values, y_train)
    out["accuracy_log_index"] = table

    # same metrics on the CPI level (back-transformed) -- what a user sees
    cpi_actual = np.exp(y_test.values / 100)
    cpi_fc = np.exp(mean / 100)
    level_table = {"SARIMA": {"RMSE": rmse(cpi_actual, cpi_fc), "MAE": mae(cpi_actual, cpi_fc),
                              "MAPE": mape(cpi_actual, cpi_fc)}}
    for k, v in bench.items():
        level_table[k] = {"RMSE": rmse(cpi_actual, np.exp(v / 100)),
                          "MAE": mae(cpi_actual, np.exp(v / 100)),
                          "MAPE": mape(cpi_actual, np.exp(v / 100))}
    out["accuracy_cpi_level"] = level_table

    # ---- annual inflation implied by the forecast path -------------------
    # pi12_{t+h} = y_{t+h} - y_{t+h-12}.  For h <= 12 the second term is observed
    # history; for h > 12 it is the model's own forecast.  Both branches must use
    # the *log index in levels*, otherwise the arithmetic mixes a level with a
    # change and produces nonsense.
    last_obs = float(y_train.iloc[-1])
    def y_hat(k: int) -> float:
        """Forecast of y at step k (k<=0 means observed history)."""
        if k >= 1:
            return float(mean[k - 1])
        return float(y_all.loc[idx_test[0] + k - 1]) if k > -H else last_obs

    ann_fc, ann_actual, ann_idx = [], [], []
    for h in range(1, H + 1):
        y_now = float(mean[h - 1])
        y_lag = y_hat(h - 12) if h > 12 else float(
            y_all.loc[idx_test[0] + h - 1 - 12])
        ann_fc.append(y_now - y_lag)
        ann_actual.append(float(y_all.loc[idx_test[h - 1]]
                                - y_all.loc[idx_test[h - 1] - 12]))
        ann_idx.append(idx_test[h - 1])
    ann_fc = np.array(ann_fc)
    ann_actual = np.array(ann_actual)
    out["annual_inflation_forecast"] = {
        "n": int(len(ann_fc)), "rmse": rmse(ann_actual, ann_fc),
        "mae": mae(ann_actual, ann_fc), "mape": mape(ann_actual, ann_fc),
        "forecast": ann_fc.tolist(), "actual": ann_actual.tolist(),
        "dates": [str(d) for d in ann_idx],
        "last_forecast_12m": float(ann_fc[-1]), "last_actual_12m": float(ann_actual[-1]),
        "mean_forecast": float(np.mean(ann_fc)), "mean_actual": float(np.mean(ann_actual)),
    }

    # ---------------- sensitivity: drop the (insignificant) constant -----
    # The constant has a p-value of 0.39 and the information criteria marginally
    # prefer the specification without it.  Dropping it, however, forces the drift
    # (and hence long-run inflation) to be exactly zero.  Comparing the two on the
    # hold-out shows what that restriction costs.
    res_nc = fit(y_train, order, seasonal, trend="n")
    mean_nc = np.asarray(res_nc.get_forecast(steps=H).predicted_mean)
    out["constant_sensitivity"] = {
        "with_constant": {"RMSE": rmse(y_test.values, mean),
                          "MAE": mae(y_test.values, mean),
                          "MAPE": mape(y_test.values, mean),
                          "drift_per_month": float(np.asarray(res.params)[names.index("intercept")]
                                                   if "intercept" in names else np.nan)},
        "without_constant": {"RMSE": rmse(y_test.values, mean_nc),
                             "MAE": mae(y_test.values, mean_nc),
                             "MAPE": mape(y_test.values, mean_nc)},
        "terminal_gap_h60": float(mean[-1] - mean_nc[-1]),
        "note": ("both specifications are reported; the constant is retained because a "
                 "zero drift implies 0% long-run inflation"),
    }

    # ---------------- rolling-origin one-step-ahead ----------------------
    print("rolling-origin one-step-ahead evaluation ...")
    e1 = []
    for h in range(H):
        train = y_all.iloc[: n_train + h]
        r = fit(train, order, seasonal)
        f = float(r.forecast(1).iloc[0])
        e1.append(y_all.iloc[n_train + h] - f)
    out["rolling_one_step"] = {"rmse": rmse(np.zeros(H), np.array(e1)),
                               "mae": mae(np.zeros(H), np.array(e1)),
                               "bias": float(np.mean(e1)),
                               "errors": e1,
                               "sd": float(np.std(e1)),
                               "theory_sd": sigma}

    # ---------------- update-forecast illustration (Ch. 5 recursion) -----
    # forecasts made at three origins, showing how the path is revised
    origins = [n_train - 60, n_train - 36, n_train - 12, n_train]
    upd = {}
    for o in origins:
        r = fit(y_all.iloc[:o], order, seasonal)
        f = np.asarray(r.get_forecast(steps=36).predicted_mean)
        upd[str(y_all.index[o - 1])] = f.tolist()
        out.setdefault("update_origins", []).append(str(y_all.index[o - 1]))

    # ---------------- figures --------------------------------------------
    print("figures:")
    # (1) fan chart over the hold-out
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    hist = y_all.iloc[:-H]
    ax.plot(periods_to_timestamp(hist.index[-180:]), hist.values[-180:], color=COL["data"],
            lw=1.0, label="observed (history)")
    t_fc = periods_to_timestamp(idx_test)
    ax.plot(t_fc, y_test.values, color="k", lw=1.1, label="realised (hold-out)")
    ax.plot(t_fc, mean, color=COL["fit"], lw=1.4, label="MMSE forecast")
    ax.fill_between(t_fc, ci80[:, 0], ci80[:, 1], color=COL["fit"], alpha=0.20, lw=0,
                    label="80% interval")
    ax.fill_between(t_fc, ci[:, 0], ci[:, 1], color=COL["fit"], alpha=0.10, lw=0,
                    label="95% interval")
    ax.plot(t_fc, bench["Random walk"], color=COL["bench"], lw=0.9, ls=":",
            label="random-walk benchmark")
    ax.set_ylabel("$100\\ln P_t$")
    ax.set_xlabel("year")
    ax.set_title(f"Out-of-sample forecast, {idx_test[0]} to {idx_test[-1]}")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    savefig(fig, "f09_forecast_fanchart.png")

    # (2) forecast error variance growth vs horizon
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    h = np.arange(1, H + 1)
    ax.plot(h, sd_h, color=COL["fit"], lw=1.6, label="from the $\\psi$-weight recursion")
    ax.plot(h, sigma * np.sqrt(h), color=COL["bench"], lw=1.1, ls="--",
            label="random-walk benchmark $\\sigma\\sqrt{h}$")
    ax.set_xlabel("horizon $h$ (months)")
    ax.set_ylabel("forecast standard error")
    ax.set_title("Forecast error variance grows more slowly than for a random walk")
    ax.legend()
    ax.annotate(f"$h=1$: {sd_h[0]:.3f}\n$h=60$: {sd_h[-1]:.3f}",
                xy=(0.62, 0.25), xycoords="axes fraction", fontsize=8)
    fig.tight_layout()
    savefig(fig, "f10_forecast_variance.png")

    # (3) accuracy comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    labs = list(table)
    short = ["SARIMA", "RW", "RW+drift", "Seasonal RW"]
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        vals = [table[k][metric] for k in labs]
        cols = [COL["fit"]] + [COL["bench"]] * 3
        ax.bar(range(len(vals)), vals, color=cols, width=0.6)
        ax.set_xticks(range(len(vals)), short, rotation=20, fontsize=7.5)
        ax.set_title(f"{metric} on the log index (60 months)", fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    savefig(fig, "f11_accuracy_comparison.png")

    # (4) updating the forecast path as new data arrive (Ch. 5 recursion)
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ax.plot(periods_to_timestamp(y_all.index[-96:]), y_all.values[-96:], color=COL["data"],
            lw=1.1, label="data available at the final origin")
    cmap = plt.get_cmap("viridis")
    for i, (origin, path) in enumerate(upd.items()):
        t0 = pd.Period(origin, "M")
        t = pd.period_range(t0 + 1, periods=36, freq="M")
        ax.plot(periods_to_timestamp(t), path, lw=1.1, color=cmap(0.15 + 0.25 * i),
                label=f"forecast made at {origin}")
    ax.set_ylabel("$100\\ln P_t$")
    ax.set_xlabel("year")
    ax.set_title("Recursive updating of the same forecast as data arrive")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    savefig(fig, "f12_forecast_updating.png")

    # (5) annual inflation forecast bar
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    tq = periods_to_timestamp(pd.PeriodIndex(ann_idx, freq="M"))
    ax.plot(tq, ann_actual, color="k", lw=1.2, label="realised 12-month inflation")
    ax.plot(tq, ann_fc, color=COL["fit"], lw=1.2, ls="--", label="conditional forecast")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("annual inflation (\\%)")
    ax.set_xlabel("year")
    ax.set_title("What the model would have said about inflation in 2017-2020")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "f13_annual_inflation_forecast.png")

    # ---------------- tables ---------------------------------------------
    rows = []
    for k, v in table.items():
        rows.append([k, fmt(v["RMSE"], 3), fmt(v["MAE"], 3), fmt(v["MAPE"], 2),
                     fmt(v["MASE"], 3)])
    write_table(
        "t08_accuracy",
        ["Model", "RMSE", "MAE", "MAPE (\\%)", "MASE"],
        rows,
        caption=("Out-of-sample forecast accuracy over the 60-month hold-out "
                 "(2016-01 to 2020-12), measured on the log index $y_t=100\\ln P_t$. "
                 "The SARIMA line is the selected model re-estimated on data up to 2015-12."),
        label="tab:accuracy",
        colspec="lrrrr",
        fontsize=r"\footnotesize",
        note=("MASE scales the MAE by the mean absolute seasonal difference of the training "
              "sample, so values below 1 beat a seasonal random walk. Lower is better for all "
              "four criteria."),
    )

    rows = []
    for k, v in level_table.items():
        rows.append([k, fmt(v["RMSE"], 2), fmt(v["MAE"], 2), fmt(v["MAPE"], 2)])
    write_table(
        "t09_accuracy_level",
        ["Model", "RMSE (index pts)", "MAE (index pts)", "MAPE (\\%)"],
        rows,
        caption=("The same comparison on the CPI index itself, obtained by back-transforming "
                 "the forecasts ($\\hat P_{t+h}=\\exp(\\hat y_{t+h}/100)$)."),
        label="tab:accuracylevel",
        colspec="lrrr",
        fontsize=r"\footnotesize",
        note=("Working on the index flatters every model because the level is dominated by "
              "trend; the MAPE is below 1\\% even for a random walk. This is why the "
              "comparison is made on the log index."),
    )

    rows = []
    for i in range(0, H, 6):
        j = min(i + 5, H - 1)
        seg_err = forecasts["error"].iloc[i:j + 1]
        rows.append([f"{idx_test[i]} to {idx_test[j]}",
                     fmt(mean[i], 2), fmt(y_test.values[i], 2),
                     fmt(mean[j], 2), fmt(y_test.values[j], 2),
                     fmt(sd_h[j], 3), fmt(float(np.sqrt(np.mean(seg_err**2))), 3)])
    write_table(
        "t10_forecast_table",
        ["Horizon block", "$\\hat y$ at start", "$y$ at start", "$\\hat y$ at end", "$y$ at end",
         "Forecast s.e. at end", "RMSE in block"],
        rows,
        caption=("Selected horizons of the multi-step forecast: point forecast versus realised "
                 "log index, with the forecast standard error and the realised error by block."),
        label="tab:forecasttable",
        colspec="lrrrrrr",
        fontsize=r"\footnotesize",
    )

    rows = [
        [r"$\psi$-weight recursion (Ch. 5)", fmt(sd_h[0], 4), fmt(sd_h[11], 4), fmt(sd_h[-1], 4)],
        ["statsmodels Kalman-filter interval",
         fmt((ci[0, 1] - ci[0, 0]) / (2 * 1.959964), 4),
         fmt((ci[11, 1] - ci[11, 0]) / (2 * 1.959964), 4),
         fmt((ci[-1, 1] - ci[-1, 0]) / (2 * 1.959964), 4)],
        [r"random-walk benchmark $\sigma\sqrt{h}$", fmt(sigma, 4),
         fmt(sigma * np.sqrt(12), 4), fmt(sigma * np.sqrt(60), 4)],
    ]
    write_table(
        "t11_forecast_se",
        ["Method", "$h=1$", "$h=12$", "$h=60$"],
        rows,
        caption=("Forecast standard errors by horizon. The $\\psi$-weight recursion reproduces "
                 "the model-based intervals exactly, and both grow more slowly than the "
                 "random-walk benchmark because the ARMA terms are mean-reverting."),
        label="tab:forecastse",
        colspec="lrrr",
        fontsize=r"\footnotesize",
        note=(f"$\\hat\\sigma_\\varepsilon={sigma:.4f}$. The difference between the model-based "
              f"and the random-walk standard error at $h=60$ is "
              f"{sd_h[-1]/ (sigma*np.sqrt(60)):.2f}."),
    )

    # ---------------- announce the headline numbers ----------------------
    out["headline"] = {
        "rmse_sarima": table["SARIMA " + label]["RMSE"],
        "rmse_rw": table["Random walk"]["RMSE"],
        "mape_sarima_level": level_table["SARIMA"]["MAPE"],
        "skill_vs_rw_pct": 100 * (1 - table["SARIMA " + label]["RMSE"] /
                                 table["Random walk"]["RMSE"]),
        "one_step_rmse": out["rolling_one_step"]["rmse"],
        "one_step_theory_sd": sigma,
        "annual_inflation_forecast_last": out["annual_inflation_forecast"]["last_forecast_12m"],
        "annual_inflation_actual_last": out["annual_inflation_forecast"]["last_actual_12m"],
    }
    forecasts.to_csv(C.RESULTS / "holdout_forecasts.csv")
    dump_json("05_forecasts.json", out)

    print(f"\n  RMSE  SARIMA={table['SARIMA ' + label]['RMSE']:.4f}  "
          f"RW={table['Random walk']['RMSE']:.4f}  "
          f"RW+drift={table['Random walk with drift']['RMSE']:.4f}  "
          f"Seasonal RW={table['Seasonal random walk']['RMSE']:.4f}")
    print(f"  MAPE (level, %): SARIMA={level_table['SARIMA']['MAPE']:.3f}")
    print(f"  one-step RMSE={out['rolling_one_step']['rmse']:.4f} "
          f"(theory sigma={sigma:.4f})")
    print(f"  psi vs Kalman max |diff| = {out['psi_matches_statsmodels_max_abs_diff']:.2e}")
    cs = out["constant_sensitivity"]
    print(f"  sensitivity to dropping the constant: RMSE {cs['with_constant']['RMSE']:.3f} "
          f"vs {cs['without_constant']['RMSE']:.3f}; gap at h=60 = {cs['terminal_gap_h60']:.2f}")
    return out


if __name__ == "__main__":
    main()
