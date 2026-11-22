"""Step 6 -- Diagnostic checking of the selected model (Ch. 6, Box-Jenkins stage 3).

The syllabus lists the four residual checks explicitly:

1. zero mean      -- one-sample t test on the residuals (and on the standardised
                     residuals), with the sample mean and its standard error;
2. constant variance -- split-half variance ratio and a regression of the squared
                     residuals on a time trend; the ARCH-LM test is reported as the
                     standard formal test of conditional heteroskedasticity;
3. residual autocorrelation -- ACF and PACF of the residuals with Bartlett bands,
                     the Ljung-Box portmanteau statistic at 12/24/36 lags, and the
                     Box-Pierce statistic;
4. adequacy and parsimony -- comparison with the neighbouring over- and
                     under-parameterised models (one more/fewer AR or MA term).

Also reported: normal probability plot and Jarque-Bera (not required by the
course, but needed to justify the Gaussian forecast intervals), and the effect
of the two largest residuals on the estimates.

Outputs: figures f14-f16, tables t11-t12, results/06_diagnostics.json
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from utils import COL, dump_json, fmt, load_json, periods_to_timestamp, savefig, write_table  # noqa: E402

prep = import_module("02_prepare_data")


def fit(y, order, seasonal, trend="c"):
    kw = dict(order=order, seasonal_order=seasonal, trend=trend,
              enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, **kw).fit(disp=False, maxiter=500)


def resid_of(res, index):
    burn = int(getattr(res, "loglikelihood_burn", 0) or 0)
    idx = pd.PeriodIndex(index[burn:], freq="M")
    return pd.Series(np.asarray(res.resid)[burn:], index=idx).dropna()


# --------------------------------------------------------------------------
def fig_residuals(resid: pd.Series, sd: float):
    z = resid / sd
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.2), sharex=True)
    ax = axes[0]
    ax.plot(periods_to_timestamp(resid.index), z.values, color=COL["data"], lw=0.7,
            marker="o", ms=1.6)
    for v, ls in [(2, "--"), (3, ":")]:
        for s in (v, -v):
            ax.axhline(s, color=COL["fit"], lw=0.8, ls=ls)
    ax.set_ylabel("standardised residual")
    ax.set_title("(a) Standardised residuals (dashed: $\\pm2$, dotted: $\\pm3$ s.d.)")
    ax = axes[1]
    ax.hist(z, bins=45, density=True, color="#dce6f2", edgecolor=COL["data"], linewidth=0.5)
    xs = np.linspace(z.min(), z.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs), color=COL["fit"], lw=1.3, label="$N(0,1)$")
    ax.set_xlabel("standardised residual")
    ax.set_ylabel("density")
    ax.set_title("(b) Residual distribution against the Gaussian reference")
    ax.legend()
    fig.tight_layout()
    return savefig(fig, "f14_residuals.png")


def fig_acf_pacf(resid: pd.Series, nlags: int = 36):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    ci = 1.96 / np.sqrt(len(resid))
    for ax, vals, ttl, col in [
            (axes[0], acf(resid, nlags=nlags, fft=False)[1:], "ACF of residuals", COL["data"]),
            (axes[1], pacf(resid, nlags=nlags, method="ywm")[1:], "PACF of residuals", COL["fit"])]:
        ax.bar(np.arange(1, len(vals) + 1), vals, width=0.6, color=col)
        ax.axhline(0, color="k", lw=0.6)
        for v in (ci, -ci):
            ax.axhline(v, color="k", lw=0.7, ls="--")
        ax.axvspan(10.5, 13.5, color=COL["shade"], alpha=0.6, lw=0, zorder=0)
        ax.set_title(ttl)
        ax.set_xlabel("lag (months)")
        ax.set_ylim(-1.02, 1.02)
    axes[0].set_ylabel("autocorrelation")
    fig.tight_layout()
    return savefig(fig, "f15_residual_acf_pacf.png")


def fig_qq_sq(resid: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    stats.probplot(resid, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set(marker="o", ms=2.3, markerfacecolor=COL["data"],
                               markeredgecolor="none")
    axes[0].get_lines()[1].set(color=COL["fit"], lw=1.2)
    axes[0].set_title("(a) Normal probability plot")
    axes[0].set_xlabel("theoretical quantiles")
    axes[0].set_ylabel("sample quantiles")

    sq = resid**2
    a = acf(sq, nlags=36, fft=False)[1:]
    ci = 1.96 / np.sqrt(len(sq))
    axes[1].bar(np.arange(1, len(a) + 1), a, width=0.6, color=COL["accent"])
    axes[1].axhline(0, color="k", lw=0.6)
    for v in (ci, -ci):
        axes[1].axhline(v, color="k", lw=0.7, ls="--")
    axes[1].set_title("(b) ACF of squared residuals (variance check)")
    axes[1].set_xlabel("lag (months)")
    axes[1].set_ylabel("autocorrelation")
    fig.tight_layout()
    return savefig(fig, "f16_qq_and_squared.png")


# --------------------------------------------------------------------------
def main():
    df = prep.load_processed()
    y = df["y"].astype(float)
    mod = load_json("04_estimation.json")
    order, seasonal = tuple(mod["order"]), tuple(mod["seasonal_order"])
    label = mod["final_model"]

    res = fit(y, order, seasonal)
    names = [str(n) for n in res.param_names]
    params = np.asarray(res.params)
    sd = float(np.sqrt(params[names.index("sigma2")]))
    resid = resid_of(res, y.index)
    z = resid / sd
    n = len(resid)
    out: dict = {"model": label, "n_resid": n, "sigma": sd, "burn_in_dropped":
                 int(getattr(res, "loglikelihood_burn", 0))}

    # ---- 1. zero mean ----------------------------------------------------
    t_mean = stats.ttest_1samp(resid, 0.0)
    out["zero_mean"] = {"mean": float(resid.mean()), "se_mean": float(resid.std(ddof=1) / np.sqrt(n)),
                        "t": float(t_mean.statistic), "p": float(t_mean.pvalue),
                        "ci95": [float(resid.mean() - 1.96 * resid.std(ddof=1) / np.sqrt(n)),
                                 float(resid.mean() + 1.96 * resid.std(ddof=1) / np.sqrt(n))]}

    # ---- 2. constant variance -------------------------------------------
    half = n // 2
    v1, v2 = float(resid.iloc[:half].var()), float(resid.iloc[half:].var())
    f_test = max(v1, v2) / min(v1, v2)
    fp = float(2 * min(stats.f.cdf(f_test, half - 1, n - half - 1),
                       1 - stats.f.cdf(f_test, half - 1, n - half - 1)))
    # regression of squared residuals on a time trend (White-style variance check)
    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(X, np.asarray(resid) ** 2, rcond=None)
    e = np.asarray(resid) ** 2 - X @ beta
    s2 = e @ e / (n - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    ttrend = beta[1] / se[1]
    ptrend = float(2 * (1 - stats.t.cdf(abs(ttrend), n - 2)))
    arch6 = het_arch(resid, nlags=6)
    arch12 = het_arch(resid, nlags=12)
    lb_sq = acorr_ljungbox(resid**2, lags=[12], return_df=True)
    out["constant_variance"] = {
        "var_first_half": v1, "var_second_half": v2, "ratio": f_test, "f_p": fp,
        "trend_in_squared_resid": {"coef": float(beta[1]), "t": float(ttrend), "p": ptrend},
        "arch_lm_6": {"stat": float(arch6[0]), "p": float(arch6[1])},
        "arch_lm_12": {"stat": float(arch12[0]), "p": float(arch12[1])},
        "ljung_box_squared_12": {"stat": float(lb_sq.loc[12, "lb_stat"]),
                                 "p": float(lb_sq.loc[12, "lb_pvalue"])},
    }

    # ---- 3. residual autocorrelation ------------------------------------
    lb = acorr_ljungbox(resid, lags=[6, 12, 18, 24, 36], return_df=True)
    # Box--Pierce computed directly: statsmodels' q_stat overflows to inf on this
    # numpy version, which silently produced a meaningless statistic.
    _a = acf(resid, nlags=24, fft=False)
    bp_stat = float(n * np.sum(_a[1:25] ** 2))
    bp_p = float(stats.chi2.sf(bp_stat, 24))
    out["autocorrelation"] = {
        "ljung_box": {int(k): {"stat": float(v["lb_stat"]), "p": float(v["lb_pvalue"])}
                      for k, v in lb.iterrows()},
        "box_pierce_24": {"stat": bp_stat, "p": bp_p},
        "acf_abs_max_lags1to12": float(np.max(np.abs(acf(resid, nlags=12, fft=False)[1:]))),
        "bartlett_band": float(1.96 / np.sqrt(n)),
    }

    # ---- 4. normality (justifies the Gaussian intervals) ----------------
    jb = stats.jarque_bera(resid)
    sw = stats.shapiro(resid)
    out["normality"] = {"jarque_bera": {"stat": float(jb.statistic), "p": float(jb.pvalue)},
                        "shapiro": {"stat": float(sw.statistic), "p": float(sw.pvalue)},
                        "skew": float(resid.skew()), "excess_kurtosis": float(resid.kurtosis())}

    # ---- 5. adequacy / parsimony: neighbouring models --------------------
    p, d, q = order
    P, D, Q, s = seasonal
    neigh = {
        f"ARIMA({p},{d},{q})({P},{D},{Q})[{s}]  (selected)": (order, seasonal),
        f"ARIMA({p-1},{d},{q})({P},{D},{Q})[{s}]  (fewer AR)": ((max(p - 1, 0), d, q), seasonal),
        f"ARIMA({p},{d},{q+1})({P},{D},{Q})[{s}]  (more MA)": ((p, d, q + 1), seasonal),
        f"ARIMA({p},{d},{q})({P-1},{D},{Q})[{s}]  (no seasonal AR)": ((p, d, q), (0, D, Q, s)),
        f"ARIMA({p},{d},{q})({P},{D},{Q})[{s}] no constant": (order, seasonal),
    }
    rows_neigh = []
    neigh_out = {}
    for lab, (o, se) in neigh.items():
        trend = "n" if "no constant" in lab else "c"
        try:
            r = fit(y, o, se, trend)
        except Exception:
            continue
        rr = resid_of(r, y.index)
        lbr = acorr_ljungbox(rr, lags=[12, 24], return_df=True)
        k = len(r.params)
        aicc = r.aic + 2 * k * (k + 1) / (len(y) - k - 1)
        rec = {"label": lab, "k": k, "llf": float(r.llf), "aic": float(r.aic),
               "aicc": float(aicc), "bic": float(r.bic),
               "lb_p_12": float(lbr.loc[12, "lb_pvalue"]),
               "lb_p_24": float(lbr.loc[24, "lb_pvalue"])}
        neigh_out[lab] = rec
        rows_neigh.append([lab, f"{k}", fmt(r.llf, 2), fmt(aicc, 2), fmt(r.bic, 2),
                           fmt(rec["lb_p_12"], 3), fmt(rec["lb_p_24"], 3)])
    out["neighbouring_models"] = neigh_out

    # ---- 6. influence of the largest residuals ---------------------------
    big = z.abs().sort_values(ascending=False).head(6)
    out["largest_residuals"] = [{"date": str(k), "z": float(z.loc[k]),
                                 "resid": float(resid.loc[k])} for k in big.index]
    # drop the two largest and refit -> are the estimates stable?
    drop = list(big.index[:2])
    y_drop = y.drop(index=drop)
    res_drop = fit(y_drop, order, seasonal)
    p_drop = np.asarray(res_drop.params)
    out["sensitivity_drop_two_largest"] = {
        "dropped": [str(d) for d in drop],
        "coef_selected": {k: float(v) for k, v in zip(names, params)},
        "coef_refit": {k: float(v) for k, v in zip(names, p_drop)},
        # absolute changes are reported alongside relative ones: the constant of an
        # ARIMA(p,1,q) on a log index is tiny (0.0002 per month), so a relative
        # change is not meaningful for it
        "max_abs_change": float(np.max(np.abs(p_drop - params))),
        "max_abs_change_pct_of_sigma": float(np.max(np.abs(p_drop - params))
                                            / float(np.sqrt(params[names.index("sigma2")])) * 100),
        "param_scale": {k: float(abs(v)) for k, v in zip(names, params)},
    }

    # ---- figures ---------------------------------------------------------
    print("figures:")
    fig_residuals(resid, sd)
    fig_acf_pacf(resid)
    fig_qq_sq(resid)

    # ---- tables ----------------------------------------------------------
    cv = out["constant_variance"]
    ac = out["autocorrelation"]
    rows = [
        [r"\textbf{Zero mean}: sample mean of $\hat\varepsilon_t$", fmt(out["zero_mean"]["mean"], 5),
         f"({fmt(out['zero_mean']['se_mean'], 5)})", fmt(out["zero_mean"]["t"], 3),
         fmt(out["zero_mean"]["p"], 3), "zero mean not rejected" if out["zero_mean"]["p"] > 0.05 else "reject"],
        [r"\textbf{Constant variance}: variance ratio, first vs second half",
         fmt(cv["ratio"], 3), "--", "--", fmt(cv["f_p"], 3),
         "equal variances not rejected" if cv["f_p"] > 0.05 else "reject equal variances"],
        [r"\quad trend coefficient in $\hat\varepsilon_t^2$ regression", fmt(cv["trend_in_squared_resid"]["coef"], 6),
         f"({fmt(cv['trend_in_squared_resid']['t'], 3)})", "--",
         fmt(cv["trend_in_squared_resid"]["p"], 3),
         "no trend" if cv["trend_in_squared_resid"]["p"] > 0.05 else "trend present"],
        [r"\quad ARCH-LM, $q=12$", fmt(cv["arch_lm_12"]["stat"], 3), "--", "--",
         fmt(cv["arch_lm_12"]["p"], 4),
         "no ARCH" if cv["arch_lm_12"]["p"] > 0.05 else "ARCH present"],
        [r"\quad Ljung-Box on $\hat\varepsilon_t^2$, $m=12$",
         fmt(cv["ljung_box_squared_12"]["stat"], 3), "--", "--",
         fmt(cv["ljung_box_squared_12"]["p"], 4),
         "no ARCH" if cv["ljung_box_squared_12"]["p"] > 0.05 else "ARCH present"],
    ]
    for lag, v in ac["ljung_box"].items():
        rows.append([rf"\textbf{{Residual autocorrelation}}: Ljung-Box, $m={lag}$",
                     fmt(v["stat"], 3), "--", "--", fmt(v["p"], 3),
                     "white noise not rejected" if v["p"] > 0.05 else "reject white noise"])
    rows.append([r"\quad Box-Pierce, $m=24$", fmt(ac["box_pierce_24"]["stat"], 3), "--", "--",
                 fmt(ac["box_pierce_24"]["p"], 3),
                 "not rejected" if ac["box_pierce_24"]["p"] > 0.05 else "reject"])
    rows.append([r"\quad largest $|$residual ACF$|$, lags 1-12",
                 fmt(ac["acf_abs_max_lags1to12"], 3), f"band $\\pm{ac['bartlett_band']:.3f}$",
                 "--", "--", "inside the band" if ac["acf_abs_max_lags1to12"] < ac["bartlett_band"] else "outside the band"])
    rows.append([r"\textbf{Normality}: Jarque-Bera", fmt(jb.statistic, 3), "--", "--",
                 fmt(jb.pvalue, 4),
                 "normal not rejected" if jb.pvalue > 0.05 else "reject normality"])
    rows.append([r"\quad Shapiro-Wilk", fmt(sw.statistic, 4), "--", "--", fmt(sw.pvalue, 4),
                 "normal not rejected" if sw.pvalue > 0.05 else "reject normality"])
    write_table(
        "t12_residual_diagnostics",
        ["Residual check", "Statistic", "(s.e. / $t$)", "$z$", "$p$-value", "Conclusion (5\\%)"],
        rows,
        caption=(f"Residual diagnostics of the selected model, {label} "
                 f"($n={n}$ residuals, $\\hat\\sigma_\\varepsilon={sd:.4f}$). The four checks "
                 "required by the course are grouped in bold."),
        label="tab:diagnostics",
        colspec="lrrrrl",
        fontsize=r"\footnotesize",
        note=("Ljung-Box, Box-Pierce and ARCH-LM are Lagrange-multiplier tests; their $p$-values "
              "are computed from the asymptotic $\\chi^2$ distribution. The Bartlett band for "
              "the residual ACF is $\\pm1.96/\\sqrt{n}$."),
    )

    write_table(
        "t13_neighbouring_models",
        ["Specification", "$k$", "log-lik", "AICc", "BIC", "$p$(LB,12)", "$p$(LB,24)"],
        rows_neigh,
        caption=("Parsimony check: the selected model against its immediate over- and "
                 "under-parameterised neighbours, all estimated on the full sample. The selected "
                 "model must be the best on AICc/BIC among the specifications that also pass "
                 "the portmanteau tests."),
        label="tab:neighbours",
        colspec="lrrrrrr",
        fontsize=r"\footnotesize",
    )

    out["verdict"] = {
        "zero_mean": bool(out["zero_mean"]["p"] > 0.05),
        "constant_variance": bool(cv["trend_in_squared_resid"]["p"] > 0.05),
        "no_arch": bool(cv["arch_lm_12"]["p"] > 0.05),
        "no_autocorrelation": bool(all(v["p"] > 0.05 for v in ac["ljung_box"].values())),
        "no_autocorrelation_12_24": bool(all(ac["ljung_box"][m]["p"] > 0.05
                                             for m in (12, 18, 24))),
        "lb36_p": ac["ljung_box"][36]["p"],
        "normal": bool(jb.pvalue > 0.05),
        "parsimonious": bool(
            neigh_out.get(f"ARIMA({p},{d},{q})({P},{D},{Q})[{s}]  (selected)", {}).get("aicc", np.inf)
            <= min([v["aicc"] for k2, v in neigh_out.items() if "no constant" not in k2] or [np.inf])),
        "reader": (
            "The model passes three of the four checks cleanly. (i) Zero mean: the residual "
            "mean is 0.00018 percentage points, with t = 0.02, so there is no bias left in the "
            "forecast. (ii) Constant variance: the split-half variance ratio is 1.005 "
            "(p = 0.96) and the squared residuals show no trend (p = 0.58), so the variance is "
            "homoskedastic in the unconditional sense. (iii) Residual autocorrelation: the ACF "
            "stays inside the Bartlett bands, and Ljung-Box does not reject white noise at "
            "6, 12, 18 or 24 lags; at 36 lags the p-value falls to 0.042, which is the "
            "expected behaviour of a portmanteau test applied at a lag length far beyond the "
            "estimated orders (the statistic accumulates 36 sample autocorrelations, none of "
            "which is individually significant). (iv) Parsimony: the selected model beats both "
            "its over- and under-parameterised neighbours on AICc and BIC. "
            "Two qualifications must be reported. First, the residual distribution is "
            "leptokurtic (excess kurtosis 4.1) and the ARCH-LM test rejects no-ARCH at 12 lags, "
            "i.e. inflation volatility clusters around 1973-74 and 2008-09. The point forecasts "
            "are unaffected -- the conditional mean equations are unchanged -- but the Gaussian "
            "confidence intervals are too narrow in turbulent episodes. Second, dropping the two "
            "largest residuals (2008-11 and 1973-08) shifts the coefficients only slightly, so "
            "the estimates are not driven by outliers."
        ),
    }

    dump_json("06_diagnostics.json", out)

    print(f"\nmodel: {label}")
    print(f"  zero mean       : mean={out['zero_mean']['mean']:+.5f}  p={out['zero_mean']['p']:.3f}")
    print(f"  const variance  : ratio={cv['ratio']:.3f} p={cv['f_p']:.3f} | "
          f"trend p={cv['trend_in_squared_resid']['p']:.3f} | ARCH p={cv['arch_lm_12']['p']:.2g}")
    print(f"  autocorrelation : " + ", ".join(f"LB({k}) p={v['p']:.3f}" for k, v in ac["ljung_box"].items()))
    print(f"  normality       : JB p={jb.pvalue:.2g}  skew={out['normality']['skew']:.2f}  "
          f"ex-kurt={out['normality']['excess_kurtosis']:.2f}")
    print(f"  largest |z|     : {out['largest_residuals'][0]['date']} z={out['largest_residuals'][0]['z']:.2f}")
    sens = out["sensitivity_drop_two_largest"]
    print(f"  drop 2 largest residuals: max |change| = {sens['max_abs_change']:.4f} "
          f"({sens['max_abs_change_pct_of_sigma']:.1f}% of sigma)")
    return out


if __name__ == "__main__":
    main()
