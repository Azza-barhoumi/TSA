"""Step 4 -- Identification and estimation (Box-Jenkins stages 1-2, Ch. 3-4, 6, 7).

Model family
------------
The analysis is carried out on the *general multiplicative SARIMA model*

    phi(B) Phi(B^12) (1-B)^d (1-B^12)^D y_t = theta(B) Theta(B^12) e_t ,

with y_t = 100 ln P_t.  Step 3 fixed d = 1, D = 0 because the level is
integrated but there is no seasonal unit root, and flagged an annual echo in the
ACF that calls for Theta(B^12).

Identification (Ch. 6)
----------------------
Tentative orders come from the ACF/PACF of y_t and of Delta y_t (step 3), and
every candidate is then compared on the information criteria.  The grid covers
the non-seasonal ARMA orders and the multiplicative seasonal block up to
(P,Q) = (1,1) at s = 12.

Estimation (Ch. 4)
------------------
The syllabus requires the method of moments, least squares and maximum
likelihood.  All three are applied to the same data:

* method of moments  -- Yule-Walker (and Burg) equations for an AR model, and
  the closed-form MA(1) moment estimator, computed explicitly in this file;
* least squares      -- conditional sum of squares for MA(1)/ARMA(1,1),
  and Hannan-Rissanen (a two-step regression estimator) for ARMA;
* maximum likelihood -- the Kalman-filter likelihood used for the SARIMA model
  that is finally selected.

Outputs: figures f06-f08, tables t04-t06, results/04_estimation.json
"""
from __future__ import annotations

import itertools
import pickle
import sys
import warnings
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA as ARIMA_sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from utils import COL, dump_json, fmt, periods_to_timestamp, savefig, write_table  # noqa: E402

prep = import_module("02_prepare_data")

Y = "y"

# candidate grid: (p, q, P, Q)
GRID_P = range(0, 4)
GRID_Q = range(0, 4)
GRID_PQ = [(0, 0), (1, 0), (0, 1), (1, 1)]
MAX_K = 10


# --------------------------------------------------------------------------
# estimation machinery
# --------------------------------------------------------------------------
def fit_sarima(y: pd.Series, order, seasonal, trend="c"):
    """Gaussian maximum likelihood (Kalman filter) for a SARIMA model."""
    kw = dict(order=order, seasonal_order=seasonal, trend=trend,
              enforce_stationarity=True, enforce_invertibility=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, **kw).fit(disp=False, maxiter=500)
        # a non-converged optimum would bias both the information criteria and the
        # residual tests, so retry from a robust starting point and then polish
        if not res.mle_retvals.get("converged", True) or not np.isfinite(res.aic):
            alt = SARIMAX(y, **kw).fit(disp=False, method="nm", maxiter=2000)
            res = SARIMAX(y, **kw).fit(disp=False, start_params=alt.params, maxiter=1000)
    return res


def clean_resid(res, index) -> pd.Series:
    """Residuals on the estimation sample (the diffuse initialisation is dropped)."""
    burn = int(getattr(res, "loglikelihood_burn", 0) or 0)
    idx = pd.PeriodIndex(index[burn:], freq="M")
    return pd.Series(np.asarray(res.resid)[burn:], index=idx).dropna()


def ic(res, n: int) -> dict:
    k = len(res.params)
    aicc = res.aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.nan
    return {"llf": float(res.llf), "aic": float(res.aic), "aicc": float(aicc),
            "bic": float(res.bic), "k": k, "n": n}


def resid_diagnostics(resid: pd.Series) -> dict:
    lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
    ac = acf(resid, nlags=36, fft=False)
    return {"lb_stat_12": float(lb.loc[12, "lb_stat"]), "lb_p_12": float(lb.loc[12, "lb_pvalue"]),
            "lb_p_24": float(lb.loc[24, "lb_pvalue"]),
            "acf_11": float(ac[11]), "acf_12": float(ac[12]), "acf_13": float(ac[13]),
            "passes_lb": bool(lb.loc[12, "lb_pvalue"] > 0.05 and lb.loc[24, "lb_pvalue"] > 0.05)}


def label(order, seasonal, trend="c") -> str:
    p, d, q = order
    P, D, Q, m = seasonal
    s = f"ARIMA({p},{d},{q})"
    if m:
        s += f"({P},{D},{Q})[{m}]"
    if trend == "c":
        s += " + const"
    return s


# --------------------------------------------------------------------------
# Ch. 4: the three estimation methods on the stationary series
# --------------------------------------------------------------------------
def method_of_moments(x: pd.Series) -> dict:
    """Yule-Walker and Burg estimates of an AR(1) for pi_t, plus the MA(1) moment estimator.

    AR(1):  rho_1 = phi_1  =>  phi_1 = r_1.
    MA(1):  rho_1 = -theta / (1 + theta^2)  =>  theta solves theta^2 + theta/rho_1 + 1 = 0,
            and the invertible root is the one with |theta| < 1.
    Var(e) = gamma_0 (1 + theta^2) for MA(1), gamma_0 (1 - phi^2) for AR(1).
    """
    x = x.dropna().values.astype(float)
    n = len(x)
    xc = x - x.mean()
    g0 = float(np.mean(xc**2))
    g1 = float(np.mean(xc[1:] * xc[:-1]))
    r1 = g1 / g0

    phi_mom = r1
    var_e_ar = g0 * (1 - phi_mom**2)

    # MA(1) moment estimator: solve theta^2 + theta/r1 + 1 = 0.
    # The discriminant 1/r1^2 - 4 is negative when |r1| > 0.5, in which case the
    # moment estimator has NO real solution -- a textbook limitation of the
    # method of moments that motivates least squares / ML (Ch. 4).
    ma1_moment_feasible = bool(abs(r1) <= 0.5)
    if not ma1_moment_feasible:
        theta_sol = np.array([np.nan, np.nan])
    else:
        disc = 1 / r1**2 - 4
        theta_sol = np.array([(-1 / r1 + np.sqrt(disc)) / 2, (-1 / r1 - np.sqrt(disc)) / 2])
        theta_sol = theta_sol[np.argsort(np.abs(theta_sol))]   # invertible root first
    theta_mom = float(theta_sol[0])
    var_e_ma = g0 * (1 + theta_mom**2)

    # Yule-Walker / Burg via statsmodels for the AR(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yw = ARIMA_sm(pd.Series(x), order=(1, 0, 0), trend="c").fit(method="yule_walker")
        bg = ARIMA_sm(pd.Series(x), order=(1, 0, 0), trend="c").fit(method="burg")
    return {
        "n": n, "mean": float(x.mean()), "gamma0": g0, "gamma1": g1, "rho1": float(r1),
        "ar1_phi_mom": float(phi_mom), "ar1_var_e_mom": float(var_e_ar),
        "ar1_phi_yule_walker": float(np.asarray(yw.params)[1]),
        "ar1_phi_burg": float(np.asarray(bg.params)[1]),
        "ma1_theta_mom": theta_mom, "ma1_var_e_mom": float(var_e_ma),
        "ma1_theta_roots": theta_sol.tolist(),
        "ma1_moment_feasible": ma1_moment_feasible,
        "ma1_discriminant": float(1 / r1**2 - 4) if r1 != 0 else float("nan"),
    }


def conditional_sum_of_squares(x: pd.Series, order=(0, 0, 1)) -> dict:
    """Conditional least squares: minimise sum of squared one-step errors.

    MA(1):  e_t = x_t - mu - theta e_{t-1},  e_0 = 0
    ARMA(1,1): e_t = x_t - mu - phi x_{t-1} - theta e_{t-1}
    """
    v = x.dropna().values.astype(float)

    def css(params):
        if order == (0, 0, 1):
            mu, th = params
            e = np.zeros_like(v)
            for t in range(1, len(v)):
                e[t] = v[t] - mu - th * e[t - 1]
            return float(np.sum(e[1:] ** 2))
        mu, ph, th = params
        e = np.zeros_like(v)
        for t in range(1, len(v)):
            e[t] = v[t] - mu - ph * v[t - 1] - th * e[t - 1]
        return float(np.sum(e[1:] ** 2))

    x0 = [v.mean(), 0.3] if order == (0, 0, 1) else [v.mean(), 0.5, 0.2]
    res = optimize.minimize(css, x0, method="L-BFGS-B")
    out = {"sse": float(res.fun), "sigma2_css": float(res.fun / (len(v) - 1)),
           "n": len(v), "x": res.x.tolist()}
    if order == (0, 0, 1):
        out.update(mu=float(res.x[0]), theta=float(res.x[1]))
    else:
        out.update(mu=float(res.x[0]), phi=float(res.x[1]), theta=float(res.x[2]))
    return out


def hannan_rissanen(x: pd.Series, order=(1, 0, 1)) -> dict:
    """Two-step least-squares estimator (Hannan-Rissanen 1982)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA_sm(x.dropna(), order=order, trend="c").fit(method="hannan_rissanen")
    return {"params": [float(v) for v in np.asarray(res.params)],
            "names": [str(n) for n in res.param_names],
            "sigma2": float(np.asarray(res.params)[-1])}


def mle_small(x: pd.Series, order=(1, 0, 1)) -> dict:
    """Exact Gaussian MLE for the same small models (for the comparison table)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA_sm(x.dropna(), order=order, trend="c").fit()
    return {"params": [float(v) for v in np.asarray(res.params)],
            "names": [str(n) for n in res.param_names],
            "llf": float(res.llf), "sigma2": float(np.asarray(res.params)[-1])}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_ic(rows):
    rows = sorted(rows, key=lambda r: r["bic"])
    top = rows[:14]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(top))
    ax.barh(y - 0.2, [r["bic"] for r in top], height=0.38, color=COL["data"], label="BIC")
    ax.barh(y + 0.2, [r["aicc"] for r in top], height=0.38, color=COL["accent"], label="AICc")
    ax.set_yticks(y, [r["label"] for r in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("information criterion (lower is better)")
    ax.set_title("Fourteen best candidate models by BIC")
    ax.legend()
    lo = min(min(r["bic"], r["aicc"]) for r in top)
    hi = min(max(r["bic"], r["aicc"]) for r in top)
    ax.set_xlim(lo - 5, hi + 10)
    fig.tight_layout()
    return savefig(fig, "f06_model_selection.png")


def fig_seasonal_block(no_seasonal, with_seasonal):
    """Show the annual echo in the ACF and what the seasonal block does to it."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for ax, r, ttl in [(axes[0], no_seasonal, "(a) without seasonal block"),
                       (axes[1], with_seasonal, "(b) with $\\Theta(B^{12})$")]:
        s = r["resid"]
        a = acf(s, nlags=36, fft=False)[1:]
        ci = 1.96 / np.sqrt(len(s))
        ax.bar(np.arange(1, len(a) + 1), a, width=0.6,
               color=COL["fit"] if r is with_seasonal else COL["bench"])
        ax.axhline(0, color="k", lw=0.6)
        for v in (ci, -ci):
            ax.axhline(v, color="k", lw=0.7, ls="--")
        ax.axvspan(10.5, 13.5, color=COL["shade"], alpha=0.7, lw=0, zorder=0)
        ax.set_title(f"{ttl}\n{r['label']}", fontsize=8)
        ax.set_xlabel("lag (months)")
    axes[0].set_ylabel("residual ACF")
    fig.suptitle("Residual autocorrelation with and without the seasonal MA term", fontsize=9)
    fig.tight_layout()
    return savefig(fig, "f07_seasonal_block.png")


def fig_fitted(res, y, name):
    burn = int(getattr(res, "loglikelihood_burn", 0) or 0)
    fitted = np.asarray(res.fittedvalues)
    resid = np.asarray(res.resid)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.3), sharex=True)
    ax = axes[0]
    ax.plot(periods_to_timestamp(y.index), y.values, color=COL["data"], lw=0.9, label="observed")
    ax.plot(periods_to_timestamp(y.index[burn:]), fitted[burn:], color=COL["fit"], lw=0.9,
            ls="--", label="fitted")
    ax.set_ylabel("$100\\ln P_t$")
    ax.set_title(f"(a) Observed and fitted log CPI: {name}")
    ax.legend(loc="upper left")
    ax = axes[1]
    r = resid[burn:]
    ax.plot(periods_to_timestamp(y.index[burn:]), r, color=COL["bench"], lw=0.7)
    ax.axhline(0, color="k", lw=0.6)
    sd = np.nanstd(r)
    ax.fill_between(periods_to_timestamp(y.index[burn:]), -2 * sd, 2 * sd,
                    color=COL["shade"], alpha=0.7, lw=0)
    ax.set_ylabel("residual")
    ax.set_xlabel("year")
    ax.set_title("(b) Residuals (shaded band: $\\pm 2$ standard deviations)")
    fig.tight_layout()
    return savefig(fig, "f08_fitted_residuals.png")


# --------------------------------------------------------------------------
def main():
    df = prep.load_processed()
    y = df[Y].astype(float)
    pi = df["pi"]
    n = len(y)

    # ================= Ch. 3/6: identification over the grid =============
    print("estimating the candidate grid (this takes ~2 minutes) ...")
    rows = []
    for p, q in itertools.product(GRID_P, GRID_Q):
        for P, Q in GRID_PQ:
            if (p, q) == (0, 0) and (P, Q) == (0, 0):
                continue
            order, seas = (p, 1, q), (P, 0, Q, 12)
            try:
                res = fit_sarima(y, order, seas)
            except Exception:
                continue
            if not np.isfinite(res.aic):
                continue
            m = ic(res, n)
            if m["k"] > MAX_K:
                continue
            resid = clean_resid(res, y.index)
            d = resid_diagnostics(resid)
            rec = {"order": order, "seasonal": seas, "label": label(order, seas), **m, **d,
                   "resid": resid, "res": res}
            rows.append(rec)
            print(f"  {rec['label']:34s} k={m['k']:2d} AICc={m['aicc']:8.2f} "
                  f"BIC={m['bic']:8.2f} LB12={d['lb_p_12']:.3f} "
                  f"{'passes' if d['passes_lb'] else '      '}")

    # selection: BIC among models whose residuals pass Ljung-Box at 12 and 24
    clean = [r for r in rows if r["passes_lb"]]
    pool = clean if clean else rows
    final = min(pool, key=lambda r: r["bic"])
    res = final["res"]
    print(f"\nselected by BIC-with-clean-residuals: {final['label']}  BIC={final['bic']:.2f}")

    # ================= Ch. 4: three estimation methods ==================
    print("estimating with the three methods of Ch. 4 ...")
    mom = method_of_moments(pi)
    css_ma1 = conditional_sum_of_squares(pi, (0, 0, 1))
    css_arma11 = conditional_sum_of_squares(pi, (1, 0, 1))
    hr_arma11 = hannan_rissanen(pi, (1, 0, 1))
    mle_ma1 = mle_small(pi, (0, 0, 1))
    mle_arma11 = mle_small(pi, (1, 0, 1))

    # ================= figures ==========================================
    print("figures:")
    fig_ic(rows)
    # The comparison of interest is like-for-like: the best model with a seasonal
    # block against the *same* non-seasonal orders with the seasonal block removed.
    # Picking the best of each family independently would compare different orders
    # and so would not isolate the contribution of the seasonal term.
    with_seas = next((r for r in sorted(rows, key=lambda r: r["bic"])
                      if r["seasonal"][0] > 0 and r["seasonal"][2] > 0), None)
    no_seas = None
    if with_seas is not None:
        p_, _, q_ = with_seas["order"]
        no_seas = next((r for r in rows
                        if r["order"] == (p_, 1, q_)
                        and r["seasonal"][0] == 0 and r["seasonal"][2] == 0), None)
    if no_seas and with_seas:
        fig_seasonal_block(no_seas, with_seas)
    fig_fitted(res, y, final["label"])

    # ================= tables ===========================================
    # T4: information criteria for the identified candidates
    pick = sorted([r for r in rows if r["seasonal"][0] == 0 and r["seasonal"][2] == 0],
                  key=lambda r: r["bic"])[:8]
    pick += sorted([r for r in rows if r["seasonal"][0] > 0 or r["seasonal"][2] > 0],
                   key=lambda r: r["bic"])[:8]
    seen, rows_t = set(), []
    for r in pick:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        rows_t.append([r["label"].replace(" + const", ""), f"{r['k']}", fmt(r["llf"], 2),
                       fmt(r["aic"], 2), fmt(r["aicc"], 2), fmt(r["bic"], 2),
                       fmt(r["lb_p_12"], 3), fmt(r["lb_p_24"], 3),
                       "yes" if r["passes_lb"] else "no"])
    write_table(
        "t05_model_selection",
        ["SARIMA model for $y_t=100\\ln P_t$", "$k$", "log-lik", "AIC", "AICc", "BIC",
         "$p$(LB,12)", "$p$(LB,24)", "Adequate?"],
        rows_t,
        caption=("Candidate multiplicative SARIMA models, all with $d=1$, $D=0$ and a constant. "
                 "The first block has no seasonal block ($P=Q=0$); the second block includes "
                 "$\\Phi(B^{12})$ and/or $\\Theta(B^{12})$."),
        label="tab:selection",
        colspec="lrrrrrrrl",
        fontsize=r"\footnotesize",
        note=("$k$ counts estimated parameters; LB is the Ljung-Box portmanteau statistic on the "
              "residuals. ``Adequate?'' requires the null of white-noise residuals not to be "
              "rejected at 12 and 24 lags."),
    )

    # T5: MoM / CSS / HR / MLE comparison on pi_t
    rows_t = []
    rows_t.append([r"AR(1): $\phi_1$", fmt(mom["ar1_phi_mom"], 4), "--",
                   fmt(mom["ar1_phi_yule_walker"], 4), fmt(mom["ar1_phi_burg"], 4), "--"])
    rows_t.append([r"AR(1): $\sigma^2_\varepsilon$", fmt(mom["ar1_var_e_mom"], 4), "--", "--",
                   "--", "--"])
    feas = mom["ma1_moment_feasible"]
    rows_t.append([r"MA(1): $\theta_1$",
                   fmt(mom["ma1_theta_mom"], 4) if feas else r"no real root$^{\dagger}$",
                   fmt(css_ma1["theta"], 4), "--", "--", fmt(mle_ma1["params"][1], 4)])
    rows_t.append([r"MA(1): $\sigma^2_\varepsilon$",
                   fmt(mom["ma1_var_e_mom"], 4) if feas else "--",
                   fmt(css_ma1["sigma2_css"], 4), "--", "--", fmt(mle_ma1["sigma2"], 4)])
    rows_t.append([r"ARMA(1,1): $\phi_1$", "--", fmt(css_arma11["phi"], 4),
                   "--", fmt(hr_arma11["params"][1], 4), fmt(mle_arma11["params"][1], 4)])
    rows_t.append([r"ARMA(1,1): $\theta_1$", "--", fmt(css_arma11["theta"], 4),
                   "--", fmt(hr_arma11["params"][2], 4), fmt(mle_arma11["params"][2], 4)])
    write_table(
        "t06_estimators",
        ["Parameter", "Moments", "Cond. LS", "Yule-Walker", "Hannan-Rissanen", "MLE"],
        rows_t,
        caption=("The three estimation methods of Ch. 4 applied to monthly inflation $\\pi_t$. "
                 "The moment estimator for AR(1) is $\\hat\\phi_1=r_1$; for MA(1) it solves "
                 "$\\theta^2+\\theta/r_1+1=0$ and keeps the invertible root."),
        label="tab:estimators",
        colspec="lrrrrr",
        fontsize=r"\footnotesize",
        note=(f"$n={mom['n']}$, $\\bar\\pi={mom['mean']:.4f}$, $r_1={mom['rho1']:.4f}$. "
              "Conditional LS minimises $\\sum \\hat\\varepsilon_t^2$ with "
              "$\\hat\\varepsilon_0=0$; MLE uses the exact Gaussian likelihood. "
              "$^{\\dagger}$For $|r_1|>0.5$ the quadratic $\\theta^2+\\theta/r_1+1=0$ "
              "has a negative discriminant, so the method of moments has no real solution: "
              "the moment estimator is moment-infeasible for this series and least squares or "
              "ML must be used instead."),
    )

    # T6: final parameter table
    names = [str(nm) for nm in res.param_names]
    params, bse = np.asarray(res.params), np.asarray(res.bse)
    tv, pv = np.asarray(res.tvalues), np.asarray(res.pvalues)
    ci = np.asarray(res.conf_int())
    pretty = {"intercept": r"$c$ (constant)", "ar.S.L12": r"$\Phi_1$ (seasonal AR, $B^{12}$)",
              "ma.S.L12": r"$\Theta_1$ (seasonal MA, $B^{12}$)",
              "sigma2": r"$\sigma^2_\varepsilon$"}
    rows_t = []
    for i, nm in enumerate(names):
        lab = pretty.get(nm)
        if lab is None:
            if nm.startswith("ar.L"):
                lab = rf"$\phi_{{{nm.split('L')[1]}}}$ (AR, $B^{{{nm.split('L')[1]}}}$)"
            elif nm.startswith("ma.L"):
                lab = rf"$\theta_{{{nm.split('L')[1]}}}$ (MA, $B^{{{nm.split('L')[1]}}}$)"
            else:
                lab = nm
        st = "***" if pv[i] < 0.01 else "**" if pv[i] < 0.05 else "*" if pv[i] < 0.10 else ""
        rows_t.append([lab, fmt(params[i], 4) + (rf"$^{{{st}}}$" if st else ""),
                       fmt(bse[i], 4), fmt(tv[i], 3), fmt(pv[i], 4),
                       f"[{fmt(ci[i, 0], 4)},\\ {fmt(ci[i, 1], 4)}]"])
    write_table(
        "t07_parameters",
        ["Parameter", "Estimate", "Std. error", "$z$", "$p$-value", "95\\% CI"],
        rows_t,
        caption=(f"Maximum-likelihood estimates of the selected SARIMA model, {final['label']} "
                 f"($n={n}$). Significance: $^{{***}}$ 1\\%, $^{{**}}$ 5\\%, $^{{*}}$ 10\\%."),
        label="tab:params",
        colspec="lrrrrl",
        fontsize=r"\footnotesize",
        note=(f"Log-likelihood ${res.llf:.2f}$; AIC ${res.aic:.2f}$; AICc ${final['aicc']:.2f}$; "
              f"BIC ${res.bic:.2f}$; $\\hat\\sigma_\\varepsilon="
              f"{np.sqrt(params[names.index('sigma2')]):.4f}$ (percentage points per month)."),
    )

    # ================= derived quantities ===============================
    idx = {k: i for i, k in enumerate(names)}

    def get(nm, default=0.0):
        return float(params[idx[nm]]) if nm in idx else default

    phi = [get(f"ar.L{i}") for i in range(1, final["order"][0] + 1)]
    theta = [get(f"ma.L{i}") for i in range(1, final["order"][2] + 1)]
    Phi, Theta = get("ar.S.L12"), get("ma.S.L12")
    inv_ar = np.roots(np.r_[1, -np.array(phi)]) if phi else np.array([])
    inv_ma = np.roots(np.r_[1, np.array(theta)]) if theta else np.array([])
    dom = float(np.max(np.abs(inv_ar))) if inv_ar.size else 0.0
    half_life = float(np.log(0.5) / np.log(dom)) if 0 < dom < 1 else np.nan

    summary = {
        "dependent_variable": "y_t = 100 * ln(CPI_t)",
        "final_model": final["label"],
        "order": list(final["order"]), "seasonal_order": list(final["seasonal"]),
        "n": n, "k": final["k"], "llf": float(res.llf), "aic": float(res.aic),
        "aicc": float(final["aicc"]), "bic": float(res.bic),
        "sigma2": float(params[idx["sigma2"]]),
        "sigma_eps": float(np.sqrt(params[idx["sigma2"]])),
        "params": {k: float(v) for k, v in zip(names, params)},
        "bse": {k: float(v) for k, v in zip(names, bse)},
        "pvalues": {k: float(v) for k, v in zip(names, pv)},
        "conf_int": {k: [float(ci[i, 0]), float(ci[i, 1])] for i, k in enumerate(names)},
        "resid_stats": {k: (float(v) if np.isscalar(v) and not isinstance(v, bool) else v)
                        for k, v in final["resid"].items()},
        "lb_p_12": final["lb_p_12"], "lb_p_24": final["lb_p_24"],
        "stationary": bool(np.all(np.abs(inv_ar) < 1) and abs(Phi) < 1),
        "invertible": bool(np.all(np.abs(inv_ma) < 1) and abs(Theta) < 1),
        "ar_coefs": phi, "ma_coefs": theta, "seasonal_ar": Phi, "seasonal_ma": Theta,
        "half_life_months": half_life,
        "estimators": {"moments": mom, "css_ma1": css_ma1, "css_arma11": css_arma11,
                       "hannan_rissanen_arma11": hr_arma11, "mle_ma1": mle_ma1,
                       "mle_arma11": mle_arma11},
        "candidates": [{k: (v if not isinstance(v, pd.Series) else None)
                        for k, v in r.items() if k not in ("res", "resid")} for r in rows],
    }

    # persistence / implied long-run inflation for ARIMA(p,1,q) without seasonal unit root
    ar_sum = (1 - sum(phi)) * (1 - Phi)
    if "intercept" in idx and abs(ar_sum) > 1e-9:
        summary["implied_mean_monthly_inflation"] = float(get("intercept") / ar_sum)
        summary["implied_mean_annual_inflation"] = summary["implied_mean_monthly_inflation"] * 12
    else:
        summary["implied_mean_monthly_inflation"] = float("nan")
        summary["implied_mean_annual_inflation"] = float("nan")

    dump_json("04_estimation.json", summary)

    # save the final fit for the forecasting step
    with open(C.RESULTS / "final_model.pkl", "wb") as fh:
        pickle.dump({"label": final["label"], "order": final["order"],
                     "seasonal": final["seasonal"], "params": params.tolist()}, fh)

    print(f"\nselected: {final['label']}")
    print(f"  AICc={final['aicc']:.2f} BIC={final['bic']:.2f} loglik={res.llf:.2f} k={final['k']}")
    print(f"  sigma_eps={summary['sigma_eps']:.4f}  stationary={summary['stationary']} "
          f"invertible={summary['invertible']}")
    print(f"  half-life={half_life:.2f} months  implied annual inflation="
          f"{summary['implied_mean_annual_inflation']:.3f}%")
    print(f"  residual LB(12) p={final['lb_p_12']:.3f}, LB(24) p={final['lb_p_24']:.3f}")
    print(f"  MoM: phi={mom['ar1_phi_mom']:.3f} theta_MA1={mom['ma1_theta_mom']:.3f} "
          f"| CSS: theta={css_ma1['theta']:.3f} | MLE: theta={mle_ma1['params'][1]:.3f}")
    return summary


if __name__ == "__main__":
    main()
