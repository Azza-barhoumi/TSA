"""Step 3 -- Exploratory data analysis and the figures used in the report.

Produces
--------
figures/f01_level_and_inflation.png   level, monthly and annual inflation
figures/f02_cpi_history.png           full Kaggle sample 1913-2020 (context)
figures/f03_regimes.png               rolling mean/std of inflation by regime
figures/f04_seasonal_box.png          monthly boxplots + autocorrelation of pi
figures/f05_acf_pacf_levels.png       ACF/PACF of y_t and of pi_t
figures/f06_hist_qq.png               distribution of monthly inflation

tables/t01_descriptives.tex
tables/t02_regimes.tex
results/03_eda.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from utils import (  # noqa: E402
    COL, dump_json, fmt, periods_to_timestamp, savefig, write_table,
)
from importlib import import_module

prep = import_module("02_prepare_data")


def fig_level_and_inflation(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.6), sharex=True)
    ax = axes[0]
    ax.plot(periods_to_timestamp(df.index.to_timestamp().to_period("M")), df["cpi"], color=COL["data"], lw=1.0)
    ax.set_ylabel("CPI-U index\n(1982-84 = 100)")
    ax.set_title("(a) Consumer Price Index, level $P_t$")

    ax = axes[1]
    ax.plot(df.index.to_timestamp(), df["pi_year"], color=COL["data"], lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(df["pi_year"].mean(), color=COL["fit"], lw=1.0, ls="--",
               label=f"sample mean = {df['pi_year'].mean():.2f}%")
    ax.set_ylabel("annual inflation\n$\\pi^{{12}}_t$ (\\%)")
    ax.set_title("(b) Year-on-year inflation, $100\\Delta_{12}\\ln P_t$")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(df.index.to_timestamp(), df["pi"], color=COL["accent"], lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    sd = df["pi"].std()
    ax.fill_between(periods_to_timestamp(df.index), -2 * sd, 2 * sd, color=COL["shade"], alpha=0.6,
                    lw=0, label=r"$\pm 2$ s.d. band")
    ax.set_ylabel("monthly inflation\n$\\pi_t$ (\\%)")
    ax.set_title("(c) Month-on-month inflation, $100\\Delta\\ln P_t$")
    ax.legend(loc="upper right")
    ax.set_xlabel("year")
    for a in axes:
        a.margins(x=0.01)
    fig.suptitle("US CPI-U monthly inflation, 1957-01 to 2020-12", y=0.995, fontsize=10)
    fig.tight_layout()
    return savefig(fig, "f01_level_and_inflation.png")


def fig_full_history(df_est: pd.DataFrame):
    raw = pd.read_csv(C.RAW_CSV)
    raw["date"] = pd.to_datetime(raw["Yearmon"], format="%d-%m-%Y")
    raw = raw.sort_values("date")
    full = pd.Series(raw["CPI"].values, index=pd.PeriodIndex(raw["date"], freq="M"))
    dlog = 100 * np.log(full).diff(12)

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    axes[0].plot(periods_to_timestamp(full.index), full.values, color=COL["data"], lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("CPI-U (log scale)")
    axes[0].axvspan(pd.Timestamp("1957-01-01"), pd.Timestamp("2020-12-31"),
                    color=COL["shade"], alpha=0.7, lw=0)
    axes[0].set_title("(a) Full Kaggle sample, 1913-01 to 2020-12 (log scale); shaded = estimation sample")

    axes[1].plot(periods_to_timestamp(dlog.index), dlog.values, color=COL["fit"], lw=0.8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].axvspan(pd.Timestamp("1957-01-01"), pd.Timestamp("2020-12-31"),
                    color=COL["shade"], alpha=0.7, lw=0)
    axes[1].set_ylabel("annual inflation (\\%)")
    axes[1].set_xlabel("year")
    axes[1].set_title("(b) Year-on-year inflation over the same 108 years")
    fig.tight_layout()
    return savefig(fig, "f02_cpi_history.png")


def fig_regimes(df: pd.DataFrame):
    ann = df["pi_year"].dropna()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    roll_m = ann.rolling(60).mean()
    roll_s = ann.rolling(60).std()
    ax.plot(periods_to_timestamp(ann.index), ann.values, color=COL["shade"], lw=1.0, label="annual inflation")
    ax.plot(periods_to_timestamp(roll_m.index), roll_m.values, color=COL["data"], lw=1.8, label="5-year rolling mean")
    ax.plot(periods_to_timestamp(roll_s.index), roll_s.values, color=COL["fit"], lw=1.4, ls="--",
            label="5-year rolling s.d. (volatility)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("\\% (annualised)")
    ax.set_xlabel("year")
    ax.set_title("Inflation level and volatility: Great Inflation vs Great Moderation")
    ax.legend(loc="upper left", ncol=1)
    ax.margins(x=0.01)
    return savefig(fig, "f03_regimes.png")


def fig_seasonal(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    data = [df.loc[df["calendar_month"] == m, "pi"].dropna().values for m in range(1, 13)]
    bp = axes[0].boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                         flierprops=dict(marker="o", ms=2.5, alpha=0.5,
                                         markerfacecolor=COL["accent"], markeredgecolor="none"))
    for p in bp["boxes"]:
        p.set(facecolor="#dce6f2", edgecolor=COL["data"], linewidth=0.8)
    for k in ("whiskers", "caps", "medians"):
        for p in bp[k]:
            p.set(color=COL["data"], linewidth=0.8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("$\\pi_t$ (\\%)")
    axes[0].set_title("(a) Monthly inflation by calendar month")

    plot_acf(df["pi"].dropna(), lags=36, ax=axes[1], color=COL["data"], vlines_kwargs={"color": COL["data"]})
    axes[1].set_title("(b) Autocorrelation of $\\pi_t$, 36 lags")
    axes[1].set_xlabel("lag (months)")
    axes[1].set_ylabel("ACF")
    fig.tight_layout()
    return savefig(fig, "f04_seasonal_box.png")


def fig_acf_pacf(df: pd.DataFrame, nlags: int = 36):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4))
    for row, (col, name) in enumerate([("y", "$y_t=100\\ln P_t$ (level)"),
                                      ("pi", "$\\pi_t$ (monthly log-change)")]):
        s = df[col].dropna()
        plot_acf(s, lags=nlags, ax=axes[row, 0], color=COL["data"],
                 vlines_kwargs={"color": COL["data"]})
        plot_pacf(s, lags=nlags, ax=axes[row, 1], method="ywm", color=COL["fit"],
                  vlines_kwargs={"color": COL["fit"]})
        axes[row, 0].set_title(f"ACF: {name}")
        axes[row, 1].set_title(f"PACF: {name}")
        for a in axes[row]:
            a.set_xlabel("lag (months)")
            a.set_ylabel("")
    fig.tight_layout()
    return savefig(fig, "f05_acf_pacf_levels.png")


def fig_hist(df: pd.DataFrame):
    s = df["pi"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    axes[0].hist(s, bins=40, density=True, color="#dce6f2", edgecolor=COL["data"], linewidth=0.5)
    xs = np.linspace(s.min(), s.max(), 400)
    axes[0].plot(xs, stats.norm.pdf(xs, s.mean(), s.std()), color=COL["fit"], lw=1.4, label="normal fit")
    kde = stats.gaussian_kde(s)
    axes[0].plot(xs, kde(xs), color=COL["accent"], lw=1.4, ls="--", label="kernel density")
    axes[0].set_xlabel("$\\pi_t$ (\\%)")
    axes[0].set_ylabel("density")
    axes[0].set_title("(a) Distribution of monthly inflation")
    axes[0].legend()

    stats.probplot(s, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set(marker="o", ms=2.5, markerfacecolor=COL["data"], markeredgecolor="none")
    axes[1].get_lines()[1].set(color=COL["fit"], lw=1.2)
    axes[1].set_title("(b) Normal Q-Q plot")
    axes[1].set_xlabel("theoretical quantiles")
    axes[1].set_ylabel("sample quantiles")
    fig.tight_layout()
    return savefig(fig, "f06_hist_qq.png")


def main():
    df = prep.load_processed()
    out: dict = {}

    out["descriptives"] = {
        "pi": {"n": int(df["pi"].notna().sum()), "mean": float(df["pi"].mean()),
               "sd": float(df["pi"].std()), "skew": float(df["pi"].skew()),
               "kurt": float(df["pi"].kurtosis())},
        "pi_year": {"n": int(df["pi_year"].notna().sum()), "mean": float(df["pi_year"].mean()),
                    "sd": float(df["pi_year"].std())},
    }
    jb = stats.jarque_bera(df["pi"].dropna())
    sw = stats.shapiro(df["pi"].dropna())
    out["normality"] = {"jarque_bera_stat": float(jb.statistic), "jarque_bera_p": float(jb.pvalue),
                        "shapiro_stat": float(sw.statistic), "shapiro_p": float(sw.pvalue)}
    # Ljung-Box on pi at several lags
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(df["pi"].dropna(), lags=[6, 12, 24, 36], return_df=True)
    out["ljung_box_pi"] = lb.round(4).to_dict(orient="index")

    # Kruskal-Wallis on seasonal means (H0: identical location across months)
    groups = [df.loc[df["calendar_month"] == m, "pi"].dropna() for m in range(1, 13)]
    kw = stats.kruskal(*groups)
    out["seasonal_kruskal"] = {"stat": float(kw.statistic), "p": float(kw.pvalue)}

    # variance ratio: sd of pi in 1968-1982 vs 1990-2019
    early = df.loc["1968":"1982", "pi"].dropna()
    late = df.loc["1990":"2019", "pi"].dropna()
    out["volatility_ratio"] = {"sd_1968_1982": float(early.std()), "sd_1990_2019": float(late.std()),
                              "ratio": float(early.std() / late.std())}

    # ---------- figures ----------
    print("figures:")
    fig_level_and_inflation(df)
    fig_full_history(df)
    fig_regimes(df)
    fig_seasonal(df)
    fig_acf_pacf(df)
    fig_hist(df)

    # ---------- tables ----------
    print("tables:")
    rows = []
    for col, lab in [("cpi", r"CPI level $P_t$"), ("y", r"Log index $y_t=100\ln P_t$"),
                     ("pi", r"Monthly inflation $\pi_t$"), ("pi_year", r"Annual inflation $\pi^{12}_t$")]:
        s = df[col].dropna()
        rows.append([lab, f"{s.size:d}", fmt(s.mean(), 3), fmt(s.median(), 3), fmt(s.std(), 3),
                     fmt(s.min(), 3), fmt(s.max(), 3), fmt(s.skew(), 2), fmt(s.kurtosis(), 2)])
    write_table(
        "t01_descriptives",
        ["Series", "$n$", "Mean", "Median", "S.d.", "Min.", "Max.", "Skew.", "Ex. kurt."],
        rows,
        caption="Descriptive statistics of the US CPI-U series, monthly, 1957-01 to 2020-12.",
        label="tab:descriptives",
        note=("Monthly inflation is $\\pi_t=100\\Delta\\ln P_t$; annual inflation is "
              "$\\pi^{12}_t=100\\Delta_{12}\\ln P_t$. Excess kurtosis is reported (0 for a normal)."),
    )

    reg = [
        ("1957--1967", "Bretton Woods, price stability", "1957", "1967"),
        ("1968--1979", "Vietnam war financing, two oil shocks", "1968", "1979"),
        ("1980--1989", "Volcker disinflation, double-dip recessions", "1980", "1989"),
        ("1990--1999", "Great Moderation, phase I", "1990", "1999"),
        ("2000--2007", "Great Moderation, phase II", "2000", "2007"),
        ("2008--2009", "Global financial crisis", "2008", "2009"),
        ("2010--2019", "Below-target inflation", "2010", "2019"),
        ("2020", "COVID-19 shock", "2020", "2020"),
    ]
    rows = []
    for per, desc, a, b in reg:
        s = df.loc[a:b, "pi_year"].dropna()
        rows.append([per, desc, f"{s.size:d}", fmt(s.mean(), 2), fmt(s.std(), 2),
                     fmt(s.min(), 2), fmt(s.max(), 2)])
    write_table(
        "t02_regimes",
        ["Period", "Historical context", "$n$", "Mean", "S.d.", "Min.", "Max."],
        rows,
        caption="Annual CPI inflation by historical regime (percent, twelve-month log change).",
        label="tab:regimes",
        colspec="llrrrrr",
        fontsize=r"\footnotesize",
    )

    dump_json("03_eda.json", out)
    return out


if __name__ == "__main__":
    main()
