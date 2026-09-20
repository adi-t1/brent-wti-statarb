"""
Figures for the research note.

Conventions used throughout (and why)
------------------------------------
* No dual-axis charts anywhere. Two series on two different y-scales invites
  the reader to "see" a relationship whose apparent strength is an artefact of
  how the two axes were scaled. Where two quantities of different units need
  comparing, they get stacked panels sharing an x-axis instead.
* A fixed categorical colour order, checked for colour-vision deficiency
  separation, rather than matplotlib's default cycle.
* Every series is identified by a legend AND, where there is room, a direct
  label, so identity never depends on colour alone.
* Grid and axes are recessive; the data is the darkest thing on the page.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# --- palette (validated for CVD separation) -------------------------------
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
C_MAGENTA = "#e87ba4"
C_VIOLET = "#4a3aa7"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#d8d8d4"
C_SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE,
    "axes.edgecolor": C_GRID,
    "axes.labelcolor": C_TEXT2,
    "axes.titlecolor": C_TEXT,
    "text.color": C_TEXT,
    "xtick.color": C_TEXT2,
    "ytick.color": C_TEXT2,
    "grid.color": C_GRID,
    "grid.linewidth": 0.7,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9.5,
    "axes.titlesize": 11.5,
    "legend.frameon": False,
    "figure.dpi": 130,
})

# Regime annotations used on the spread chart. These are dated from published
# market history, NOT chosen by looking at where the spread happened to move --
# that distinction matters, because shading a chart with regimes you picked by
# staring at the same chart is circular.
REGIMES = [
    ("2007-07-30", "2010-12-31", "Pre-shale parity", "#f2f2ef"),
    ("2011-01-01", "2014-12-31", "Shale glut: Cushing\nbottleneck, US export ban", "#fdf0e8"),
    ("2015-01-01", "2019-12-31", "Export ban repealed\n(Dec 2015), spread narrows", "#eef6ee"),
    ("2020-01-01", "2021-12-31", "COVID demand collapse\n& negative WTI print", "#fdecef"),
    ("2022-01-01", "2022-12-31", "Russia/Ukraine:\nBrent premium blows out", "#fdf0e8"),
]


def _save(fig, name):
    path = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
def plot_prices(px, exclude_dates=config.ESTIMATION_EXCLUDE_DATES):
    """Raw front-month price levels, with the negative print called out."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(px.index, px["BRENT"], color=C_BLUE, lw=1.1, label="Brent (BZ=F)")
    ax.plot(px.index, px["WTI"], color=C_ORANGE, lw=1.1, label="WTI (CL=F)")
    ax.axhline(0, color=C_TEXT2, lw=0.8, ls=":")

    for d in exclude_dates:
        ts = pd.Timestamp(d)
        if ts in px.index:
            ax.scatter([ts], [px.loc[ts, "WTI"]], s=42, color=C_MAGENTA,
                       zorder=5, edgecolor=C_SURFACE, linewidth=1.2)
            ax.annotate(
                "2020-04-20: WTI settles at -$37.63\n(Cushing storage crisis at expiry)\n"
                "excluded from ALL parameter estimation",
                xy=(ts, px.loc[ts, "WTI"]), xytext=(-40, 58),
                textcoords="offset points", fontsize=8, color=C_TEXT,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=C_MAGENTA, lw=1.1),
            )
    ax.set_title("Front-month crude futures, daily settlement")
    ax.set_ylabel("$ / bbl")
    ax.legend(loc="upper left", ncol=2)
    ax.margins(x=0.01)
    return _save(fig, "01_prices.png")


# ==========================================================================
def plot_spread(spread, beta, alpha, annotate_regimes=True):
    """The full-sample cointegrating residual with regime shading."""
    fig, ax = plt.subplots(figsize=(11, 5.0))

    # Clip the y-axis to the body of the distribution. The 2020-04-20 print
    # drags the spread to about -$63, which would compress 19 years of data
    # into the top third of the panel and hide everything the chart is for.
    # We clip the AXIS, never the data, and label the excursion explicitly so
    # the reader is told what is off-screen rather than silently shown less.
    lo_q, hi_q = spread.quantile(0.001), spread.quantile(0.999)
    pad = 0.35 * (hi_q - lo_q)
    y_lo, y_hi = lo_q - pad, hi_q + pad
    ax.set_ylim(y_lo, y_hi)

    if annotate_regimes:
        for i, (lo, hi, label, colour) in enumerate(REGIMES):
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
            ax.axvspan(lo, hi, color=colour, zorder=0)
            # Stagger the labels vertically: the later regimes are short and
            # their captions would otherwise collide.
            y = y_hi - (0.04 + 0.13 * (i % 2)) * (y_hi - y_lo)
            ax.text(lo + (hi - lo) / 2, y, label, ha="center",
                    va="top", fontsize=7.2, color=C_TEXT2, zorder=2)

    ax.plot(spread.index, spread.values, color=C_BLUE, lw=0.85, zorder=3,
            label="spread = WTI - (%.2f + %.3f x Brent)" % (alpha, beta))
    mu, sd = spread.mean(), spread.std()
    ax.axhline(mu, color=C_TEXT2, lw=1.0, zorder=4)
    for k, ls in ((2, "--"), (-2, "--")):
        ax.axhline(mu + k * sd, color=C_ORANGE, lw=1.0, ls=ls, zorder=4,
                   label="full-sample mean +/- 2 sd" if k == 2 else None)
    # Tell the reader about the clipped observation instead of hiding it.
    worst = spread.idxmin()
    if spread.min() < y_lo:
        ax.annotate("2020-04-20: spread reaches $%.2f\n"
                    "(off-scale; the negative WTI settlement --\n"
                    "excluded from all parameter fitting)" % spread.min(),
                    xy=(worst, y_lo), xytext=(-8, 34),
                    textcoords="offset points", fontsize=7.6, color=C_TEXT,
                    ha="right",
                    arrowprops=dict(arrowstyle="->", color=C_MAGENTA, lw=1.1))

    ax.set_title("Full-sample cointegrating residual (exploratory only -- the "
                 "backtest re-estimates this on every fold)")
    ax.set_ylabel("$ / bbl")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.margins(x=0.01)
    return _save(fig, "02_spread_regimes.png")


# ==========================================================================
def plot_rolling_half_life(hl_naive, hl_arma=None, folds=None):
    """Rolling half-life, showing how unstable the mean-reversion speed is."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(hl_naive.index, hl_naive["half_life"], color=C_BLUE, lw=1.1,
            label="AR(1) half-life (1y rolling)")
    if hl_arma is not None:
        ax.plot(hl_arma.index, hl_arma["half_life"], color=C_AQUA, lw=1.1,
                label="ARMA(1,1) half-life -- noise-robust")
    ax.axhline(config.WF.test_days, color=C_ORANGE, lw=1.0, ls="--",
               label="test-window length (%d d): above this, a trade cannot "
                     "be expected to close in-fold" % config.WF.test_days)
    ax.set_ylim(0, min(300, np.nanmax(hl_naive["half_life"]) * 1.15 if
                       np.isfinite(np.nanmax(hl_naive["half_life"])) else 300))
    ax.set_title("Rolling mean-reversion half-life of the spread")
    ax.set_ylabel("trading days")
    ax.legend(loc="upper left", fontsize=8.2)
    ax.margins(x=0.01)
    return _save(fig, "03_rolling_half_life.png")


# ==========================================================================
def plot_oos_pnl(daily, folds=None):
    """Cumulative out-of-sample P&L, gross vs net of costs."""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    cg = daily["gross_pnl"].cumsum()
    cn = daily["net_pnl"].cumsum()
    cc = daily["total_cost"].cumsum()

    ax.plot(cg.index, cg.values, color=C_BLUE, lw=1.4, label="Gross P&L")
    ax.plot(cn.index, cn.values, color=C_ORANGE, lw=1.4, label="Net of all costs")
    ax.fill_between(cg.index, cn.values, cg.values, color=C_ORANGE, alpha=0.11,
                    lw=0)
    ax.axhline(0, color=C_TEXT2, lw=0.9)

    ax.annotate("cumulative cost drag: $%.2f" % cc.iloc[-1],
                xy=(cg.index[-1], (cg.iloc[-1] + cn.iloc[-1]) / 2),
                xytext=(-160, 6), textcoords="offset points",
                fontsize=8.2, color=C_TEXT2)
    ax.text(cn.index[-1], cn.iloc[-1], "  net $%.2f" % cn.iloc[-1],
            fontsize=8.6, color=C_ORANGE, va="center")
    ax.text(cg.index[-1], cg.iloc[-1], "  gross $%.2f" % cg.iloc[-1],
            fontsize=8.6, color=C_BLUE, va="center")

    ax.set_title("Stitched out-of-sample P&L (no training days, "
                 "no overlapping test windows)")
    ax.set_ylabel("cumulative $ / bbl")
    ax.legend(loc="upper left")

    # position panel: makes the low duty cycle visible
    ax2.fill_between(daily.index, 0, daily["position"], color=C_VIOLET,
                     alpha=0.55, lw=0, step="mid")
    ax2.set_ylabel("position")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["short", "flat", "long"], fontsize=8)
    ax2.set_ylim(-1.4, 1.4)
    ax2.margins(x=0.01)
    return _save(fig, "04_oos_pnl.png")


# ==========================================================================
def plot_drawdown(daily):
    """Underwater chart of the net cumulative P&L."""
    cum = daily["net_pnl"].cumsum()
    dd = cum - cum.cummax()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(dd.index, dd.values, 0, color=C_ORANGE, alpha=0.35, lw=0)
    ax.plot(dd.index, dd.values, color=C_ORANGE, lw=1.0)
    worst = dd.idxmin()
    ax.scatter([worst], [dd.min()], color=C_MAGENTA, zorder=5, s=35,
               edgecolor=C_SURFACE, linewidth=1.1)
    ax.annotate("max drawdown $%.2f/bbl\n%s" % (dd.min(), worst.date()),
                xy=(worst, dd.min()), xytext=(12, 16),
                textcoords="offset points", fontsize=8.4, color=C_TEXT,
                arrowprops=dict(arrowstyle="->", color=C_MAGENTA, lw=1.0))
    ax.set_title("Out-of-sample drawdown (net of costs)")
    ax.set_ylabel("$ / bbl below prior peak")
    ax.margins(x=0.01)
    return _save(fig, "05_drawdown.png")


# ==========================================================================
def plot_fold_diagnostics(folds):
    """Per-fold hedge ratio, half-life and trade/skip decision."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True)
    x = folds["test_start"]

    ax = axes[0]
    ax.plot(x, folds["beta"], color=C_BLUE, lw=1.3, marker="o", ms=3.5,
            label="hedge ratio (beta), re-estimated each fold")
    ax.axhline(1.0, color=C_TEXT2, lw=0.9, ls=":")
    ax.set_ylabel("beta")
    ax.set_title("Walk-forward fold diagnostics (every value estimated on "
                 "training data only)")
    ax.legend(loc="lower right", fontsize=8.4)

    ax = axes[1]
    ax.plot(x, folds["half_life"], color=C_BLUE, lw=1.2, marker="o", ms=3.2,
            label="AR(1) half-life")
    if "half_life_arma" in folds:
        ax.plot(x, folds["half_life_arma"], color=C_AQUA, lw=1.2, marker="s",
                ms=3.2, label="ARMA(1,1) half-life (noise-robust)")
    ax.axhline(config.WF.test_days, color=C_ORANGE, lw=1.0, ls="--",
               label="test window (%d d)" % config.WF.test_days)
    ax.set_ylabel("half-life (days)")
    ax.legend(loc="upper right", fontsize=8.2)

    ax = axes[2]
    colours = [C_AQUA if t else C_MAGENTA for t in folds["tradeable"]]
    ax.bar(x, folds["eg_pvalue"], width=100, color=colours)
    ax.axhline(0.10, color=C_TEXT2, lw=1.0, ls="--")
    ax.text(x.iloc[0], 0.115, "gate: trade only if training EG p < 0.10",
            fontsize=8.2, color=C_TEXT2)
    ax.set_ylabel("EG p-value\n(training window)")
    ax.set_ylim(0, 1.0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_AQUA, label="fold traded"),
                       Patch(color=C_MAGENTA, label="fold stood aside")],
              loc="upper right", fontsize=8.4, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, "06_fold_diagnostics.png")


# ==========================================================================
def plot_case_study_2022(daily, px, start="2021-11-01", end="2023-06-30"):
    """Zoom on the 2022 Brent-WTI dislocation."""
    sl = daily.loc[start:end]
    pxs = px.loc[start:end]
    if len(sl) == 0:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1, 2]})

    ax = axes[0]
    ax.plot(pxs.index, pxs["BRENT"], color=C_BLUE, lw=1.1, label="Brent")
    ax.plot(pxs.index, pxs["WTI"], color=C_ORANGE, lw=1.1, label="WTI")
    ax.axvline(pd.Timestamp("2022-02-24"), color=C_TEXT2, lw=1.0, ls="--")
    ax.text(pd.Timestamp("2022-02-24"), pxs.max().max(), " invasion of Ukraine",
            fontsize=8.2, color=C_TEXT2, va="top")
    ax.set_ylabel("$ / bbl")
    ax.set_title("Case study: the 2022 Brent-WTI dislocation")
    ax.legend(loc="upper right", fontsize=8.4, ncol=2)

    ax = axes[1]
    ax.plot(sl.index, sl["spread"], color=C_BLUE, lw=1.2, label="spread (fold beta)")
    ax.axhline(0, color=C_TEXT2, lw=0.8, ls=":")
    ax.set_ylabel("spread $/bbl")
    ax.legend(loc="upper right", fontsize=8.4)

    ax = axes[2]
    ax.plot(sl.index, sl["z"], color=C_VIOLET, lw=1.1, label="z-score")
    for k in (2, -2):
        ax.axhline(k, color=C_ORANGE, lw=0.9, ls="--")
    for k in (0.5, -0.5):
        ax.axhline(k, color=C_AQUA, lw=0.9, ls=":")
    ax.set_ylabel("z")
    ax.legend(loc="upper right", fontsize=8.4)

    ax = axes[3]
    ax.plot(sl.index, sl["net_pnl"].cumsum(), color=C_ORANGE, lw=1.4,
            label="cumulative net P&L in window")
    ax.axhline(0, color=C_TEXT2, lw=0.8)
    ax.fill_between(sl.index, 0, sl["position"] * abs(sl["net_pnl"].cumsum()).max() * 0.25,
                    color=C_VIOLET, alpha=0.18, lw=0, step="mid",
                    label="position (scaled)")
    ax.set_ylabel("$ / bbl")
    ax.legend(loc="lower left", fontsize=8.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return _save(fig, "07_case_study_2022.png")


# ==========================================================================
def plot_cost_sensitivity(table):
    """Net Sharpe as a function of assumed per-leg transaction cost."""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(table["cost_per_leg_side"] * 100, table["sharpe_net"],
            color=C_BLUE, lw=1.6, marker="o", ms=5, label="net Sharpe")
    ax.axhline(0, color=C_TEXT2, lw=0.9)
    base = config.COSTS.cost_per_leg_side * 100
    ax.axvline(base, color=C_ORANGE, lw=1.0, ls="--")
    ax.text(base, ax.get_ylim()[1], " base case (%.0fc)" % base, fontsize=8.4,
            color=C_ORANGE, va="top")
    for _, r in table.iterrows():
        ax.annotate("%.2f" % r["sharpe_net"],
                    (r["cost_per_leg_side"] * 100, r["sharpe_net"]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center", color=C_TEXT2)
    ax.set_xlabel("assumed cost per leg per side (cents / bbl)")
    ax.set_ylabel("annualised net Sharpe")
    ax.set_title("Sensitivity of the result to the cost assumption")
    ax.legend(loc="lower left", fontsize=8.4)
    return _save(fig, "08_cost_sensitivity.png")


# ==========================================================================
def plot_validation_half_life(df):
    """Estimator recovery check: estimated vs true half-life."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(df["true_half_life"], df["true_half_life"], color=C_TEXT2, lw=1.0,
            ls="--", label="perfect recovery (45 degrees)")
    ax.plot(df["true_half_life"], df["mean_est_half_life"], color=C_BLUE,
            lw=1.5, marker="o", ms=5, label="exact OU inversion")
    ax.plot(df["true_half_life"], df["mean_est_half_life_euler"],
            color=C_YELLOW, lw=1.3, marker="s", ms=4.5,
            label="Euler approximation")
    ax.set_xlabel("true half-life used to simulate (days)")
    ax.set_ylabel("mean estimated half-life (days)")
    ax.set_title("Estimator validated against synthetic data with known truth")
    ax.legend(loc="upper left", fontsize=8.4)
    return _save(fig, "09_validation_half_life.png")
