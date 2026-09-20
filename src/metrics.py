"""
Performance metrics.

Everything in here is computed on STITCHED OUT-OF-SAMPLE days only. There is
no function in this module that takes training data, by design.

A note on the Sharpe ratio that matters for interpretation
----------------------------------------------------------
A Sharpe ratio estimated from a finite sample is itself a random variable with
a standard error. For iid returns the classic Lo (2002) approximation is

    SE(SR_annual) ~= sqrt( (1 + 0.5*SR^2) / n_years / 252 ) * sqrt(252)
                   = sqrt( (1 + 0.5*SR^2) / N )  * sqrt(252)

with N the number of daily observations. Over ~10 years of daily data the SE
on an annualised Sharpe is roughly 0.3, so a reported Sharpe of 0.8 and a
reported Sharpe of 1.3 are not reliably different from each other. We report
the confidence interval alongside the point estimate rather than quoting a
single number as if it were known exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def annualised_sharpe(returns, periods=config.TRADING_DAYS_PER_YEAR):
    """Annualised Sharpe of a daily return series (excess of a zero financing
    rate -- see the README for why that is the right benchmark here: a futures
    spread is close to self-financing, funded only through margin)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sharpe_confidence_interval(returns, periods=config.TRADING_DAYS_PER_YEAR,
                               conf=0.95):
    """Lo (2002) standard error and CI for the annualised Sharpe."""
    r = pd.Series(returns).dropna()
    n = len(r)
    sr = annualised_sharpe(r, periods)
    if not np.isfinite(sr) or n < 10:
        return {"sharpe": sr, "se": np.nan, "lo": np.nan, "hi": np.nan, "n": n}
    from scipy import stats
    zc = stats.norm.ppf(0.5 + conf / 2.0)
    se = np.sqrt((1.0 + 0.5 * sr ** 2) / n) * np.sqrt(periods)
    return {"sharpe": sr, "se": float(se),
            "lo": float(sr - zc * se), "hi": float(sr + zc * se), "n": n}


def max_drawdown_dollars(pnl):
    """Largest peak-to-trough decline of the CUMULATIVE $ P&L curve.

    Reported in $/bbl of WTI notional, i.e. in the same units as the spread,
    so it can be compared directly against the spread's own volatility.
    """
    cum = pd.Series(pnl).fillna(0.0).cumsum()
    peak = cum.cummax()
    dd = cum - peak
    if len(dd) == 0:
        return {"max_dd": np.nan, "trough_date": None, "peak_date": None,
                "recovery_date": None, "dd_series": dd}
    trough = dd.idxmin()
    peak_date = cum.loc[:trough].idxmax()
    after = cum.loc[trough:]
    rec = after[after >= cum.loc[peak_date]]
    return {
        "max_dd": float(dd.min()),
        "trough_date": trough,
        "peak_date": peak_date,
        "recovery_date": rec.index[0] if len(rec) else None,
        "dd_series": dd,
    }


def max_drawdown_pct(returns):
    """Largest peak-to-trough decline of the compounded return index."""
    r = pd.Series(returns).fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return {"max_dd_pct": float(dd.min()) if len(dd) else np.nan,
            "dd_series": dd, "equity": eq}


def summarise(daily, trades, half_life_ref=None,
              periods=config.TRADING_DAYS_PER_YEAR, label="out-of-sample"):
    """Full metric block for a stitched OOS daily frame.

    `daily` must contain gross_ret / net_ret / gross_pnl / net_pnl / position.
    `half_life_ref` is the average OU half-life across the training folds, used
    only to sanity-check the realised holding period.
    """
    d = daily.copy()
    n = len(d)
    years = n / periods

    net_r = d["net_ret"]
    gross_r = d["gross_ret"]

    sr_net = sharpe_confidence_interval(net_r, periods)
    sr_gross = sharpe_confidence_interval(gross_r, periods)

    dd_d = max_drawdown_dollars(d["net_pnl"])
    dd_p = max_drawdown_pct(net_r)

    in_mkt = (d["position"] != 0)

    if trades is not None and len(trades):
        hit = float((trades["net_pnl"] > 0).mean())
        hit_gross = float((trades["gross_pnl"] > 0).mean())
        avg_hold = float(trades["holding_days"].mean())
        med_hold = float(trades["holding_days"].median())
        avg_win = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean()) \
            if (trades["net_pnl"] > 0).any() else np.nan
        avg_loss = float(trades.loc[trades["net_pnl"] <= 0, "net_pnl"].mean()) \
            if (trades["net_pnl"] <= 0).any() else np.nan
        n_trades = int(len(trades))
    else:
        hit = hit_gross = avg_hold = med_hold = avg_win = avg_loss = np.nan
        n_trades = 0

    res = {
        "label": label,
        "start": d.index.min(),
        "end": d.index.max(),
        "n_days": n,
        "n_years": years,
        "n_trades": n_trades,
        "trades_per_year": n_trades / years if years else np.nan,
        "time_in_market_pct": 100.0 * float(in_mkt.mean()),

        "gross_pnl_total": float(d["gross_pnl"].sum()),
        "total_costs": float(d["total_cost"].sum()),
        "trade_costs": float(d["trade_cost"].sum()),
        "roll_costs": float(d["roll_cost"].sum()),
        "net_pnl_total": float(d["net_pnl"].sum()),
        "cost_pct_of_gross": (100.0 * d["total_cost"].sum() / d["gross_pnl"].sum()
                              if d["gross_pnl"].sum() != 0 else np.nan),

        "sharpe_gross": sr_gross["sharpe"],
        "sharpe_net": sr_net["sharpe"],
        "sharpe_net_se": sr_net["se"],
        "sharpe_net_ci_lo": sr_net["lo"],
        "sharpe_net_ci_hi": sr_net["hi"],

        "ann_return_pct": 100.0 * float(net_r.mean()) * periods,
        "ann_vol_pct": 100.0 * float(net_r.std(ddof=1)) * np.sqrt(periods),

        "hit_rate_net": hit,
        "hit_rate_gross": hit_gross,
        "avg_holding_days": avg_hold,
        "median_holding_days": med_hold,
        "avg_win": avg_win,
        "avg_loss": avg_loss,

        "max_dd_dollars": dd_d["max_dd"],
        "max_dd_peak": dd_d["peak_date"],
        "max_dd_trough": dd_d["trough_date"],
        "max_dd_recovery": dd_d["recovery_date"],
        "max_dd_pct": 100.0 * dd_p["max_dd_pct"],

        "half_life_ref": half_life_ref,
        "hold_vs_halflife_ratio": (avg_hold / half_life_ref
                                   if half_life_ref and np.isfinite(avg_hold)
                                   else np.nan),
        "sharpe_suspicious": bool(np.isfinite(sr_net["sharpe"])
                                  and sr_net["sharpe"] > config.SUSPICIOUS_SHARPE),
    }
    return res


def format_summary(res):
    """Human-readable metric block."""
    L = []
    A = L.append
    A("-" * 68)
    A("PERFORMANCE: %s" % res["label"])
    A("-" * 68)
    A("Period                  : %s -> %s  (%.1f years, %d trading days)"
      % (res["start"].date(), res["end"].date(), res["n_years"], res["n_days"]))
    A("Trades                  : %d  (%.1f per year)"
      % (res["n_trades"], res["trades_per_year"]))
    A("Time in market          : %.1f%% of days" % res["time_in_market_pct"])
    A("")
    A("P&L ($/bbl of WTI notional)")
    A("  Gross P&L             : %+8.2f" % res["gross_pnl_total"])
    A("  - transaction costs   : %8.2f" % res["trade_costs"])
    A("  - futures roll costs  : %8.2f" % res["roll_costs"])
    A("  = Net P&L             : %+8.2f" % res["net_pnl_total"])
    A("  Costs as %% of gross   : %.1f%%" % res["cost_pct_of_gross"])
    A("")
    A("Risk-adjusted")
    A("  Sharpe (gross)        : %6.3f" % res["sharpe_gross"])
    A("  Sharpe (net)          : %6.3f   95%% CI [%.3f, %.3f]  (SE %.3f)"
      % (res["sharpe_net"], res["sharpe_net_ci_lo"],
         res["sharpe_net_ci_hi"], res["sharpe_net_se"]))
    A("  Annualised return     : %6.2f%%" % res["ann_return_pct"])
    A("  Annualised volatility : %6.2f%%" % res["ann_vol_pct"])
    A("  Max drawdown          : %6.2f $/bbl   (%.2f%% of notional)"
      % (res["max_dd_dollars"], res["max_dd_pct"]))
    A("    peak %s -> trough %s -> recovery %s"
      % (res["max_dd_peak"].date() if res["max_dd_peak"] is not None else "n/a",
         res["max_dd_trough"].date() if res["max_dd_trough"] is not None else "n/a",
         res["max_dd_recovery"].date() if res["max_dd_recovery"] is not None else "not recovered"))
    A("")
    A("Trade statistics")
    A("  Hit rate (net)        : %5.1f%%" % (100 * res["hit_rate_net"]))
    A("  Hit rate (gross)      : %5.1f%%" % (100 * res["hit_rate_gross"]))
    A("  Avg holding period    : %5.1f days  (median %.1f)"
      % (res["avg_holding_days"], res["median_holding_days"]))
    if res["half_life_ref"]:
        A("  Mean training half-life: %.1f days -> holding/half-life = %.2f"
          % (res["half_life_ref"], res["hold_vs_halflife_ratio"]))
    A("  Avg winning trade     : %+.3f $/bbl" % res["avg_win"])
    A("  Avg losing trade      : %+.3f $/bbl" % res["avg_loss"])
    if res["sharpe_suspicious"]:
        A("")
        A("  *** WARNING: net Sharpe exceeds %.1f. Treat as a leakage "
          "suspect and investigate before reporting. ***"
          % config.SUSPICIOUS_SHARPE)
    A("-" * 68)
    return "\n".join(L)
