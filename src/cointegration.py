"""
Cointegration engine: Engle-Granger (both directions) and Johansen.

Design notes that a reviewer will ask about
-------------------------------------------
* Engle-Granger residuals are NOT an observed series -- they are the output of
  a first-stage OLS whose coefficient was chosen to make them look as
  stationary as possible. Applying standard Dickey-Fuller critical values to
  them therefore over-rejects the null of no cointegration. We report the
  naive ADF statistic (because it is what most write-ups show) AND the
  MacKinnon cointegration p-value from statsmodels.tsa.stattools.coint, which
  uses the correct critical values for an ESTIMATED cointegrating vector. The
  second is the one we act on.

* Engle-Granger is not symmetric: regressing WTI on Brent and regressing Brent
  on WTI give different residuals and can give different verdicts in finite
  samples. Reporting only the direction that "works" is a classic silent
  specification search, so we always run and report both.

* Johansen is a systems/VAR-based test that treats both series symmetrically
  and can identify more than one cointegrating relation. Its asymptotic
  critical values assume iid Gaussian innovations -- an assumption daily crude
  returns violate badly (volatility clustering, fat tails). See decide_rank
  for how we handle the resulting over-rejection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from . import config


# ==========================================================================
# Engle-Granger
# ==========================================================================
class EGResult(dict):
    """Dict with attribute access, so results are easy to log and tabulate."""

    __getattr__ = dict.__getitem__


def engle_granger(y, x, y_name=None, x_name=None, autolag=None):
    """Two-step Engle-Granger with y regressed on x.

    `autolag` controls lag augmentation in the second-stage unit-root test and
    is a genuine specification choice, not a detail -- on the 2021-2024
    training windows of this dataset it moves the p-value from ~0.2 (AIC,
    which selects 14-16 lags) to ~0.0003 (BIC, which selects far fewer). We
    default to config.EG_AUTOLAG (AIC), the MORE CONSERVATIVE of the two here
    because it declares fewer windows tradeable, and report the BIC variant as
    a documented robustness case rather than choosing the flattering one. See
    the README section on specification sensitivity.

    Step 1: y_t = alpha + beta * x_t + e_t   (OLS on the LEVELS)
    Step 2: test e_t for a unit root.

    Both inputs must already be masked to estimation-eligible observations.
    """
    autolag = autolag or config.EG_AUTOLAG
    y_name = y_name or (y.name or "y")
    x_name = x_name or (x.name or "x")
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)

    X = sm.add_constant(xv)
    ols = sm.OLS(yv, X).fit()
    alpha, beta = float(ols.params[0]), float(ols.params[1])
    resid = pd.Series(yv - (alpha + beta * xv), index=y.index, name="resid")

    # Naive ADF on the residual. We use regression="n" because OLS residuals
    # have exactly zero sample mean by construction, so including a constant
    # in the ADF regression would estimate a parameter already known to be
    # zero and would cost power.
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
        resid.values, regression="n", autolag=autolag.upper(),
        result_object=False
    )

    # Correct test: MacKinnon critical values for an ESTIMATED cointegrating
    # vector. This is the number we use for the go/no-go decision.
    cr = coint(yv, xv, trend="c", autolag=autolag)

    return EGResult(
        spec=y_name + " ~ " + x_name,
        y_name=y_name,
        x_name=x_name,
        alpha=alpha,
        beta=beta,
        r_squared=float(ols.rsquared),
        resid=resid,
        adf_stat_naive=float(adf_stat),
        adf_p_naive=float(adf_p),
        adf_lags=int(adf_lags),
        adf_nobs=int(adf_nobs),
        adf_crit_naive={k: float(v) for k, v in adf_crit.items()},
        eg_stat=float(cr.coint_t),
        eg_pvalue=float(cr.pvalue),
        eg_crit={
            "1%": float(cr.critical_values[0]),
            "5%": float(cr.critical_values[1]),
            "10%": float(cr.critical_values[2]),
        },
    )


def engle_granger_both(df, a="WTI", b="BRENT", autolag=None):
    """Run EG in both directions and return (a_on_b, b_on_a)."""
    return (
        engle_granger(df[a], df[b], a, b, autolag=autolag),
        engle_granger(df[b], df[a], b, a, autolag=autolag),
    )


# ==========================================================================
# Unit-root checks on the levels themselves
# ==========================================================================
def adf_levels(df):
    """ADF on each price level and on its first difference.

    Cointegration is only a meaningful claim if each series is individually
    I(1): non-stationary in levels, stationary in differences. If a level were
    already stationary, a "cointegrating" combination would be trivial. This
    is the precondition check that many write-ups skip.
    """
    rows = []
    for col in df.columns:
        pairs = (("level", df[col]), ("first difference", df[col].diff().dropna()))
        for label, s in pairs:
            # trend="c": prices have a non-zero mean, but we do not impose a
            # deterministic time trend (see config.JOHANSEN_DET_ORDER note).
            stat, p, lags, nobs, crit, _ = adfuller(
                s.values, regression="c", autolag="AIC", result_object=False
            )
            rows.append(
                {
                    "series": col,
                    "transform": label,
                    "adf_stat": stat,
                    "p_value": p,
                    "lags": lags,
                    "nobs": nobs,
                    "cv_5pct": crit["5%"],
                    "reject_unit_root_5pct": p < 0.05,
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# VAR lag-order selection
# ==========================================================================
def select_var_lag(levels, maxlags=config.MAX_VAR_LAGS):
    """Choose the VAR lag order p on the LEVELS, via information criteria.

    Why on the levels and not the differences? The Johansen procedure is
    derived from a levels VAR(p) rewritten in VECM form; its k_ar_diff
    argument is the number of lagged DIFFERENCES in that VECM, which equals
    p - 1. Selecting p on the levels and passing p-1 is the internally
    consistent choice, and it is why we do not simply hardcode k_ar_diff=1.

    Why BIC/HQIC and not AIC? AIC is not consistent for lag-order selection --
    it over-parameterises asymptotically. With ~4,700 daily observations we
    have ample data, and an over-long lag specification eats degrees of
    freedom and distorts the Johansen trace statistic. BIC is consistent;
    HQIC sits between the two and is reported as a cross-check.
    """
    model = VAR(levels.values)
    sel = model.select_order(maxlags=maxlags)
    table = {}
    for crit in ("aic", "bic", "hqic", "fpe"):
        try:
            table[crit] = int(getattr(sel, crit))
        except Exception:
            table[crit] = None
    kb = max(int(table["bic"]) - 1, 0) if table["bic"] is not None else None
    kh = max(int(table["hqic"]) - 1, 0) if table["hqic"] is not None else None
    return {
        "selected": sel,
        "p_by_criterion": table,
        "p_bic": table["bic"],
        "p_hqic": table["hqic"],
        "k_ar_diff_bic": kb,
        "k_ar_diff_hqic": kh,
    }


# ==========================================================================
# Johansen
# ==========================================================================
def johansen(levels, det_order=config.JOHANSEN_DET_ORDER, k_ar_diff=1):
    """Run the Johansen trace test and return a tidy result dict."""
    res = coint_johansen(levels.values, det_order, k_ar_diff)
    ci = config.JOHANSEN_CRIT_IDX
    n = levels.shape[1]
    trace = pd.DataFrame(
        {
            "hypothesis": ["r<=" + str(i) for i in range(n)],
            "trace_stat": res.lr1,
            "cv_90": res.cvt[:, 0],
            "cv_95": res.cvt[:, 1],
            "cv_99": res.cvt[:, 2],
        }
    )
    trace["reject"] = trace["trace_stat"] > res.cvt[:, ci]
    # Margin as a percentage of the critical value: how comfortably did it
    # clear? A 2% margin and a 200% margin are very different evidence.
    trace["margin_pct_of_cv"] = (
        100.0 * (trace["trace_stat"] - res.cvt[:, ci]) / res.cvt[:, ci]
    )

    maxeig = pd.DataFrame(
        {
            "hypothesis": ["r=" + str(i) for i in range(n)],
            "max_eig_stat": res.lr2,
            "cv_90": res.cvm[:, 0],
            "cv_95": res.cvm[:, 1],
            "cv_99": res.cvm[:, 2],
        }
    )
    maxeig["reject"] = maxeig["max_eig_stat"] > res.cvm[:, ci]

    # Trace test: walk up from r<=0; rank = number of consecutive rejections.
    rank_mechanical = 0
    for rej in trace["reject"]:
        if rej:
            rank_mechanical += 1
        else:
            break

    # First cointegrating vector, normalised so the first variable has
    # coefficient 1. The relation is then levels[:,0] - beta*levels[:,1] ~ I(0).
    v = res.evec[:, 0]
    v_norm = v / v[0]
    beta_johansen = -float(v_norm[1])

    return {
        "trace": trace,
        "max_eig": maxeig,
        "rank_mechanical": rank_mechanical,
        "evec_normalised": v_norm,
        "beta": beta_johansen,
        "eigenvalues": res.eig,
        "k_ar_diff": k_ar_diff,
        "det_order": det_order,
        "n_obs": len(levels),
    }


def johansen_lag_robustness(levels, lags=(1, 2, 3, 5, 8),
                            det_order=config.JOHANSEN_DET_ORDER):
    """Re-run Johansen across several k_ar_diff values.

    If the rank verdict flips with the lag length, the result is an artefact
    of the specification rather than a property of the data, and should be
    reported as such instead of quoting the most convenient lag.
    """
    rows = []
    for k in lags:
        j = johansen(levels, det_order, k)
        t = j["trace"]
        rows.append(
            {
                "k_ar_diff": k,
                "trace_r0": t.loc[0, "trace_stat"],
                "cv95_r0": t.loc[0, "cv_95"],
                "reject_r0": bool(t.loc[0, "reject"]),
                "trace_r1": t.loc[1, "trace_stat"],
                "cv95_r1": t.loc[1, "cv_95"],
                "reject_r1": bool(t.loc[1, "reject"]),
                "margin_r1_pct": t.loc[1, "margin_pct_of_cv"],
                "rank_mechanical": j["rank_mechanical"],
                "beta": j["beta"],
            }
        )
    return pd.DataFrame(rows)


def decide_rank(johansen_res, eg_results, levels, marginal_threshold_pct=25.0):
    """Turn the mechanical trace verdict into an economically defensible rank.

    Johansen's asymptotic critical values are derived under iid Gaussian
    innovations. Daily crude oil returns are neither: they are heavily
    fat-tailed and show strong volatility clustering. Both features inflate
    the trace statistic, so the test over-rejects and reports SPURIOUS extra
    cointegrating vectors. (This is easy to demonstrate -- run the test on
    simulated independent random walks with GARCH errors and the r<=1
    hypothesis is rejected far more often than 5% of the time; see
    validation.py, which does exactly that.)

    For a bivariate system, rank=2 has a very specific and very strong
    meaning: the 2-dimensional system is FULL rank, i.e. every linear
    combination of the two series is stationary, i.e. the WTI and Brent price
    LEVELS are each individually stationary. That is economically absurd for a
    commodity price that travelled from $147 to -$37 to $100, and it is
    directly contradicted by the level ADF tests. So we apply three guards:

      1. Statistical margin: if the r<=1 trace statistic clears its critical
         value by less than marginal_threshold_pct OF that critical value,
         treat the rejection as unreliable.
      2. Economic plausibility: rank=2 implies stationary levels. Check the
         level ADF results; if they fail to reject a unit root, rank=2 is
         rejected outright regardless of the trace statistic.
      3. Agreement with Engle-Granger: EG is a different (single-equation)
         route to the same question. Concordance raises confidence.
    """
    t = johansen_res["trace"]
    r1_margin = float(t.loc[1, "margin_pct_of_cv"])
    r1_reject = bool(t.loc[1, "reject"])
    r0_reject = bool(t.loc[0, "reject"])

    # Guard 2: are the raw levels stationary? If not, the rank cannot be 2.
    lvl = adf_levels(levels)
    lvl_only = lvl[lvl["transform"] == "level"]
    levels_stationary = bool(lvl_only["reject_unit_root_5pct"].all())

    eg_agree = all(e["eg_pvalue"] < 0.05 for e in eg_results)

    notes = []
    rank = johansen_res["rank_mechanical"]

    if rank >= 2:
        if not levels_stationary:
            notes.append(
                "Mechanical trace verdict was rank=2, which in a bivariate "
                "system implies BOTH price levels are stationary. The level "
                "ADF tests fail to reject a unit root for at least one "
                "series, so rank=2 is economically incoherent and is "
                "overridden to 1."
            )
            rank = 1
        elif r1_margin < marginal_threshold_pct:
            notes.append(
                "The r<=1 trace statistic cleared its 95 pct critical value "
                "by only %.1f pct of that critical value. Given the known "
                "upward bias of the trace statistic under fat-tailed, "
                "volatility-clustered innovations, this is treated as a "
                "spurious rejection and the rank is set to 1." % r1_margin
            )
            rank = 1

    if rank == 0 and eg_agree:
        notes.append(
            "Johansen failed to reject r<=0 at 95 pct but both Engle-Granger "
            "directions reject the null of no cointegration. The tests "
            "disagree -- the evidence for cointegration is weak and this "
            "should be treated as a marginal case."
        )

    if not notes:
        notes.append("Mechanical trace verdict accepted; no override required.")

    return {
        "rank_mechanical": johansen_res["rank_mechanical"],
        "rank_adopted": rank,
        "r0_reject": r0_reject,
        "r1_reject": r1_reject,
        "r1_margin_pct_of_cv": r1_margin,
        "levels_individually_stationary": levels_stationary,
        "eg_both_reject": eg_agree,
        "level_adf_table": lvl,
        "notes": notes,
    }


# ==========================================================================
# Spread construction
# ==========================================================================
def build_spread(px, alpha, beta, y="WTI", x="BRENT"):
    """spread_t = y_t - (alpha + beta * x_t).

    Interpretation for trading: hold +1 unit of y against -beta units of x.
    alpha is a constant, so it shifts the LEVEL of the spread but never
    affects its day-to-day CHANGE, and therefore never affects P&L; it only
    determines where the z-score sits.
    """
    return (px[y] - (alpha + beta * px[x])).rename("spread")
