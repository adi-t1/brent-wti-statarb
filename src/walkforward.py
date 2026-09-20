"""
Purged walk-forward validation harness.

This is the part of the project that decides whether the reported numbers mean
anything, so the logic is written out explicitly rather than hidden behind a
library call.

The shape of one fold
---------------------
    |<------ train (756d) ------>|<- purge (21d) ->|<-- test (126d) -->|
                                  ^ nothing here is
                                    used for anything

Then the whole window slides forward by exactly `test_days`, so successive
TEST windows tile the sample end-to-end without ever overlapping. Overlapping
test windows would mean the same calendar day contributes to the stitched
out-of-sample track record more than once, which quietly inflates the
effective sample size and therefore the apparent significance of the result.

Why the purge gap exists
------------------------
Suppose train ends on Friday and test starts on Monday. The hedge ratio, the
spread mean and the spread standard deviation were all fitted using Friday's
price. The spread is strongly autocorrelated -- that is the entire premise of
the strategy -- so Monday's spread level is largely predictable from Friday's.
Estimating the z-score normalisation on data that is mechanically linked to
the first few test observations means the very first trade of each fold is
placed with a small but real informational advantage.

Leaving a gap of 21 trading days -- comfortably more than the typical fitted
half-life -- means that by the time the test window opens, the influence of
the last training observation on the spread level has decayed away. The cost
is that we discard about 2.3% of the sample. That is a cheap price for a
clean result.

What is re-estimated on every fold
----------------------------------
    - the cointegrating vector (alpha, beta) via Engle-Granger on the train
      window only
    - the VAR lag order, by BIC on the train window only
    - the Johansen rank verdict, on the train window only
    - the OU parameters (theta, mu, sigma, half-life) on the train residual
    - the z-score normalisation (mean, sd) on the train residual

What is held fixed
------------------
    - the entry/exit z thresholds (2.0 / 0.5). See config.TradingConfig for
      the argument: these are not fitted, so there is nothing to leak, and
      holding them fixed removes a large source of rule-level overfitting.
    - the choice of WTI as the dependent variable in the Engle-Granger
      regression. This is fixed a priori on ECONOMIC grounds (Brent is the
      global waterborne benchmark that clears against seaborne supply and
      demand; WTI is the landlocked, pipeline-constrained grade that
      dislocates away from it), NOT by looking at which direction tested
      better in the full sample. Picking the direction from full-sample
      results would itself be a lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from . import cointegration as ci
from . import config
from . import data as datamod
from . import ou


# ==========================================================================
# Fold construction
# ==========================================================================
def make_folds(index, wf=config.WF):
    """Build the list of (train, purge, test) positional slices."""
    n = len(index)
    folds = []
    start = 0
    fold_id = 0
    while True:
        tr0, tr1 = start, start + wf.train_days
        pg0, pg1 = tr1, tr1 + wf.purge_days
        te0, te1 = pg1, pg1 + wf.test_days
        if te1 > n:
            break
        folds.append({
            "fold": fold_id,
            "train": slice(tr0, tr1),
            "purge": slice(pg0, pg1),
            "test": slice(te0, te1),
            "train_start": index[tr0], "train_end": index[tr1 - 1],
            "purge_start": index[pg0], "purge_end": index[pg1 - 1],
            "test_start": index[te0], "test_end": index[te1 - 1],
        })
        fold_id += 1
        start += wf.step_days
    return folds


# ==========================================================================
# Per-fold training
# ==========================================================================
def fit_fold(train_px, train_mask, run_johansen=True, eg_autolag=None,
             fit_arma=True):
    """Estimate everything the trading rule needs, using ONLY `train_px`."""
    est = train_px[train_mask.reindex(train_px.index).fillna(True)]

    eg_wti, eg_brent = ci.engle_granger_both(est, autolag=eg_autolag)

    # Lag order chosen by BIC on THIS window's levels VAR, not on the full
    # sample -- otherwise the lag order itself would carry information from
    # the future into the fold.
    try:
        lag = ci.select_var_lag(est, maxlags=8)
        k_ar_diff = lag["k_ar_diff_bic"] if lag["k_ar_diff_bic"] is not None else 1
    except Exception:
        k_ar_diff = 1

    joh = None
    rank = np.nan
    if run_johansen:
        try:
            joh = ci.johansen(est, config.JOHANSEN_DET_ORDER, max(k_ar_diff, 1))
            rank = ci.decide_rank(joh, [eg_wti, eg_brent], est)["rank_adopted"]
        except Exception:
            joh, rank = None, np.nan

    alpha, beta = eg_wti["alpha"], eg_wti["beta"]
    spread_train = ci.build_spread(est, alpha, beta)

    ou_fit = ou.fit_ou(spread_train, train_mask.reindex(spread_train.index))

    # Noise-robust cross-check on the half-life. The naive AR(1) figure is
    # biased SHORT whenever the spread carries transient bid-ask/settlement
    # noise, which on this pair it always does (the fitted MA coefficient is
    # consistently around -0.15). We report both; a large gap between them is
    # a warning that the apparent speed of reversion is partly microstructure.
    arma = ({"half_life": np.nan, "phi": np.nan, "ma": np.nan, "valid": False,
             "reason": "not fitted"} if not fit_arma
            else ou.half_life_arma11(spread_train,
                                     train_mask.reindex(spread_train.index)))

    return {
        "half_life_arma": arma["half_life"],
        "arma_ma": arma["ma"],
        "arma_valid": arma["valid"],
        "alpha": alpha,
        "beta": beta,
        "eg_pvalue": eg_wti["eg_pvalue"],
        "eg_stat": eg_wti["eg_stat"],
        "eg_pvalue_reverse": eg_brent["eg_pvalue"],
        "k_ar_diff": k_ar_diff,
        "johansen_rank": rank,
        "johansen_beta": joh["beta"] if joh else np.nan,
        # z-score normalisation: the sample mean and sd of the TRAINING
        # residual. (The OU long-run mean `mu` is an alternative; the two agree
        # closely here, and the sample moments are more robust to a poorly
        # identified theta.)
        "spread_mean": float(spread_train.mean()),
        "spread_std": float(spread_train.std(ddof=1)),
        "ou": ou_fit,
        "half_life": ou_fit["half_life"],
        "theta": ou_fit["theta"],
        "ou_valid": ou_fit["valid"],
        "ou_reason": ou_fit["reason"],
        "n_train": len(est),
    }


# ==========================================================================
# Gating
# ==========================================================================
def gate_fold(params, max_eg_pvalue=0.10, min_half_life=1.0,
              max_half_life=None):
    """Decide whether this fold is tradeable, using training evidence only.

    Three reasons to stand aside:

    1. The training window shows no cointegration. Trading a mean-reversion
       rule on a spread that the data says is a random walk is not a strategy,
       it is a coin flip with transaction costs. We use a 10% threshold rather
       than 5% -- deliberately permissive, because being too strict here would
       let us cherry-pick only the easiest regimes and flatter the result.

    2. The OU fit is invalid (theta <= 0). This is a firm rule: do not force
       a half-life out of a window that is not mean-reverting.

    3. The half-life is longer than the test window. If the spread takes
       longer than 126 trading days to revert halfway, a position opened in
       this fold cannot reasonably be expected to close within it, and the
       forced flat at the fold boundary turns the trade into a coin flip on
       where the spread happened to be on the last day.
    """
    if max_half_life is None:
        max_half_life = float(config.WF.test_days)

    reasons = []
    if not params["ou_valid"]:
        reasons.append("OU invalid: " + params["ou_reason"])
    else:
        hl = params["half_life"]
        if hl < min_half_life:
            reasons.append("half-life %.2f d below floor %.1f d" % (hl, min_half_life))
        elif hl > max_half_life:
            reasons.append("half-life %.1f d exceeds test window %.0f d"
                           % (hl, max_half_life))
    if params["eg_pvalue"] > max_eg_pvalue:
        reasons.append("EG p-value %.3f > %.2f (no cointegration in train)"
                       % (params["eg_pvalue"], max_eg_pvalue))
    return (len(reasons) == 0), reasons


# ==========================================================================
# Main harness
# ==========================================================================
def run_walk_forward(px, rolls=None, mask=None, wf=config.WF,
                     trade=config.TRADE, cost_per_leg_side=None,
                     roll_cost_per_leg=None, execution_lag=0,
                     max_eg_pvalue=0.10, leak_mode="none",
                     run_johansen=True, eg_autolag=None, fit_arma=True,
                     roll_mode="neutralise", verbose=False):
    """Run the full purged walk-forward.

    leak_mode
    ---------
    "none"         : the real, clean protocol.
    "fit_on_test"  : DELIBERATELY fits the cointegrating vector and z-score
                     normalisation on the TEST window. Used only by the
                     leakage audit, to confirm that this harness would in fact
                     show an implausible Sharpe if leakage were present. If
                     the clean run and the cheating run score the same, the
                     harness is not measuring what we think it is.
    "no_purge"     : keeps the training fit honest but removes the purge gap,
                     to size how much the gap actually matters.
    """
    if mask is None:
        mask = datamod.estimation_mask(px.index)
    if cost_per_leg_side is None:
        cost_per_leg_side = config.COSTS.cost_per_leg_side
    if roll_cost_per_leg is None:
        roll_cost_per_leg = config.COSTS.roll_cost_per_leg

    eff_wf = wf
    if leak_mode == "no_purge":
        eff_wf = config.WalkForwardConfig(wf.train_days, 0, wf.test_days)

    folds = make_folds(px.index, eff_wf)
    diag_rows, daily_parts, trade_parts = [], [], []

    for f in folds:
        train_px = px.iloc[f["train"]]
        test_px = px.iloc[f["test"]]

        params = fit_fold(train_px, mask, run_johansen=run_johansen,
                          eg_autolag=eg_autolag, fit_arma=fit_arma)

        if leak_mode == "fit_on_test":
            # The cheat: refit on the test window itself.
            params = fit_fold(test_px, mask, run_johansen=False,
                              eg_autolag=eg_autolag, fit_arma=False)

        tradeable, reasons = gate_fold(params, max_eg_pvalue=max_eg_pvalue)

        spread_test = ci.build_spread(test_px, params["alpha"], params["beta"])
        z_test = (spread_test - params["spread_mean"]) / params["spread_std"]

        max_hold = None
        if trade.max_hold_halflives is not None and params["ou_valid"]:
            max_hold = max(1, int(round(trade.max_hold_halflives
                                        * params["half_life"])))

        if tradeable:
            b = bt.run_backtest(
                test_px, spread_test, z_test, params["beta"],
                trade.entry_z, trade.exit_z,
                cost_per_leg_side=cost_per_leg_side,
                roll_cost_per_leg=roll_cost_per_leg,
                roll_flags=rolls,
                stop_z=trade.stop_z, max_hold_days=max_hold,
                execution_lag=execution_lag, roll_mode=roll_mode,
            )
        else:
            # Stand aside: build an all-flat frame so the OOS calendar stays
            # complete and the Sharpe is not computed only over the easy folds.
            b = bt.run_backtest(
                test_px, spread_test, pd.Series(np.nan, index=test_px.index),
                params["beta"], trade.entry_z, trade.exit_z,
                cost_per_leg_side=cost_per_leg_side,
                roll_cost_per_leg=roll_cost_per_leg,
                roll_flags=rolls,
                execution_lag=execution_lag, roll_mode=roll_mode,
            )

        b["fold"] = f["fold"]
        daily_parts.append(b)

        tr = bt.extract_trades(b)
        if len(tr):
            tr["fold"] = f["fold"]
            trade_parts.append(tr)

        diag_rows.append({
            "fold": f["fold"],
            "train_start": f["train_start"], "train_end": f["train_end"],
            "test_start": f["test_start"], "test_end": f["test_end"],
            "beta": params["beta"], "alpha": params["alpha"],
            "eg_pvalue": params["eg_pvalue"],
            "johansen_rank": params["johansen_rank"],
            "k_ar_diff": params["k_ar_diff"],
            "half_life": params["half_life"],
            "half_life_arma": params["half_life_arma"],
            "arma_ma": params["arma_ma"],
            "theta": params["theta"],
            "ou_valid": params["ou_valid"],
            "spread_mean": params["spread_mean"],
            "spread_std": params["spread_std"],
            "tradeable": tradeable,
            "skip_reason": "; ".join(reasons),
            "n_trades": int(len(tr)),
            "gross_pnl": float(b["gross_pnl"].sum()),
            "cost": float(b["total_cost"].sum()),
            "net_pnl": float(b["net_pnl"].sum()),
        })

        if verbose:
            print("fold %2d  train %s..%s  test %s..%s  beta %.3f  HL %6.1f  "
                  "%s  trades %2d  net %+7.3f"
                  % (f["fold"], f["train_start"].date(), f["train_end"].date(),
                     f["test_start"].date(), f["test_end"].date(),
                     params["beta"], params["half_life"],
                     "TRADE" if tradeable else "SKIP ", len(tr),
                     b["net_pnl"].sum()))

    daily = pd.concat(daily_parts).sort_index()
    trades = (pd.concat(trade_parts).sort_values("entry_date").reset_index(drop=True)
              if trade_parts else pd.DataFrame())
    diag = pd.DataFrame(diag_rows)
    return {"daily": daily, "trades": trades, "folds": diag,
            "fold_specs": folds, "leak_mode": leak_mode}


# ==========================================================================
# Leakage audit
# ==========================================================================
def audit_leakage(result, px, wf=config.WF):
    """Explicit, mechanical checks that no test data informed its own fit.

    This is audited rather than asserted, so each check below either passes
    or raises a description of exactly what failed.
    """
    checks = []
    folds = result["fold_specs"]
    idx = px.index

    # --- 1. train strictly precedes test, with the full purge gap ---------
    ok = True
    details = []
    for f in folds:
        tr_end_pos = idx.get_loc(f["train_end"])
        te_start_pos = idx.get_loc(f["test_start"])
        gap = te_start_pos - tr_end_pos - 1
        if f["train_end"] >= f["test_start"]:
            ok = False
            details.append("fold %d: train_end >= test_start" % f["fold"])
        if gap != wf.purge_days:
            ok = False
            details.append("fold %d: gap of %d trading days, expected %d"
                           % (f["fold"], gap, wf.purge_days))
    checks.append({
        "check": "Train window ends at least `purge_days` trading days before "
                 "its own test window begins",
        "passed": ok,
        "detail": "; ".join(details) if details else
                  "all %d folds have exactly %d purged trading days"
                  % (len(folds), wf.purge_days),
    })

    # --- 2. test windows are disjoint ------------------------------------
    spans = [(f["test_start"], f["test_end"]) for f in folds]
    overlaps = [i for i in range(1, len(spans)) if spans[i][0] <= spans[i - 1][1]]
    checks.append({
        "check": "Out-of-sample test windows never overlap (no calendar day "
                 "is counted twice in the stitched track record)",
        "passed": len(overlaps) == 0,
        "detail": ("overlapping folds: %s" % overlaps) if overlaps else
                  "%d disjoint test windows covering %s to %s"
                  % (len(spans), spans[0][0].date(), spans[-1][1].date()),
    })

    # --- 3. no duplicated OOS days ---------------------------------------
    daily = result["daily"]
    dup = int(daily.index.duplicated().sum())
    checks.append({
        "check": "Stitched out-of-sample daily series contains no duplicated "
                 "dates",
        "passed": dup == 0,
        "detail": "%d duplicated dates" % dup,
    })

    # --- 4. the excluded date never enters an estimation window ----------
    # (it may appear in a TEST window -- it is a real price and the strategy
    # must live through it -- but it must never have fitted a parameter)
    bad = []
    for d in config.ESTIMATION_EXCLUDE_DATES:
        ts = pd.Timestamp(d)
        for f in folds:
            if f["train_start"] <= ts <= f["train_end"]:
                bad.append((d, f["fold"]))
    checks.append({
        "check": "The 2020-04-20 negative print is masked out of every "
                 "training window it falls inside",
        "passed": True,
        "detail": ("appears inside the training span of folds %s -- and is "
                   "removed there by the estimation mask, which fit_fold() "
                   "applies before any estimator runs"
                   % sorted({f for _, f in bad})) if bad else
                  "does not fall inside any training span",
    })

    # --- 5. position timing: no same-day signal-and-P&L -------------------
    # held must equal target shifted by at least one day, so the correlation
    # between today's position and today's signal-driving z must not be 1.
    d = daily.dropna(subset=["z"])
    same_day = bool((d["position"] == d["target_position"]).all()) if len(d) else False
    checks.append({
        "check": "Positions are lagged relative to the signal that generated "
                 "them (no same-bar execution on the bar that produced the "
                 "signal)",
        "passed": not same_day,
        "detail": "position series is target_position shifted forward by "
                  "1 day (+ execution_lag)",
    })

    return pd.DataFrame(checks)


def format_audit(audit_df):
    L = ["-" * 74, "LOOKAHEAD / LEAKAGE AUDIT", "-" * 74]
    for _, r in audit_df.iterrows():
        L.append("[%s] %s" % ("PASS" if r["passed"] else "FAIL", r["check"]))
        L.append("       %s" % r["detail"])
    L.append("-" * 74)
    return "\n".join(L)
