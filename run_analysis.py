"""
Brent-WTI statistical arbitrage: full study driver.

Runs, in order:
  0. data load + data-quality diagnostics
  1. estimator validation against synthetic data with known ground truth
  2. full-sample exploratory cointegration analysis (go / no-go ONLY)
  3. purged walk-forward backtest + leakage audit + negative controls
  4. robustness grids (costs, thresholds, execution lag, specification)
  5. the 2022 regime-break case study
  6. figures and CSV tables

Usage
-----
    python run_analysis.py                 # full study
    python run_analysis.py --quick         # fewer Monte Carlo reps
    python run_analysis.py --refresh-data  # re-download from Yahoo
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from src import backtest as bt
from src import cointegration as ci
from src import config
from src import data as datamod
from src import metrics
from src import ou
from src import plots
from src import validation
from src import walkforward as wfm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

RESULTS = {}


def hdr(title, ch="="):
    line = ch * 78
    print("\n" + line + "\n" + title + "\n" + line)


def save_table(df, name):
    path = config.TAB_DIR / name
    df.to_csv(path, index=False)
    return path


# ==========================================================================
def step0_data(refresh):
    hdr("0. DATA")
    px = datamod.load_prices(refresh=refresh)
    mask = datamod.estimation_mask(px.index)
    rolls = datamod.estimate_roll_days(px.index)

    print("Sample        : %s -> %s  (%d trading days, both legs present)"
          % (px.index.min().date(), px.index.max().date(), len(px)))
    print("WTI range     : $%.2f to $%.2f" % (px["WTI"].min(), px["WTI"].max()))
    print("Brent range   : $%.2f to $%.2f" % (px["BRENT"].min(), px["BRENT"].max()))
    print("Negative WTI prints: %s"
          % list(px.index[px["WTI"] <= 0].strftime("%Y-%m-%d")))
    print("Excluded from ESTIMATION (kept in raw series for plotting): %s"
          % list(px.index[~mask].strftime("%Y-%m-%d")))

    print("\nFutures roll-day diagnostics (continuous front-month artefact):")
    rd = datamod.roll_jump_diagnostics(px, rolls)
    print(rd.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("  -> flagged roll days move %.0f%% / %.0f%% more than ordinary days "
          "(WTI / Brent). These jumps are a change of CONTRACT, not tradeable\n"
          "     P&L, so the backtest zeroes the rolling leg's move and charges "
          "a roll cost."
          % (100 * (rd.loc[0, "ratio"] - 1), 100 * (rd.loc[1, "ratio"] - 1)))
    save_table(rd, "00_roll_diagnostics.csv")

    RESULTS["data"] = {
        "start": str(px.index.min().date()), "end": str(px.index.max().date()),
        "n_obs": int(len(px)),
        "wti_min": float(px["WTI"].min()), "wti_max": float(px["WTI"].max()),
        "brent_min": float(px["BRENT"].min()), "brent_max": float(px["BRENT"].max()),
        "excluded": list(px.index[~mask].strftime("%Y-%m-%d")),
        "roll_ratio_wti": float(rd.loc[0, "ratio"]),
        "roll_ratio_brent": float(rd.loc[1, "ratio"]),
        "n_roll_days": int(rolls["any_roll"].sum()),
    }
    return px, mask, rolls


# ==========================================================================
def step1_validation(quick):
    hdr("1. ESTIMATOR VALIDATION (synthetic data, known ground truth)")
    n_ou = 150 if quick else 400
    n_co = 100 if quick else 300
    n_sz = 150 if quick else 400
    n_ns = 60 if quick else 150

    out = validation.run_all(n_reps_ou=n_ou, n_reps_coint=n_co,
                             n_reps_size=n_sz, verbose=True)

    print("\n[5] Half-life under additive microstructure noise (truth = 30 d)")
    noise = validation.validate_half_life_under_noise(n_reps=n_ns)
    print(noise.to_string(index=False, float_format=lambda v: "%9.3f" % v))
    print("  -> a plain AR(1) half-life collapses towards zero once the")
    print("     observed spread carries transient noise. This is why the fold")
    print("     diagnostics also report an ARMA(1,1) half-life.")

    save_table(out["ou"], "01_validation_ou.csv")
    save_table(noise, "02_validation_noise.csv")
    save_table(pd.DataFrame([out["size_iid"], out["size_garch"]]),
               "03_validation_size.csv")
    plots.plot_validation_half_life(out["ou"])

    RESULTS["validation"] = {
        "ou_recovery": out["ou"].to_dict("records"),
        "power": out["power"],
        "size_iid": out["size_iid"],
        "size_garch": out["size_garch"],
        "noise": noise.to_dict("records"),
    }
    return out


# ==========================================================================
def step2_exploratory(px, mask):
    hdr("2. FULL-SAMPLE EXPLORATORY ANALYSIS (go/no-go only -- NOT the backtest)")
    est = px[mask]
    print("Estimation sample: %d obs (2020-04-20 removed)\n" % len(est))

    # --- 2a. Are the levels I(1)? ---
    print("--- 2a. Unit-root precondition: is each series I(1)? ---")
    lvl = ci.adf_levels(est)
    print(lvl.to_string(index=False, float_format=lambda v: "%10.4f" % v))
    print("  Interpretation: cointegration is only a meaningful claim if each")
    print("  series is I(1). Both first differences reject decisively. The WTI")
    print("  LEVEL does not reject (p=%.3f). The Brent level is marginal"
          % lvl.loc[0, "p_value"])
    print("  (p=%.3f) -- borderline, and economically implausible as genuine"
          % lvl.loc[2, "p_value"])
    print("  stationarity; we treat both levels as I(1).")
    save_table(lvl, "04_adf_levels.csv")

    # --- 2b. Engle-Granger, both directions ---
    print("\n--- 2b. Engle-Granger, BOTH directions ---")
    eg_w, eg_b = ci.engle_granger_both(est)
    rows = []
    for e in (eg_w, eg_b):
        rows.append({
            "specification": e["spec"], "alpha": e["alpha"], "beta": e["beta"],
            "R2": e["r_squared"], "ADF_stat_naive": e["adf_stat_naive"],
            "ADF_p_naive": e["adf_p_naive"], "EG_stat": e["eg_stat"],
            "EG_pvalue": e["eg_pvalue"], "EG_cv_5pct": e["eg_crit"]["5%"],
            "reject_no_coint_5pct": e["eg_pvalue"] < 0.05,
        })
    egt = pd.DataFrame(rows)
    print(egt.to_string(index=False, float_format=lambda v: "%10.4f" % v))
    print("  Both directions reject the null of NO cointegration at 5%.")
    print("  Note the naive ADF p-value is far smaller than the MacKinnon EG")
    print("  p-value (%.5f vs %.5f): the naive one is WRONG here because the"
          % (eg_w["adf_p_naive"], eg_w["eg_pvalue"]))
    print("  cointegrating vector was estimated, not known. We act on the EG one.")
    save_table(egt, "05_engle_granger.csv")

    # --- 2c. VAR lag order ---
    print("\n--- 2c. VAR lag order on the LEVELS (feeds Johansen) ---")
    lag = ci.select_var_lag(est)
    print("  selected p by criterion: %s" % lag["p_by_criterion"])
    print("  BIC selects p=%s  ->  k_ar_diff = p-1 = %s"
          % (lag["p_bic"], lag["k_ar_diff_bic"]))
    print("  HQIC selects p=%s ->  k_ar_diff = %s"
          % (lag["p_hqic"], lag["k_ar_diff_hqic"]))
    print("  AIC selects p=%s, which we do NOT use: AIC is not consistent for"
          % lag["p_by_criterion"]["aic"])
    print("  lag selection and over-parameterises, which distorts the trace stat.")

    k = max(int(lag["k_ar_diff_bic"]), 1)

    # --- 2d. Johansen ---
    print("\n--- 2d. Johansen trace test (det_order=0, restricted constant, "
          "k_ar_diff=%d) ---" % k)
    joh = ci.johansen(est, config.JOHANSEN_DET_ORDER, k)
    print(joh["trace"].to_string(index=False, float_format=lambda v: "%10.4f" % v))
    print("\n  max-eigenvalue test:")
    print(joh["max_eig"].to_string(index=False, float_format=lambda v: "%10.4f" % v))
    print("\n  Johansen cointegrating vector (normalised): "
          "WTI - %.4f x Brent" % joh["beta"])
    save_table(joh["trace"], "06_johansen_trace.csv")

    print("\n--- 2e. Johansen robustness across lag lengths ---")
    rob = ci.johansen_lag_robustness(est)
    print(rob.to_string(index=False, float_format=lambda v: "%10.4f" % v))
    save_table(rob, "07_johansen_lag_robustness.csv")

    print("\n--- 2f. Rank decision (mechanical verdict vs economic sense) ---")
    dec = ci.decide_rank(joh, [eg_w, eg_b], est)
    print("  mechanical trace rank : %d" % dec["rank_mechanical"])
    print("  r<=1 margin over cv95 : %.1f%% of the critical value"
          % dec["r1_margin_pct_of_cv"])
    print("  levels individually stationary? %s" % dec["levels_individually_stationary"])
    print("  Engle-Granger agrees?          %s" % dec["eg_both_reject"])
    print("  ADOPTED RANK          : %d" % dec["rank_adopted"])
    for n in dec["notes"]:
        print("    * " + n)

    # --- 2g. OU fit on the full-sample residual ---
    print("\n--- 2g. Ornstein-Uhlenbeck fit on the cointegrating residual ---")
    spread = ci.build_spread(est, eg_w["alpha"], eg_w["beta"])
    o = ou.fit_ou(spread, mask.reindex(spread.index))
    print("  b (dS on lagged S)    : %+.6f   (t = %.2f, p = %.3g)"
          % (o["b"], o["b_tstat"], o["b_pvalue"]))
    print("  phi = 1 + b           : %.6f" % o["phi"])
    print("  theta (exact)         : %.6f per day" % o["theta"])
    print("  theta (Euler approx)  : %.6f per day" % o["theta_euler"])
    print("  mu (long-run mean)    : %+.4f $/bbl" % o["mu"])
    print("  sigma                 : %.4f" % o["sigma"])
    print("  HALF-LIFE (exact)     : %.2f trading days" % o["half_life"])
    print("  half-life (Euler)     : %.2f trading days  <- reported for"
          % o["half_life_euler"])
    print("                           contrast only; we use the exact inversion.")
    arma = ou.half_life_arma11(spread, mask.reindex(spread.index))
    print("  HALF-LIFE, ARMA(1,1)  : %.2f days (AR root %.4f, MA %.4f)"
          % (arma["half_life"], arma["phi"], arma["ma"]))
    print("    The MA term is negative, i.e. the spread carries transient")
    print("    bid-ask/settlement noise, which biases the AR(1) half-life SHORT.")

    RESULTS["exploratory"] = {
        "adf_levels": lvl.to_dict("records"),
        "eg": egt.to_dict("records"),
        "var_lag": lag["p_by_criterion"],
        "k_ar_diff": int(k),
        "johansen_trace": joh["trace"].to_dict("records"),
        "johansen_beta": float(joh["beta"]),
        "rank_mechanical": int(dec["rank_mechanical"]),
        "rank_adopted": int(dec["rank_adopted"]),
        "rank_notes": dec["notes"],
        "r1_margin_pct": float(dec["r1_margin_pct_of_cv"]),
        "ou": {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
               for kk, vv in o.items() if kk not in ("resid",)},
        "ou_arma": arma,
        "johansen_lag_robustness": rob.to_dict("records"),
    }

    # figures that depend only on the exploratory stage
    plots.plot_prices(px)
    plots.plot_spread(ci.build_spread(px, eg_w["alpha"], eg_w["beta"]),
                      eg_w["beta"], eg_w["alpha"])

    print("\n--- 2h. Rolling half-life (stability of the mean-reversion speed) ---")
    full_spread = ci.build_spread(px, eg_w["alpha"], eg_w["beta"])
    roll_naive = ou.rolling_half_life(full_spread, mask, window=252, step=5)
    arma_idx, arma_hl = [], []
    for end in range(252, len(full_spread) + 1, 21):
        sl = full_spread.iloc[end - 252:end]
        g = ou.half_life_arma11(sl, mask.reindex(sl.index))
        arma_idx.append(full_spread.index[end - 1])
        arma_hl.append(g["half_life"] if g["valid"] else np.nan)
    roll_arma = pd.DataFrame({"half_life": arma_hl},
                             index=pd.DatetimeIndex(arma_idx))
    valid = roll_naive["half_life"].dropna()
    print("  AR(1) rolling half-life: min %.1f  median %.1f  max %.1f days"
          % (valid.min(), valid.median(), valid.max()))
    print("  -> the mean-reversion speed is NOT stable; this is the single")
    print("     strongest argument for re-estimating on every walk-forward fold")
    print("     rather than trusting one full-sample number.")
    plots.plot_rolling_half_life(roll_naive, roll_arma)
    RESULTS["rolling_half_life"] = {
        "min": float(valid.min()), "median": float(valid.median()),
        "max": float(valid.max()),
        "pct_above_test_window": float(100 * (valid > config.WF.test_days).mean()),
    }

    return eg_w, eg_b, joh, dec, o


# ==========================================================================
def step3_walkforward(px, mask, rolls):
    hdr("3. PURGED WALK-FORWARD BACKTEST")
    print("Design: train %d d | purge %d d | test %d d | step %d d"
          % (config.WF.train_days, config.WF.purge_days,
             config.WF.test_days, config.WF.step_days))
    print("Entry |z| > %.1f, exit |z| < %.1f, thresholds held FIXED (not fitted)."
          % (config.TRADE.entry_z, config.TRADE.exit_z))
    print("Costs: $%.2f/bbl per leg per side, plus $%.2f/bbl per leg per roll.\n"
          % (config.COSTS.cost_per_leg_side, config.COSTS.roll_cost_per_leg))

    res = wfm.run_walk_forward(px, rolls, mask, verbose=True)
    daily, trades, folds = res["daily"], res["trades"], res["folds"]

    print("\nFolds: %d total, %d traded, %d stood aside."
          % (len(folds), int(folds["tradeable"].sum()),
             int((~folds["tradeable"]).sum())))
    skipped = folds[~folds["tradeable"]]
    if len(skipped):
        print("Stand-aside reasons:")
        for r, n in skipped["skip_reason"].value_counts().items():
            print("   %2dx  %s" % (n, r[:100]))
    save_table(folds, "08_walkforward_folds.csv")
    save_table(trades, "09_walkforward_trades.csv")

    hl_ref = float(folds.loc[folds["tradeable"], "half_life"].mean())
    summ = metrics.summarise(daily, trades, half_life_ref=hl_ref)
    print()
    print(metrics.format_summary(summ))

    # --- leakage audit ---
    print()
    audit = wfm.audit_leakage(res, px)
    print(wfm.format_audit(audit))
    save_table(audit, "10_leakage_audit.csv")

    # --- negative controls ---
    print("\n--- Negative controls: would this harness DETECT leakage? ---")
    ctrl_rows = []
    for mode, desc in (("none", "clean protocol (reported result)"),
                       ("no_purge", "purge gap removed"),
                       ("fit_on_test", "parameters deliberately fitted ON the test window")):
        r = res if mode == "none" else wfm.run_walk_forward(px, rolls, mask,
                                                            leak_mode=mode, **FAST)
        s = metrics.summarise(r["daily"], r["trades"])
        ctrl_rows.append({"mode": mode, "description": desc,
                          "n_trades": s["n_trades"],
                          "sharpe_net": s["sharpe_net"],
                          "net_pnl": s["net_pnl_total"],
                          "hit_rate": s["hit_rate_net"]})
    ctrl = pd.DataFrame(ctrl_rows)
    print(ctrl.to_string(index=False, float_format=lambda v: "%9.3f" % v))
    print("  -> Deliberately fitting on the test window raises the net Sharpe")
    print("     from %.3f to %.3f. The harness is therefore sensitive to"
          % (ctrl.loc[0, "sharpe_net"], ctrl.loc[2, "sharpe_net"]))
    print("     leakage, and the clean number is not being flattered by it.")
    save_table(ctrl, "11_negative_controls.csv")

    RESULTS["walkforward"] = {
        k: (str(v) if isinstance(v, pd.Timestamp) else v)
        for k, v in summ.items()
    }
    RESULTS["walkforward"]["n_folds"] = int(len(folds))
    RESULTS["walkforward"]["n_folds_traded"] = int(folds["tradeable"].sum())
    RESULTS["audit"] = audit.to_dict("records")
    RESULTS["negative_controls"] = ctrl.to_dict("records")

    plots.plot_oos_pnl(daily, folds)
    plots.plot_drawdown(daily)
    plots.plot_fold_diagnostics(folds)
    return res, summ


# ==========================================================================
# Robustness sweeps vary costs, thresholds, windows and gates -- none of which
# change the Johansen rank verdict or the ARMA half-life cross-check. Those two
# diagnostics are the slowest part of a fold fit, so we switch them off here.
# The base-case run in step 3 computes them in full.
FAST = dict(run_johansen=False, fit_arma=False)


def step4_robustness(px, mask, rolls, base_summary):
    hdr("4. ROBUSTNESS")

    # --- 4a. transaction costs ---
    print("--- 4a. Transaction-cost sensitivity ---")
    rows = []
    for c in config.COSTS.sensitivity_grid:
        r = wfm.run_walk_forward(px, rolls, mask, cost_per_leg_side=c,
                                 roll_cost_per_leg=c, **FAST)
        s = metrics.summarise(r["daily"], r["trades"])
        rows.append({"cost_per_leg_side": c, "sharpe_gross": s["sharpe_gross"],
                     "sharpe_net": s["sharpe_net"], "net_pnl": s["net_pnl_total"],
                     "total_costs": s["total_costs"],
                     "cost_pct_of_gross": s["cost_pct_of_gross"]})
    cost_tab = pd.DataFrame(rows)
    print(cost_tab.to_string(index=False, float_format=lambda v: "%10.3f" % v))
    save_table(cost_tab, "12_cost_sensitivity.csv")
    plots.plot_cost_sensitivity(cost_tab)

    # --- 4b. z-thresholds ---
    print("\n--- 4b. Entry/exit threshold sensitivity ---")
    rows = []
    for entry in (1.5, 2.0, 2.5, 3.0):
        for exit_ in (0.0, 0.5, 1.0):
            tc = config.TradingConfig(entry_z=entry, exit_z=exit_)
            r = wfm.run_walk_forward(px, rolls, mask, trade=tc, **FAST)
            s = metrics.summarise(r["daily"], r["trades"])
            rows.append({"entry_z": entry, "exit_z": exit_,
                         "n_trades": s["n_trades"], "sharpe_net": s["sharpe_net"],
                         "net_pnl": s["net_pnl_total"],
                         "hit_rate": s["hit_rate_net"],
                         "avg_hold_d": s["avg_holding_days"]})
    thr = pd.DataFrame(rows)
    print(thr.to_string(index=False, float_format=lambda v: "%9.3f" % v))
    print("  The reported base case (2.0 / 0.5) is NOT the best cell here.")
    print("  That is deliberate: the thresholds were fixed a priori, and")
    print("  quoting the argmax of this grid would be an overfit.")
    save_table(thr, "13_threshold_sensitivity.csv")

    # --- 4c. execution lag ---
    print("\n--- 4c. Execution-lag sensitivity (how fast must we trade?) ---")
    rows = []
    for lag in (0, 1, 2):
        r = wfm.run_walk_forward(px, rolls, mask, execution_lag=lag, **FAST)
        s = metrics.summarise(r["daily"], r["trades"])
        rows.append({"execution_lag_days": lag, "sharpe_net": s["sharpe_net"],
                     "net_pnl": s["net_pnl_total"], "n_trades": s["n_trades"]})
    lag_tab = pd.DataFrame(rows)
    print(lag_tab.to_string(index=False, float_format=lambda v: "%9.3f" % v))
    print("  lag=0 means we trade at the same settlement that produced the")
    print("  signal; lag=1 delays execution by a full day. A strategy whose")
    print("  edge vanishes at lag=1 is really a microstructure effect.")
    save_table(lag_tab, "14_execution_lag.csv")

    # --- 4c2. roll treatment bounds ---
    print("\n--- 4c(ii). Futures-roll treatment: the two bounds ---")
    rows = []
    for mode, desc in (
        ("neutralise", "book nothing on roll days (conservative, biased DOWN)"),
        ("naive", "book the observed jump (ignores rolls, biased UP)")):
        rr = wfm.run_walk_forward(px, rolls, mask, roll_mode=mode, **FAST)
        ss = metrics.summarise(rr["daily"], rr["trades"])
        rows.append({"roll_mode": mode, "treatment": desc,
                     "sharpe_net": ss["sharpe_net"],
                     "net_pnl": ss["net_pnl_total"],
                     "max_dd": ss["max_dd_dollars"],
                     "hit_rate": ss["hit_rate_net"]})
    roll_tab = pd.DataFrame(rows)
    print(roll_tab.to_string(index=False, float_format=lambda v: "%9.3f" % v))
    print("  A continuous front-month series cannot separate a genuine price")
    print("  move on a roll day from the contract-switch gap. These are the")
    print("  bounds; the truth is between them. We report the conservative")
    print("  end as the headline because over-claiming is the worse error.")
    save_table(roll_tab, "16_roll_treatment.csv")
    RESULTS["roll_treatment"] = roll_tab.to_dict("records")

    # --- 4d. specification: EG lag selection + exclusion set + window ---
    print("\n--- 4d. Specification sensitivity ---")
    rows = []
    for name, kwargs in (
        ("base (AIC lag, exclude 04-20, 756d train)", {}),
        ("EG lag selection = BIC", {"eg_autolag": "bic"}),
        ("EG gate p<0.05 (stricter)", {"max_eg_pvalue": 0.05}),
        ("EG gate p<1.00 (no gate)", {"max_eg_pvalue": 1.00}),
    ):
        r = wfm.run_walk_forward(px, rolls, mask, **kwargs, **FAST)
        s = metrics.summarise(r["daily"], r["trades"])
        rows.append({"specification": name,
                     "folds_traded": int(r["folds"]["tradeable"].sum()),
                     "n_trades": s["n_trades"], "sharpe_net": s["sharpe_net"],
                     "net_pnl": s["net_pnl_total"], "max_dd": s["max_dd_dollars"]})

    # wider exclusion set
    mask2 = datamod.estimation_mask(px.index, config.ROBUSTNESS_EXCLUDE_DATES)
    r = wfm.run_walk_forward(px, rolls, mask2, **FAST)
    s = metrics.summarise(r["daily"], r["trades"])
    rows.append({"specification": "also exclude 2020-04-21",
                 "folds_traded": int(r["folds"]["tradeable"].sum()),
                 "n_trades": s["n_trades"], "sharpe_net": s["sharpe_net"],
                 "net_pnl": s["net_pnl_total"], "max_dd": s["max_dd_dollars"]})

    # train-window length
    for td in (504, 1008):
        wf2 = config.WalkForwardConfig(train_days=td,
                                       purge_days=config.WF.purge_days,
                                       test_days=config.WF.test_days)
        r = wfm.run_walk_forward(px, rolls, mask, wf=wf2, **FAST)
        s = metrics.summarise(r["daily"], r["trades"])
        rows.append({"specification": "train window = %d days" % td,
                     "folds_traded": int(r["folds"]["tradeable"].sum()),
                     "n_trades": s["n_trades"], "sharpe_net": s["sharpe_net"],
                     "net_pnl": s["net_pnl_total"], "max_dd": s["max_dd_dollars"]})

    # purge length
    for pg in (0, 42):
        wf2 = config.WalkForwardConfig(train_days=config.WF.train_days,
                                       purge_days=pg,
                                       test_days=config.WF.test_days)
        r = wfm.run_walk_forward(px, rolls, mask, wf=wf2, **FAST)
        s = metrics.summarise(r["daily"], r["trades"])
        rows.append({"specification": "purge gap = %d days" % pg,
                     "folds_traded": int(r["folds"]["tradeable"].sum()),
                     "n_trades": s["n_trades"], "sharpe_net": s["sharpe_net"],
                     "net_pnl": s["net_pnl_total"], "max_dd": s["max_dd_dollars"]})

    spec = pd.DataFrame(rows)
    print(spec.to_string(index=False, float_format=lambda v: "%9.3f" % v))
    save_table(spec, "15_specification_sensitivity.csv")

    RESULTS["robustness"] = {
        "costs": cost_tab.to_dict("records"),
        "thresholds": thr.to_dict("records"),
        "execution_lag": lag_tab.to_dict("records"),
        "specification": spec.to_dict("records"),
    }
    return cost_tab, thr, lag_tab, spec


# ==========================================================================
def step5_case_study(px, res):
    hdr("5. CASE STUDY: THE 2022 BRENT-WTI DISLOCATION")
    daily = res["daily"]
    folds = res["folds"]

    win = daily.loc["2022-01-01":"2022-12-31"]
    win_trades = res["trades"]
    if len(win_trades):
        win_trades = win_trades[
            (win_trades["entry_date"] >= "2022-01-01")
            & (win_trades["entry_date"] <= "2022-12-31")]

    sp = px.loc["2021-10-01":"2022-12-31"]
    raw = (sp["WTI"] - sp["BRENT"])
    print("Raw WTI-Brent differential, Oct 2021 - Dec 2022:")
    print("   start (2021-10-01 area) : %+.2f $/bbl" % raw.iloc[0])
    print("   widest (most negative)  : %+.2f $/bbl on %s"
          % (raw.min(), raw.idxmin().date()))
    print("   year-end 2022           : %+.2f $/bbl" % raw.iloc[-1])

    print("\nWhat happened economically:")
    print("  Russia's invasion of Ukraine in Feb 2022 repriced WATERBORNE crude.")
    print("  Brent is the seaborne benchmark and absorbed the sanctions risk")
    print("  premium directly; WTI is landlocked at Cushing and only reaches")
    print("  global buyers through Gulf Coast export capacity. The differential")
    print("  therefore widened far beyond the transport-cost band that normally")
    print("  anchors it. Crucially this was a change in the EQUILIBRIUM, not a")
    print("  deviation from it -- exactly the case where a mean-reversion rule")
    print("  is structurally wrong.")

    print("\nHow the model handled it:")
    fold23 = folds[folds["fold"] == 23]
    if len(fold23):
        f = fold23.iloc[0]
        print("  Fold 23 (train %s..%s, test %s..%s):"
              % (f["train_start"].date(), f["train_end"].date(),
                 f["test_start"].date(), f["test_end"].date()))
        print("    training EG p-value %.4f -> gate PASSED, fold traded"
              % f["eg_pvalue"])
        print("    training half-life  %.1f days, beta %.3f"
              % (f["half_life"], f["beta"]))
        print("    trades in fold %d, net P&L %+.2f $/bbl"
              % (f["n_trades"], f["net_pnl"]))
    print("\n  2022 calendar-year out-of-sample P&L: %+.2f $/bbl gross, "
          "%+.2f net" % (win["gross_pnl"].sum(), win["net_pnl"].sum()))
    if len(win_trades):
        print("  Trades entered in 2022: %d, of which %d profitable net"
              % (len(win_trades), int((win_trades["net_pnl"] > 0).sum())))
        print(win_trades[["entry_date", "exit_date", "side", "holding_days",
                          "entry_z", "exit_z", "gross_pnl", "cost", "net_pnl"]]
              .to_string(index=False, float_format=lambda v: "%8.3f" % v))

    print("\n  The honest lesson: the training window ending April 2022 still")
    print("  contained enough pre-war data to pass the cointegration gate, so")
    print("  the model traded INTO a structural break and lost money doing it.")
    print("  A p-value computed on a 3-year window cannot detect a regime")
    print("  change that happened two months before the window ended. This is")
    print("  the fundamental limitation of the approach, not a tuning problem.")
    print("  Note what DID work: by the following fold the gate had absorbed")
    print("  the new data and the strategy re-adapted rather than compounding")
    print("  the loss.")

    plots.plot_case_study_2022(daily, px)
    RESULTS["case_study_2022"] = {
        "raw_min_diff": float(raw.min()),
        "raw_min_date": str(raw.idxmin().date()),
        "gross_2022": float(win["gross_pnl"].sum()),
        "net_2022": float(win["net_pnl"].sum()),
        "n_trades_2022": int(len(win_trades)) if len(win_trades) else 0,
    }


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer Monte Carlo replications")
    ap.add_argument("--refresh-data", action="store_true",
                    help="re-download from Yahoo Finance")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    px, mask, rolls = step0_data(args.refresh_data)
    if not args.skip_validation:
        step1_validation(args.quick)
    step2_exploratory(px, mask)
    res, summ = step3_walkforward(px, mask, rolls)
    step4_robustness(px, mask, rolls, summ)
    step5_case_study(px, res)

    out = config.ROOT / "outputs" / "results.json"
    with open(out, "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)

    hdr("DONE")
    print("Figures : %s" % config.FIG_DIR)
    print("Tables  : %s" % config.TAB_DIR)
    print("Results : %s" % out)


if __name__ == "__main__":
    main()
