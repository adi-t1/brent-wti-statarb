"""
Central configuration for the Brent-WTI cointegration stat-arb study.

Every parameter that a reviewer might question lives here, with the reasoning
attached, so that no "magic number" is buried inside the analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
for _d in (CACHE_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
WTI_TICKER = "CL=F"      # NYMEX Light Sweet Crude, continuous front month
BRENT_TICKER = "BZ=F"    # ICE Brent, continuous front month
START_DATE = "2007-01-01"

TRADING_DAYS_PER_YEAR = 252

# The famous negative settle. WTI printed -$37.63 on 2020-04-20 because the
# May-2020 contract expired into a physical delivery point (Cushing, OK) that
# had effectively run out of storage. It is a REAL print, not a data error, but
# it is an artifact of holding a naive front-month series into expiry: a real
# desk would have rolled days earlier and never seen it.
#
# We keep it in the plotted/raw series (it is economically meaningful context)
# but exclude it from every PARAMETER ESTIMATION step. Rationale: OLS, ADF,
# Johansen and the OU fit are all least-squares-type estimators whose influence
# function is unbounded, so a single -$37 observation in a series that otherwise
# lives in [20, 140] would dominate the hedge ratio and the mean-reversion speed
# in any window containing it.
ESTIMATION_EXCLUDE_DATES = ("2020-04-20",)

# 2020-04-21 (WTI $10.01) is also a distorted post-event print. The base case
# keeps the exclusion to the single date; we run the wider
# exclusion as a robustness check and report the difference rather than quietly
# choosing the more flattering one.
ROBUSTNESS_EXCLUDE_DATES = ("2020-04-20", "2020-04-21")

# --------------------------------------------------------------------------
# Cointegration testing
# --------------------------------------------------------------------------
# Johansen deterministic specification. det_order=0 is the "restricted constant"
# case: the cointegrating relation may have a non-zero mean (Brent trades at a
# persistent premium/discount to WTI, so the spread's mean is not zero) but the
# levels have no deterministic linear time trend (a $/bbl price does not drift
# linearly forever). det_order=-1 (no constant) would wrongly force the spread
# through zero; det_order=1 (linear trend) would fit a deterministic trend that
# has no economic justification and costs test power.
JOHANSEN_DET_ORDER = 0

# Candidate VAR lag orders searched by BIC/HQIC on the LEVELS VAR.
MAX_VAR_LAGS = 12

# Lag augmentation used in the Engle-Granger second-stage unit-root test.
# NOTE the deliberate asymmetry with the VAR lag selection above, which uses
# BIC. The two serve different purposes:
#   - The VAR feeding Johansen wants PARSIMONY: extra lags eat degrees of
#     freedom and distort the trace statistic, and BIC is consistent.
#   - The ADF regression inside Engle-Granger wants WHITE residuals: the
#     Dickey-Fuller distribution is only valid once serial correlation has
#     been soaked up, and the spread carries a negative MA component
#     (bid-ask bounce across two exchanges) that needs lags to absorb.
# This choice is materially load-bearing on this dataset -- see the README
# section on specification sensitivity -- so it is exposed here rather than
# buried, and the alternative is reported as a robustness case.
EG_AUTOLAG = "aic"

# Significance level used when reading Johansen critical values (0=90%, 1=95%, 2=99%)
JOHANSEN_CRIT_IDX = 1  # 95%

# --------------------------------------------------------------------------
# Walk-forward design
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardConfig:
    """Purged rolling walk-forward.

    train_days: 3 years. Long enough for a stable cointegrating vector and an
        ADF test with reasonable power (~750 obs), short enough that the
        estimate tracks structural change. The Brent-WTI relationship is NOT
        stable across the sample (pre-shale parity, the 2011-2014 shale
        dislocation, the 2015 US export-ban repeal, 2022 Russia sanctions), so
        an expanding window would anchor the hedge ratio on a regime that no
        longer exists.
    purge_days: 21 trading days (~1 month) left completely unused between the
        end of train and the start of test. This is the leakage firewall --
        see walkforward.py for the full argument.
    test_days: 126 trading days (~6 months) of out-of-sample evaluation, then
        roll forward by exactly that amount so test windows tile the sample
        without overlapping. Non-overlapping test windows matter: overlapping
        ones would reuse the same days in multiple folds and make the stitched
        OOS track record double-count.
    """
    train_days: int = 756
    purge_days: int = 21
    test_days: int = 126

    @property
    def step_days(self) -> int:
        # Step == test_days so that stitched OOS periods are disjoint.
        return self.test_days


WF = WalkForwardConfig()

# --------------------------------------------------------------------------
# Trading rule
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TradingConfig:
    """z-score band rule. Thresholds are held FIXED rather than optimised.

    Why fixed and not tuned per window? Tuning entry/exit on the training
    window is legitimate (it uses no test data), but with ~30 folds it invites
    overfitting of the *rule* to in-sample noise, and the resulting OOS Sharpe
    becomes hard to interpret. Holding the classic (2.0, 0.5) bands fixed means
    the ONLY things re-estimated each fold are the hedge ratio and the
    spread's mean/std -- quantities with a clear statistical meaning. We report
    a threshold-sensitivity grid instead of picking a winner.
    """
    entry_z: float = 2.0
    exit_z: float = 0.5
    # Optional risk controls (base case leaves both off so the strategy's raw
    # behaviour through the 2022 regime break is visible)
    stop_z: float | None = None          # hard stop if |z| widens beyond this
    max_hold_halflives: float | None = None  # time stop as a multiple of the OU half-life


TRADE = TradingConfig()

# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostConfig:
    """Transaction costs in $/bbl, charged per leg per side.

    CL and BZ both quote in $0.01/bbl ticks and are among the most liquid
    futures in the world; a front-month bid-ask is typically one tick. Crossing
    the spread costs half a tick ($0.005) in theory, but a backtest that
    assumes it always trades at mid+half-tick is fantasy: signals fire on
    closes, fills happen with delay, and size moves the book. We therefore use
    $0.05/bbl per leg per side as the BASE case -- 10x the theoretical
    half-spread -- and report a sensitivity grid from 1c to 10c.

    A round trip touches 2 legs x 2 sides, and the Brent leg is sized |beta|,
    so round-trip cost = 2 * cost_per_leg_side * (1 + |beta|) $/bbl of WTI.
    """
    cost_per_leg_side: float = 0.05
    sensitivity_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    # Cost of rolling an open position from the expiring to the next contract.
    # Charged per leg per roll event while a position is open.
    roll_cost_per_leg: float = 0.05


COSTS = CostConfig()

# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
SUSPICIOUS_SHARPE = 2.5   # above this we investigate for leakage rather than celebrate
RANDOM_SEED = 20260914
