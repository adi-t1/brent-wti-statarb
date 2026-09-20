"""
Backtest engine.

This module is deliberately "dumb": it takes a spread, a z-score series and a
set of ALREADY-FROZEN parameters, and turns them into positions, P&L and
trades. It does no estimation whatsoever. That separation is the structural
defence against lookahead bias -- if no fitting can happen in here, then
nothing in here can accidentally peek at the test period.

Accounting conventions (all of them chosen to be conservative)
--------------------------------------------------------------
* TIMING. The z-score on day t is computed from day t's settlement prices. The
  position implied by it is established at that same settlement, and it earns
  P&L from t to t+1. So the position series that multiplies day t's return is
  the signal from day t-1: `positions.shift(1)`. Getting this wrong by one day
  is the single most common source of a fake Sharpe ratio in pairs-trading
  backtests, so it is done once, here, and nowhere else. `execution_lag` adds
  further delay for robustness testing.

* P&L. Holding +1 unit of the spread means long 1 WTI and short beta Brent.
  Its daily P&L in $/bbl is therefore exactly the CHANGE in the spread,
  d(WTI) - beta*d(BRENT). The intercept alpha is a constant and cancels in the
  difference, so it never contributes to P&L.

* ROLL NEUTRALISATION. A continuous front-month series jumps when the quoted
  contract changes. That jump is a change of instrument, not a profit: a real
  position must be rolled and realises the calendar spread instead. Booking it
  as P&L is free money that does not exist. We therefore zero the P&L on
  flagged roll days and charge an explicit roll cost on any open position.

* RETURNS. Daily return = daily $ P&L divided by the gross dollar notional of
  the two legs at the previous close, (WTI + |beta|*BRENT). This is an
  UNLEVERED return on gross exposure. Flat days are included as zero returns
  rather than dropped -- a strategy that is only in the market 30% of the time
  should not be allowed to quote the Sharpe of the 30%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ==========================================================================
# Signal generation
# ==========================================================================
def generate_positions(z, entry_z, exit_z, stop_z=None,
                       max_hold_days=None, freeze_days=None):
    """z-score band state machine. Returns the TARGET position at each close.

    Convention: position +1 = long the spread (buy WTI, sell beta Brent),
    which is what we want when the spread is unusually LOW (z very negative)
    and we expect it to rise back to its mean.

    Exit on |z| < exit_z rather than on a strict mean crossing: waiting for an
    exact crossing of zero leaves the position on for longer with no
    additional expected edge, and in a regime break it may never cross at all.

    `freeze_days` (the roll-day flags) holds the position STATE unchanged on
    days when the quoted contract changes. This is not a cosmetic guard. The
    spread's level jumps across a roll by the calendar spread, which can move
    the z-score by more than a standard deviation in one session purely as an
    artefact of switching instrument. Left unhandled, that jump fires spurious
    entries and -- more damagingly -- spurious EXITS. In 2022 three positions
    were opened on a genuine signal and closed the very next session by a roll
    jump, booking no P&L (correctly neutralised) while paying a full round
    trip in costs. A real position would simply have been rolled and would
    never have seen that jump. So we neither open nor close on a roll day and
    let the next session -- where both prices refer to the same contract --
    make the decision.
    """
    z = pd.Series(z).astype(float)
    if freeze_days is None:
        frozen = np.zeros(len(z), dtype=bool)
    else:
        frozen = (pd.Series(freeze_days).reindex(z.index)
                  .fillna(False).astype(bool).values)
    pos = np.zeros(len(z))
    current = 0
    hold = 0
    for i, zt in enumerate(z.values):
        if np.isnan(zt):
            current, hold = 0, 0
            pos[i] = 0
            continue

        if frozen[i]:
            # The quoted contract changed today, so today's z is not
            # comparable with yesterday's. Hold, and decide tomorrow.
            if current != 0:
                hold += 1
            pos[i] = current
            continue

        if current == 0:
            if zt > entry_z:
                current, hold = -1, 1     # spread too high -> short it
            elif zt < -entry_z:
                current, hold = 1, 1      # spread too low  -> long it
        else:
            hold += 1
            exit_now = abs(zt) < exit_z
            if stop_z is not None and abs(zt) > stop_z:
                exit_now = True
            if max_hold_days is not None and hold > max_hold_days:
                exit_now = True
            # Also exit if the spread has crossed all the way through the mean
            # to the opposite band: holding through that means the original
            # thesis is gone.
            if current == 1 and zt > exit_z:
                exit_now = True
            if current == -1 and zt < -exit_z:
                exit_now = True
            if exit_now:
                current, hold = 0, 0
        pos[i] = current
    return pd.Series(pos, index=z.index, name="target_position")


# ==========================================================================
# P&L
# ==========================================================================
def run_backtest(px, spread, z, beta, entry_z, exit_z,
                 cost_per_leg_side=config.COSTS.cost_per_leg_side,
                 roll_cost_per_leg=config.COSTS.roll_cost_per_leg,
                 roll_flags=None, stop_z=None, max_hold_days=None,
                 execution_lag=0, force_flat_at_end=True,
                 roll_mode="neutralise"):
    """Run the trading rule over one window and return a per-day frame.

    All parameters (beta, the z-score's mean/std, the thresholds) must have
    been estimated OUTSIDE this window.
    """
    idx = spread.index

    # Roll flags are resolved BEFORE signal generation, because the position
    # state has to be frozen on roll days -- see generate_positions().
    if roll_flags is not None:
        rf_df = pd.DataFrame(roll_flags).reindex(idx)
        wti_roll = rf_df.get("wti_roll", pd.Series(False, index=idx)).fillna(False).astype(bool)
        brent_roll = rf_df.get("brent_roll", pd.Series(False, index=idx)).fillna(False).astype(bool)
    else:
        wti_roll = pd.Series(False, index=idx)
        brent_roll = pd.Series(False, index=idx)
    rf = wti_roll | brent_roll

    target = generate_positions(z, entry_z, exit_z, stop_z, max_hold_days,
                                freeze_days=rf)

    if force_flat_at_end and len(target):
        # We cannot carry a position across a fold boundary, because the next
        # fold re-estimates beta and the instrument itself changes. Close on
        # the last day and pay the exit cost.
        target.iloc[-1] = 0

    # `target_eff` is the position we have actually PUT ON by the close of day
    # t (execution_lag delays acting on the signal). `held` is the position
    # that is ON during day t and therefore earns day t's spread change.
    target_eff = target.shift(execution_lag).fillna(0.0)
    held = target_eff.shift(1).fillna(0.0)

    d_spread = spread.diff()

    # ---- roll neutralisation, PER LEG -----------------------------------
    # WTI and Brent roll on different dates (CL around the 21st-23rd, Brent on
    # the first business day of the month), so on a given roll day only ONE
    # leg is changing instrument. Zeroing the whole day's spread change would
    # also throw away the other leg's perfectly real price move, and charging
    # both legs a roll cost would double the true friction. We therefore
    # neutralise and charge each leg on its own roll calendar.
    #
    # Zeroing the rolling leg's move is still an approximation: the observed
    # change on a roll day is (new contract today) - (old contract yesterday),
    # which mixes a genuine move with the calendar-spread gap, and the two
    # cannot be separated without contract-level data. Setting it to zero is
    # the conservative choice -- it books no profit from a number we cannot
    # trust -- and is stated as a limitation in the README.
    d_wti = px["WTI"].reindex(idx).diff()
    d_brent = px["BRENT"].reindex(idx).diff()

    # Neutralise the WHOLE day's spread P&L whenever EITHER leg rolls.
    #
    # An earlier version of this code zeroed only the rolling leg, on the
    # reasoning that the other leg's move is genuine. That is true about the
    # leg, and wrong about the SPREAD. Zeroing one leg leaves the day's P&L
    # equal to the other leg's unhedged move -- i.e. it silently converts a
    # market-neutral spread position into an outright directional bet on
    # whichever contract did not roll, on ~10% of all days.
    #
    # The damage is not subtle. On 2022-08-01 (a Brent roll) the spread
    # genuinely moved +$5.25, while leg-wise neutralisation recorded -$4.73:
    # a $10 error on a single day, and enough to turn one 2022 trade from a
    # +$3.32 winner into a -$11.89 loser purely through accounting.
    #
    # The correct treatment is that on a day when either constituent changes
    # instrument, the spread's change is simply NOT MEASURABLE from this data.
    # We book nothing, which is conservative and market-neutral. The per-leg
    # treatment is retained for the roll COST, where it is right: you only pay
    # to roll the leg that is actually rolling.
    #
    # roll_mode gives the two BOUNDS on an effect we cannot resolve with
    # front-month data alone:
    #
    #   "neutralise" (default) -- book nothing on a roll day. Conservative:
    #       it also discards the genuine part of that day's move, so the
    #       reported P&L is biased DOWNWARD.
    #   "naive" -- book the observed change, contract gap and all. This is
    #       what a backtest that ignores rolls does, and it is biased UPWARD
    #       by the phantom P&L of the discontinuity.
    #
    # The truth lies between them. The README reports both rather than
    # presenting either as the answer.
    if roll_mode == "naive":
        d_spread_tradeable = d_wti - beta * d_brent
    else:
        d_spread_tradeable = (d_wti - beta * d_brent).where(~rf, 0.0)

    gross_pnl = held * d_spread_tradeable

    # Rolling an open position costs one bid-ask on the leg being rolled: the
    # WTI leg is 1 unit, the Brent leg is |beta| units.
    open_pos = (held != 0).astype(float)
    roll_cost = open_pos * roll_cost_per_leg * (
        wti_roll.astype(float) + brent_roll.astype(float) * abs(beta)
    )

    # ---- transaction costs ----------------------------------------------
    # One unit of |change in position| means trading 1 WTI and |beta| Brent,
    # one side each. Turnover is measured on `target_eff` (the position we put
    # ON at day t's close), not on `held`, so that the forced close on the
    # final day of the window is charged INSIDE the window instead of falling
    # off the end and quietly making the last trade of every fold free.
    turnover = target_eff.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(target_eff.iloc[0])
    trade_cost = turnover * cost_per_leg_side * (1.0 + abs(beta))

    total_cost = trade_cost + roll_cost
    net_pnl = gross_pnl - total_cost

    # ---- returns ---------------------------------------------------------
    notional = (px["WTI"].abs() + abs(beta) * px["BRENT"].abs()).reindex(idx)
    notional_lag = notional.shift(1)
    # Guard: the 2020-04-20 WTI print makes |WTI| + |beta*BRENT| small but not
    # zero; there is no division-by-zero risk, but we floor it anyway so a
    # pathological notional cannot manufacture a huge return.
    notional_lag = notional_lag.clip(lower=1.0)

    out = pd.DataFrame({
        "spread": spread,
        "z": pd.Series(z).reindex(idx),
        "target_position": target,
        "position": held,
        "d_spread": d_spread,
        "d_spread_tradeable": d_spread_tradeable,
        "gross_pnl": gross_pnl.fillna(0.0),
        "trade_cost": trade_cost.fillna(0.0),
        "roll_cost": roll_cost.fillna(0.0),
        "total_cost": total_cost.fillna(0.0),
        "net_pnl": net_pnl.fillna(0.0),
        "notional_lag": notional_lag,
        "is_roll": rf,
    })
    out["gross_ret"] = out["gross_pnl"] / out["notional_lag"]
    out["net_ret"] = out["net_pnl"] / out["notional_lag"]
    out["beta"] = beta
    return out


# ==========================================================================
# Trade extraction
# ==========================================================================
def extract_trades(bt):
    """Collapse the daily frame into one row per round-trip trade.

    Needed for the hit rate (a per-TRADE statistic, not a per-day one) and for
    the average holding period, which we sanity-check against the OU half-life.
    """
    pos = bt["position"].values
    idx = bt.index
    trades = []
    i = 0
    n = len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        side = pos[i]
        start = i
        while i < n and pos[i] == side:
            i += 1
        end = i - 1
        sl = bt.iloc[start:end + 1]
        trades.append({
            "entry_date": idx[start],
            "exit_date": idx[end],
            "side": "long_spread" if side > 0 else "short_spread",
            "holding_days": end - start + 1,
            "entry_z": float(bt["z"].iloc[start - 1]) if start > 0 else np.nan,
            "exit_z": float(bt["z"].iloc[end]),
            "entry_spread": float(bt["spread"].iloc[start - 1]) if start > 0 else np.nan,
            "exit_spread": float(bt["spread"].iloc[end]),
            "gross_pnl": float(sl["gross_pnl"].sum()),
            "cost": float(sl["total_cost"].sum()),
            "net_pnl": float(sl["net_pnl"].sum()),
        })
    return pd.DataFrame(trades)
