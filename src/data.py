"""
Data acquisition and cleaning for the Brent-WTI study.

Responsibilities
----------------
1. Pull daily front-month futures closes from Yahoo Finance, with an on-disk
   cache so the whole study is reproducible offline.
2. Align the two series onto a common trading calendar.
3. Build an explicit boolean "usable for estimation" mask that removes the
   2020-04-20 negative print (see config for the full rationale) WITHOUT
   deleting it from the raw series used for plotting.
4. Estimate futures roll dates, because a naive continuous front-month series
   contains price discontinuities that are NOT tradeable P&L.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config


# ==========================================================================
# 1. Download / cache
# ==========================================================================
def _download_close(ticker: str, start: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("No data returned for " + ticker)
    close = df["Close"]
    # yfinance returns a MultiIndex column frame for single tickers in recent
    # versions; normalise to a plain Series either way.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = ticker
    return close.sort_index()


def load_prices(use_cache=True, refresh=False, start=config.START_DATE):
    """Return a DataFrame with columns ['WTI', 'BRENT'] on a common calendar.

    Note on 'Close' vs 'Adj Close': for futures these are identical (there are
    no dividends or splits to adjust for), and Yahoo's 'Adj Close' on a
    continuous futures series is NOT roll-adjusted. We use raw Close and handle
    rolls explicitly rather than trusting an adjustment that does not exist.
    """
    cache = config.CACHE_DIR / "prices_raw.csv"
    if use_cache and cache.exists() and not refresh:
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        wti = _download_close(config.WTI_TICKER, start)
        brent = _download_close(config.BRENT_TICKER, start)
        px = pd.concat([wti.rename("WTI"), brent.rename("BRENT")], axis=1)
        px.to_csv(cache)

    px = px[["WTI", "BRENT"]].copy()

    # Restrict to days where BOTH markets printed. Brent (BZ=F) only starts
    # 2007-07-30 on Yahoo, and the two exchanges keep different holidays. We
    # require both legs because a spread trade needs both prices to mark; we do
    # NOT forward-fill, because a forward-filled price implies a zero return on
    # a day the market actually moved, which biases the measured volatility of
    # the spread downwards and therefore inflates z-scores and the Sharpe.
    # Yahoo returns a provisional bar for the session in progress (its volume
    # field is typically a stale copy of the prior day). Including a partial
    # close would make the last observation non-comparable to the rest, so we
    # drop any row dated today or later.
    px = px[px.index < pd.Timestamp.today().normalize()]

    n_before = len(px)
    px = px.dropna()
    dropped = n_before - len(px)
    if dropped:
        warnings.warn(
            "Dropped %d rows where one leg was missing (mostly pre-%s Brent "
            "history and non-overlapping exchange holidays)."
            % (dropped, px.index.min().date())
        )
    return px


# ==========================================================================
# 2. Estimation mask
# ==========================================================================
def estimation_mask(index, exclude_dates=config.ESTIMATION_EXCLUDE_DATES):
    """Boolean Series: True where an observation may be used to FIT parameters.

    Kept separate from the price frame on purpose. Every estimator in this
    project (OLS hedge ratio, ADF, Johansen, OU) takes prices plus this mask,
    so the exclusion is applied consistently and is auditable in one place,
    rather than each module silently doing its own filtering.
    """
    mask = pd.Series(True, index=index, name="usable")
    for d in exclude_dates:
        ts = pd.Timestamp(d)
        if ts in mask.index:
            mask.loc[ts] = False
    return mask


# ==========================================================================
# 3. Futures roll dates
# ==========================================================================
def _n_index_days_before(index, anchor, n):
    """Step back n positions in `index` from the last index date <= anchor."""
    pos = index.searchsorted(anchor, side="right") - 1
    if pos < 0:
        return None
    tgt = pos - n
    return index[tgt] if tgt >= 0 else None


def estimate_roll_days(index):
    """Approximate the dates on which a continuous front-month series switches
    contract, using each exchange's published expiry rule.

    NYMEX WTI (CL): trading terminates 3 business days before the 25th calendar
        day of the month PRECEDING the delivery month.
    ICE Brent (BZ): trading ceases at the end of the last business day of the
        SECOND month preceding the delivery month.

    The "business day" universe used here is the set of dates actually present
    in our price index. That is self-consistent (no external holiday-calendar
    dependency) and accurate to within a day, which is all we need: these flags
    are used to NEUTRALISE P&L around rolls, so being a day early or late costs
    a little conservatism, not correctness.

    Returns a frame with boolean columns 'wti_roll' and 'brent_roll', True on
    the first trading day on which the quoted contract has changed.
    """
    index = pd.DatetimeIndex(index)
    wti_roll = pd.Series(False, index=index)
    brent_roll = pd.Series(False, index=index)

    months = pd.period_range(index.min(), index.max(), freq="M")
    for m in months:
        # --- WTI: expiry 3 business days before the 25th of this month ---
        anchor = pd.Timestamp(year=m.year, month=m.month, day=25)
        exp = _n_index_days_before(index, anchor, 3)
        if exp is not None and exp >= index.min():
            pos = index.searchsorted(exp, side="right")
            if pos < len(index):
                wti_roll.iloc[pos] = True

        # --- Brent: expiry = last trading day of this month ---
        mstart = pd.Timestamp(m.start_time.date())
        mend = pd.Timestamp(m.end_time.date())
        in_month = index[(index >= mstart) & (index <= mend)]
        if len(in_month):
            pos = index.searchsorted(in_month[-1], side="right")
            if pos < len(index):
                brent_roll.iloc[pos] = True

    out = pd.DataFrame({"wti_roll": wti_roll, "brent_roll": brent_roll})
    out["any_roll"] = out["wti_roll"] | out["brent_roll"]
    return out


def roll_jump_diagnostics(px, rolls):
    """Quantify how much of the series' day-to-day variation sits on roll days.

    If the mean absolute move on flagged roll days is materially larger than on
    ordinary days, the continuous series really does contain untradeable
    discontinuities and neutralising them is not optional.
    """
    d = px.diff()
    rows = []
    for leg, flag in (("WTI", "wti_roll"), ("BRENT", "brent_roll")):
        on = d.loc[rolls[flag], leg].abs()
        off = d.loc[~rolls[flag], leg].abs()
        rows.append(
            {
                "leg": leg,
                "n_roll_days": int(rolls[flag].sum()),
                "mean_abs_move_roll": on.mean(),
                "mean_abs_move_other": off.mean(),
                "ratio": on.mean() / off.mean() if off.mean() else np.nan,
                "median_abs_roll": on.median(),
                "median_abs_other": off.median(),
            }
        )
    return pd.DataFrame(rows)
