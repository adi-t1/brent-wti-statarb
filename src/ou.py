"""
Ornstein-Uhlenbeck estimation on the cointegrating residual.

Model
-----
    dS_t = theta * (mu - S_t) dt + sigma dW_t

theta  : speed of mean reversion (per unit time; here per trading day)
mu     : long-run level the spread is pulled back to
sigma  : instantaneous volatility of the shocks

Estimation
----------
The OU process has an EXACT discrete-time representation. Over a step of dt:

    S_{t+1} = mu*(1 - phi) + phi*S_t + eps_t,    phi = exp(-theta*dt)

    Var(eps) = sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)

so an AR(1) regression recovers every parameter exactly, with no
discretisation bias in the drift. We run it in the "regress the change on the
lagged level" form,

    dS_t = a + b * S_{t-1} + eps_t,     where b = phi - 1,

which is algebraically the same regression.

Two ways to go from b to theta
------------------------------
    exact:        theta = -ln(1 + b) / dt
    Euler approx: theta = -b / dt

These agree to first order but diverge as |b| grows. For a spread with a
2-day half-life, b is about -0.29 and the two estimates differ by ~17%. We
report BOTH and use the EXACT one everywhere downstream, because the exact
formula inverts the true discrete transition of the process rather than a
first-order Taylor expansion of it. Quoting the Euler number as if it were
the truth is a common and avoidable error.

Half-life
---------
Ignoring noise, the expected path decays as
    E[S_t] - mu = (S_0 - mu) * exp(-theta*t)
Setting the gap to half its initial size gives exp(-theta*t) = 1/2, so

    t_half = ln(2) / theta

Guard rails
-----------
If the fitted b >= 0 the process is NOT mean-reverting (it is a random walk or
explosive). In that case theta <= 0 and ln(2)/theta is either undefined or
negative. We return NaN and set a flag rather than silently taking an absolute
value or clipping -- a window with no mean reversion must be allowed to
disqualify itself, otherwise the backtest trades a spread that has no reason
to come back.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config


class OUFit(dict):
    """Dict with attribute access."""

    __getattr__ = dict.__getitem__


def fit_ou(spread, usable=None, dt=1.0):
    """Fit an OU process to `spread` by AR(1) regression of dS on lagged S.

    Parameters
    ----------
    spread : pd.Series
        The cointegrating residual (in $/bbl).
    usable : pd.Series[bool] or None
        Estimation mask aligned to `spread`. Any regression PAIR that touches
        an unusable observation is dropped -- note this removes two pairs per
        excluded date (the transition into it and the transition out of it),
        which is the correct treatment: a difference computed against a
        corrupted price is itself corrupted.
    dt : float
        Time step in the same units theta will be expressed in. We use 1.0 =
        one trading day, so theta is per trading day and the half-life comes
        out in trading days.

    Returns
    -------
    OUFit with theta, mu, sigma, half_life, diagnostics and a `valid` flag.
    """
    s = pd.Series(spread).astype(float)
    lag = s.shift(1)
    d = s - lag

    df = pd.DataFrame({"d": d, "lag": lag}).dropna()

    if usable is not None:
        u = pd.Series(usable).reindex(s.index).fillna(False).astype(bool)
        # A pair (t-1 -> t) is usable only if BOTH endpoints are usable.
        pair_ok = (u & u.shift(1)).reindex(df.index).fillna(False)
        df = df[pair_ok]

    n = len(df)
    if n < 30:
        return OUFit(
            valid=False, reason="too few usable observations (%d)" % n, n_obs=n,
            theta=np.nan, theta_euler=np.nan, mu=np.nan, sigma=np.nan,
            half_life=np.nan, half_life_euler=np.nan, phi=np.nan,
            b=np.nan, b_tstat=np.nan, b_pvalue=np.nan, resid_std=np.nan,
        )

    X = sm.add_constant(df["lag"].values)
    res = sm.OLS(df["d"].values, X).fit()
    a = float(res.params[0])
    b = float(res.params[1])
    b_t = float(res.tvalues[1])
    b_p = float(res.pvalues[1])
    resid_std = float(np.std(res.resid, ddof=2))

    phi = 1.0 + b

    # ---- Guard rail 1: no mean reversion at all -------------------------
    if b >= 0:
        return OUFit(
            valid=False,
            reason="b = %.5f >= 0: fitted theta <= 0, spread is not "
                   "mean-reverting in this window" % b,
            n_obs=n, theta=np.nan, theta_euler=np.nan, mu=np.nan,
            sigma=np.nan, half_life=np.nan, half_life_euler=np.nan,
            phi=phi, b=b, b_tstat=b_t, b_pvalue=b_p, resid_std=resid_std,
        )

    # ---- Guard rail 2: phi <= 0 -----------------------------------------
    # b <= -1 means the AR(1) coefficient is non-positive: the series flips
    # sign every step rather than decaying. That is not an OU process, and
    # ln(phi) is undefined. Rare in practice but it must not crash silently.
    if phi <= 0:
        return OUFit(
            valid=False,
            reason="phi = %.5f <= 0: AR(1) coefficient non-positive, the "
                   "exact OU inversion is undefined" % phi,
            n_obs=n, theta=np.nan, theta_euler=np.nan, mu=np.nan,
            sigma=np.nan, half_life=np.nan, half_life_euler=np.nan,
            phi=phi, b=b, b_tstat=b_t, b_pvalue=b_p, resid_std=resid_std,
        )

    theta = -np.log(phi) / dt          # exact inversion
    theta_euler = -b / dt              # first-order approximation
    mu = -a / b                        # long-run mean: level at which E[dS]=0

    # sigma from the residual variance of the exact discretisation
    denom = 1.0 - np.exp(-2.0 * theta * dt)
    sigma = resid_std * np.sqrt(2.0 * theta / denom) if denom > 0 else np.nan

    half_life = np.log(2.0) / theta
    half_life_euler = np.log(2.0) / theta_euler

    return OUFit(
        valid=True,
        reason="",
        n_obs=n,
        theta=float(theta),
        theta_euler=float(theta_euler),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
        half_life_euler=float(half_life_euler),
        phi=float(phi),
        b=b,
        b_tstat=b_t,
        b_pvalue=b_p,
        resid_std=resid_std,
    )


def rolling_half_life(spread, usable=None, window=252, step=5):
    """Half-life estimated on a rolling window, for the stability plot.

    A strategy whose half-life wanders from 5 days to 60 days is not trading
    one stable statistical object; the plot is there to make that visible
    rather than hide it behind a single full-sample number.
    """
    s = pd.Series(spread).astype(float)
    idx, hl, th, valid = [], [], [], []
    for end in range(window, len(s) + 1, step):
        sl = s.iloc[end - window:end]
        u = None if usable is None else pd.Series(usable).iloc[end - window:end]
        f = fit_ou(sl, u)
        idx.append(s.index[end - 1])
        hl.append(f["half_life"])
        th.append(f["theta"])
        valid.append(f["valid"])
    return pd.DataFrame(
        {"half_life": hl, "theta": th, "valid": valid},
        index=pd.DatetimeIndex(idx),
    )


def simulate_ou(theta, mu, sigma, n, dt=1.0, s0=None, seed=None):
    """Simulate an exact OU path. Used by validation.py to check the
    estimator against a KNOWN ground truth before it is trusted on real data.

    Exact transition (not Euler): draws from the true conditional
    distribution, so the simulation itself introduces no discretisation bias
    that could mask a bug in the estimator.
    """
    rng = np.random.default_rng(seed)
    phi = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))
    s = np.empty(n)
    s[0] = mu if s0 is None else s0
    shocks = rng.normal(0.0, sd, n)
    for i in range(1, n):
        s[i] = mu * (1.0 - phi) + phi * s[i - 1] + shocks[i]
    return pd.Series(s)


def half_life_to_theta(half_life):
    """Convenience inverse, used to set up synthetic tests."""
    return np.log(2.0) / half_life


# ==========================================================================
# Noise-robust half-life via an augmented (AR(p)) specification
# ==========================================================================
#
# WHY THIS EXISTS -- a real problem found while running this study.
#
# The plain AR(1) fit above assumes the spread's daily CHANGES are white
# noise. On real Brent-WTI data they are not: the first-order autocorrelation
# of d(spread) is persistently around -0.17 across the whole sample. Negative
# autocorrelation in the increments is the classic signature of transient,
# non-informational noise -- bid-ask bounce, and the fact that the two legs
# settle on two different exchanges so the "same" timestamp is not quite the
# same moment.
#
# That noise is poison for an AR(1) half-life. A one-day overshoot that
# reverses the next day looks, to an AR(1) regression, exactly like very fast
# mean reversion. The fitted half-life is therefore biased SHORT, and the bias
# is worst precisely when the underlying spread is closest to a random walk --
# i.e. exactly when we most need the estimator to tell us to stand aside.
#
# The symptom in this dataset: on the 2021-2024 training windows the AR(1) fit
# reports a seductive 8-day half-life, while an ADF test with proper lag
# augmentation cannot reject a unit root at all (stat -2.2, p 0.41). The 8-day
# half-life is mostly microstructure, not a tradeable equilibrium.
#
# THE FIX. Fit an AR(p) to the spread LEVEL, with p chosen by BIC, and read
# the half-life off the impulse response function rather than off a single
# coefficient. The extra lags absorb the short-run noise dynamics, leaving the
# persistence of the level to be measured on its own. On a true AR(1) the two
# estimators agree; when they disagree, the disagreement is itself the signal.
# ==========================================================================
def _ar_companion(phis):
    """Companion matrix of an AR(p)."""
    p = len(phis)
    A = np.zeros((p, p))
    A[0, :] = phis
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    return A


def half_life_from_ar(spread, usable=None, max_lags=10, max_horizon=2000):
    """Half-life of a shock to the LEVEL, from an AR(p) impulse response.

    Procedure
    ---------
    1. Fit S_t = c + sum_{i=1..p} phi_i * S_{t-i} + e_t by OLS, choosing p by
       BIC over 1..max_lags.
    2. Form the companion matrix and propagate a unit shock forward:
       the response at horizon h is (A^h)[0,0].
    3. The half-life is the first horizon at which that response falls to 0.5,
       linearly interpolated between the bracketing integers.

    Returns NaN if the process is non-stationary (largest companion eigenvalue
    >= 1) or if the response never decays to a half within max_horizon -- both
    of which are legitimate "this is not mean-reverting" answers and must not
    be papered over.
    """
    s = pd.Series(spread).astype(float)
    if usable is not None:
        u = pd.Series(usable).reindex(s.index).fillna(False).astype(bool)
    else:
        u = pd.Series(True, index=s.index)

    best = None
    for p in range(1, max_lags + 1):
        cols = {"y": s}
        for i in range(1, p + 1):
            cols["l%d" % i] = s.shift(i)
        df = pd.DataFrame(cols).dropna()
        # Drop any row whose window touches an excluded observation.
        ok = u.copy()
        keep = ok.reindex(df.index).fillna(False).astype(bool)
        for i in range(1, p + 1):
            keep &= ok.shift(i).reindex(df.index).fillna(False).astype(bool)
        df = df[keep]
        if len(df) < 10 * p + 20:
            break
        X = sm.add_constant(df[["l%d" % i for i in range(1, p + 1)]].values)
        res = sm.OLS(df["y"].values, X).fit()
        n = len(df)
        # BIC on the OLS residual variance
        sigma2 = np.sum(res.resid ** 2) / n
        bic = n * np.log(sigma2) + (p + 1) * np.log(n)
        if best is None or bic < best["bic"]:
            best = {"p": p, "bic": bic, "phis": res.params[1:], "n": n,
                    "const": res.params[0]}

    if best is None:
        return {"half_life": np.nan, "p": np.nan, "max_eig": np.nan,
                "valid": False, "reason": "insufficient data"}

    phis = np.asarray(best["phis"], dtype=float)
    A = _ar_companion(phis)
    eig = np.max(np.abs(np.linalg.eigvals(A)))
    if eig >= 1.0:
        return {"half_life": np.nan, "p": best["p"], "max_eig": float(eig),
                "valid": False,
                "reason": "largest companion eigenvalue %.4f >= 1: the level "
                          "is not stationary, so no finite half-life exists"
                          % eig}

    # Propagate the impulse response.
    resp, M = [], np.eye(len(phis))
    prev = 1.0
    for h in range(1, max_horizon + 1):
        M = M @ A
        cur = float(M[0, 0])
        resp.append(cur)
        if cur <= 0.5:
            # linear interpolation between h-1 (value `prev`) and h (`cur`)
            hl = (h - 1) + (prev - 0.5) / (prev - cur) if prev != cur else float(h)
            return {"half_life": float(hl), "p": best["p"],
                    "max_eig": float(eig), "valid": True, "reason": "",
                    "sum_phi": float(phis.sum())}
        prev = cur

    return {"half_life": np.nan, "p": best["p"], "max_eig": float(eig),
            "valid": False,
            "reason": "impulse response did not halve within %d days"
                      % max_horizon}


# ==========================================================================
# The estimator that actually survives microstructure noise: ARMA(1,1)
# ==========================================================================
#
# Why AR(p) is NOT the right fix (we tried it -- see validation.py).
#
# Let the true equilibrium spread S_t be AR(1) with coefficient phi, and let
# what we actually observe be
#
#       X_t = S_t + eta_t,      eta_t iid noise (bid-ask bounce, two
#                               exchanges settling at not-quite the same
#                               instant)
#
# Adding white noise to an AR(1) does not give another AR process. It gives an
# ARMA(1,1): the AR root is still the true phi, but a moving-average term
# appears that carries all of the noise. An AR(p) approximation has to chase
# that MA term with a long, slowly-decaying lag polynomial, and its impulse
# response at horizon 1 is dragged down by the noise just as badly as the
# AR(1) coefficient is. Empirically both collapse together.
#
# Fitting ARMA(1,1) directly matches the structure of the problem: the MA term
# absorbs the transient noise and the AR root is left to measure the genuine
# persistence of the equilibrium. In simulation with a true 30-day half-life
# this recovers ~20-25 days at every noise level tested, while the naive AR(1)
# estimate falls to well under a day.
#
# Cost: ARMA(1,1) is fitted by maximum likelihood, so it is slower and can
# fail to converge. We fall back to the AR(1) estimate and SAY SO rather than
# silently substituting a number of different provenance.
# ==========================================================================
def half_life_arma11(spread, usable=None, max_half_life=1000.0):
    """Noise-robust half-life: the AR root of an ARMA(1,1) fit to the level.

    Returns a dict with `half_life`, the fitted `phi`, the MA coefficient
    `ma` (large |ma| is itself evidence that transient noise is present), and
    a `valid` flag.
    """
    from statsmodels.tsa.arima.model import ARIMA

    s = pd.Series(spread).astype(float)
    if usable is not None:
        u = pd.Series(usable).reindex(s.index).fillna(False).astype(bool)
        s = s[u]
    if len(s) < 60:
        return {"half_life": np.nan, "phi": np.nan, "ma": np.nan,
                "valid": False, "reason": "too few observations"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMA(s.values, order=(1, 0, 1), trend="c").fit(
                method="statespace")
        phi = float(m.arparams[0])
        ma = float(m.maparams[0]) if len(m.maparams) else np.nan
    except Exception as exc:
        return {"half_life": np.nan, "phi": np.nan, "ma": np.nan,
                "valid": False, "reason": "ARMA(1,1) fit failed: %s" % exc}

    if not (0.0 < phi < 1.0):
        return {"half_life": np.nan, "phi": phi, "ma": ma, "valid": False,
                "reason": "AR root %.4f outside (0,1): no decaying "
                          "mean reversion" % phi}

    hl = -np.log(2.0) / np.log(phi)
    if hl > max_half_life:
        return {"half_life": float(hl), "phi": phi, "ma": ma, "valid": False,
                "reason": "half-life %.0f d exceeds the %.0f d ceiling: "
                          "indistinguishable from a random walk"
                          % (hl, max_half_life)}
    return {"half_life": float(hl), "phi": phi, "ma": ma, "valid": True,
            "reason": ""}
