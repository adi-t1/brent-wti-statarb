"""
Validation of the estimators against synthetic data with KNOWN ground truth.

Why this file exists
--------------------
Every number in the rest of this project comes out of an estimator. If the
estimator is wrong, the research is wrong, and no amount of careful backtesting
will reveal it -- a buggy half-life will happily produce a plausible-looking
equity curve. So before any of these functions touch real crude oil data we
point them at simulated data whose true parameters we chose ourselves, and
check that they recover them.

Four checks, in increasing order of what they would catch:

1. POWER / RECOVERY. Simulate an OU process with a known half-life. Does
   fit_ou() get it back? This catches sign errors, dt errors, and the classic
   Euler-vs-exact discretisation mistake.

2. COINTEGRATION RECOVERY. Simulate two series that are cointegrated by
   construction with a known beta. Does Engle-Granger recover beta, and does
   the test reject "no cointegration"?

3. SIZE / FALSE POSITIVES. Simulate two INDEPENDENT random walks. There is no
   relationship at all. A test with correct size should reject the null about
   5% of the time at the 5% level. If it rejects far more often, the test is
   finding cointegration that is not there -- the spurious regression problem.

4. THE JOHANSEN FAT-TAIL PROBLEM. Repeat check 3, but drive the random walks
   with GARCH(1,1) innovations instead of iid Gaussian ones -- i.e. with the
   volatility clustering that real crude oil returns actually display. The
   Johansen trace test's asymptotic critical values assume iid Gaussian
   errors. This check measures, empirically, how badly the r<=1 rejection rate
   inflates when that assumption fails, and is the direct justification for
   the rank override implemented in cointegration.decide_rank().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cointegration as ci
from . import config
from . import ou


# ==========================================================================
# 1. OU half-life recovery
# ==========================================================================
def validate_ou_half_life(true_half_lives=(5.0, 10.0, 21.0, 42.0),
                          n_obs=config.WF.train_days,
                          n_reps=400,
                          sigma=1.0,
                          mu=0.0,
                          seed=config.RANDOM_SEED):
    """Can fit_ou() recover a half-life we chose ourselves?

    We use n_obs = the actual training window length, because small-sample
    bias is a function of sample size and we care about the bias at the size
    we will really be using, not asymptotically.

    Expected result: a small DOWNWARD bias in the estimated half-life. This is
    not a bug -- it is the well-known downward bias of the OLS AR(1)
    coefficient in a mean-reverting series (Kendall / Marriott-Pope). It means
    reported half-lives are, if anything, slightly optimistic about the speed
    of reversion, which we flag rather than correct.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for hl in true_half_lives:
        theta_true = ou.half_life_to_theta(hl)
        ests, thetas, n_invalid = [], [], 0
        euler_ests = []
        for _ in range(n_reps):
            s = ou.simulate_ou(theta_true, mu, sigma, n_obs,
                               seed=int(rng.integers(1 << 31)))
            f = ou.fit_ou(s)
            if not f["valid"]:
                n_invalid += 1
                continue
            ests.append(f["half_life"])
            euler_ests.append(f["half_life_euler"])
            thetas.append(f["theta"])
        ests = np.array(ests)
        rows.append({
            "true_half_life": hl,
            "true_theta": theta_true,
            "mean_est_half_life": ests.mean(),
            "median_est_half_life": np.median(ests),
            "bias_pct": 100.0 * (ests.mean() - hl) / hl,
            "mean_est_half_life_euler": float(np.mean(euler_ests)),
            "euler_bias_pct": 100.0 * (np.mean(euler_ests) - hl) / hl,
            "std_est": ests.std(ddof=1),
            "mean_est_theta": float(np.mean(thetas)),
            "n_invalid_fits": n_invalid,
            "n_reps": n_reps,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 2. Cointegration recovery (power)
# ==========================================================================
def _simulate_cointegrated(n, beta, alpha=0.0, half_life=10.0,
                           spread_sigma=1.0, x_sigma=1.0, x0=50.0, rng=None):
    """x is a random walk; y = alpha + beta*x + (OU spread).

    By construction x and y are each I(1) and y - beta*x is stationary, so the
    true cointegrating vector is (1, -beta) and the true rank is 1.
    """
    rng = rng or np.random.default_rng()
    x = x0 + np.cumsum(rng.normal(0, x_sigma, n))
    theta = ou.half_life_to_theta(half_life)
    s = ou.simulate_ou(theta, 0.0, spread_sigma, n,
                       seed=int(rng.integers(1 << 31))).values
    y = alpha + beta * x + s
    return pd.DataFrame({"Y": y, "X": x})


def validate_cointegration_power(n_obs=config.WF.train_days, n_reps=300,
                                 beta=0.9, half_life=10.0,
                                 seed=config.RANDOM_SEED + 1):
    """Two series that ARE cointegrated. How often do the tests notice?"""
    rng = np.random.default_rng(seed)
    eg_rej, joh_rank1, betas = 0, 0, []
    for _ in range(n_reps):
        df = _simulate_cointegrated(n_obs, beta, half_life=half_life, rng=rng)
        e = ci.engle_granger(df["Y"], df["X"], "Y", "X")
        betas.append(e["beta"])
        if e["eg_pvalue"] < 0.05:
            eg_rej += 1
        j = ci.johansen(df, config.JOHANSEN_DET_ORDER, 1)
        if j["rank_mechanical"] >= 1:
            joh_rank1 += 1
    betas = np.array(betas)
    return {
        "true_beta": beta,
        "mean_est_beta": float(betas.mean()),
        "std_est_beta": float(betas.std(ddof=1)),
        "eg_power_pct": 100.0 * eg_rej / n_reps,
        "johansen_detect_rank_ge1_pct": 100.0 * joh_rank1 / n_reps,
        "n_reps": n_reps,
        "n_obs": n_obs,
    }


# ==========================================================================
# 3 & 4. Size under iid and under GARCH innovations
# ==========================================================================
def _garch11_innovations(n, omega=0.05, a=0.09, b=0.90, rng=None):
    """GARCH(1,1) innovations: volatility clusters, unconditional kurtosis > 3.

    a + b = 0.99 is deliberately close to 1 -- that is the persistence level
    routinely estimated on daily crude oil returns, so this is a realistic
    stress, not a caricature.
    """
    rng = rng or np.random.default_rng()
    z = rng.normal(0, 1, n)
    eps = np.empty(n)
    h = omega / max(1e-12, (1 - a - b))
    for i in range(n):
        eps[i] = np.sqrt(h) * z[i]
        h = omega + a * eps[i] ** 2 + b * h
    return eps


def validate_test_size(n_obs=config.WF.train_days, n_reps=400,
                       innovation="iid", seed=config.RANDOM_SEED + 2):
    """Two INDEPENDENT random walks -- no cointegration exists.

    A correctly sized 5% test should reject about 5% of the time. Anything
    materially above that is a false-positive machine.

    We report the Johansen r<=1 rejection rate separately, because that is the
    specific hypothesis whose spurious rejection would push us to the
    economically absurd conclusion that both price levels are stationary.
    """
    rng = np.random.default_rng(seed)
    eg_rej = 0
    joh_r0_rej = 0
    joh_r1_rej = 0
    joh_rank2 = 0
    for _ in range(n_reps):
        if innovation == "iid":
            e1 = rng.normal(0, 1, n_obs)
            e2 = rng.normal(0, 1, n_obs)
        else:
            e1 = _garch11_innovations(n_obs, rng=rng)
            e2 = _garch11_innovations(n_obs, rng=rng)
        x = 50 + np.cumsum(e1)
        y = 50 + np.cumsum(e2)
        df = pd.DataFrame({"Y": y, "X": x})

        e = ci.engle_granger(df["Y"], df["X"], "Y", "X")
        if e["eg_pvalue"] < 0.05:
            eg_rej += 1

        j = ci.johansen(df, config.JOHANSEN_DET_ORDER, 1)
        t = j["trace"]
        if bool(t.loc[0, "reject"]):
            joh_r0_rej += 1
        if bool(t.loc[1, "reject"]):
            joh_r1_rej += 1
        if j["rank_mechanical"] >= 2:
            joh_rank2 += 1

    return {
        "innovation": innovation,
        "n_obs": n_obs,
        "n_reps": n_reps,
        "eg_false_positive_pct": 100.0 * eg_rej / n_reps,
        "johansen_r0_reject_pct": 100.0 * joh_r0_rej / n_reps,
        "johansen_r1_reject_pct": 100.0 * joh_r1_rej / n_reps,
        "johansen_rank2_pct": 100.0 * joh_rank2 / n_reps,
        "nominal_pct": 5.0,
    }


# ==========================================================================
# Runner
# ==========================================================================
def run_all(n_reps_ou=400, n_reps_coint=300, n_reps_size=400, verbose=True):
    """Run the full validation suite and return a dict of result objects."""
    out = {}

    out["ou"] = validate_ou_half_life(n_reps=n_reps_ou)
    out["power"] = validate_cointegration_power(n_reps=n_reps_coint)
    out["size_iid"] = validate_test_size(n_reps=n_reps_size, innovation="iid")
    out["size_garch"] = validate_test_size(n_reps=n_reps_size, innovation="garch")

    if verbose:
        print("=" * 74)
        print("ESTIMATOR VALIDATION AGAINST SYNTHETIC DATA (known ground truth)")
        print("=" * 74)
        print("\n[1] OU half-life recovery (n_obs = %d = one training window)"
              % config.WF.train_days)
        cols = ["true_half_life", "mean_est_half_life", "median_est_half_life",
                "bias_pct", "mean_est_half_life_euler", "euler_bias_pct",
                "std_est", "n_invalid_fits"]
        print(out["ou"][cols].to_string(index=False,
                                        float_format=lambda v: "%8.3f" % v))

        p = out["power"]
        print("\n[2] Cointegration recovery on truly cointegrated series")
        print("    true beta               : %.4f" % p["true_beta"])
        print("    mean estimated beta     : %.4f  (sd %.4f)"
              % (p["mean_est_beta"], p["std_est_beta"]))
        print("    Engle-Granger power     : %.1f%% of runs reject 'no coint'"
              % p["eg_power_pct"])
        print("    Johansen finds rank>=1  : %.1f%% of runs"
              % p["johansen_detect_rank_ge1_pct"])

        print("\n[3/4] Test SIZE on two INDEPENDENT random walks "
              "(truth: no cointegration)")
        rows = []
        for key in ("size_iid", "size_garch"):
            s = out[key]
            rows.append({
                "innovations": s["innovation"],
                "EG false pos %": s["eg_false_positive_pct"],
                "Joh r<=0 rej %": s["johansen_r0_reject_pct"],
                "Joh r<=1 rej %": s["johansen_r1_reject_pct"],
                "Joh says rank=2 %": s["johansen_rank2_pct"],
                "nominal %": s["nominal_pct"],
            })
        print(pd.DataFrame(rows).to_string(index=False))
    return out


# ==========================================================================
# 5. Half-life estimation under microstructure noise
# ==========================================================================
def validate_half_life_under_noise(true_half_life=30.0,
                                   noise_ratios=(0.0, 0.25, 0.5, 1.0, 2.0),
                                   n_obs=config.WF.train_days,
                                   n_reps=200,
                                   sigma=1.0,
                                   seed=config.RANDOM_SEED + 3):
    """How badly does transient noise corrupt a half-life estimate?

    Setup: a true OU process with a KNOWN 30-day half-life, observed with
    additive iid measurement noise (bid-ask bounce / non-synchronous
    settlement between two exchanges). `noise_ratio` is the noise standard
    deviation as a fraction of the OU process's own stationary standard
    deviation.

    The additive noise is serially uncorrelated, so it contributes a large
    transient component that reverses immediately. An AR(1) regression cannot
    distinguish "reverted because the equilibrium pulled it back" from
    "reverted because yesterday's print was a bid and today's is an offer",
    and reads the second as very fast mean reversion.

    Three estimators are compared, and the ORDER OF THE RESULT IS THE POINT:

      naive AR(1)   -- collapses towards zero as noise rises.
      AR(p) via IRF -- our first attempt at a fix. It ALSO collapses. Adding
                       white noise to an AR(1) does not produce a longer AR
                       process; it produces an ARMA(1,1). Chasing an MA term
                       with a long AR lag polynomial is the wrong shape of
                       tool, and the horizon-1 impulse response is dragged
                       down by the noise exactly as the AR(1) coefficient is.
      ARMA(1,1)     -- fits the structure of the problem. The MA term absorbs
                       the transient noise and the AR root measures the
                       genuine persistence. This one survives.

    We keep the failed AR(p) column in the output deliberately: it documents
    that the obvious fix was tried and rejected on evidence, rather than
    leaving the reader to wonder why a more complicated estimator was needed.

    This is the synthetic proof of the effect we observe on real Brent-WTI
    data, where d(spread) has a lag-1 autocorrelation of about -0.17 and the
    fitted MA coefficient is consistently around -0.15.
    """
    rng = np.random.default_rng(seed)
    theta = ou.half_life_to_theta(true_half_life)
    stat_sd = sigma / np.sqrt(2.0 * theta)   # stationary sd of the OU process

    rows = []
    for nr in noise_ratios:
        naive, robust, arma = [], [], []
        for _ in range(n_reps):
            s = ou.simulate_ou(theta, 0.0, sigma, n_obs,
                               seed=int(rng.integers(1 << 31)))
            obs = s + rng.normal(0.0, nr * stat_sd, n_obs)
            f = ou.fit_ou(obs)
            if f["valid"]:
                naive.append(f["half_life"])
            g = ou.half_life_from_ar(obs)
            if g["valid"]:
                robust.append(g["half_life"])
            h = ou.half_life_arma11(pd.Series(obs))
            if h["valid"]:
                arma.append(h["half_life"])
        rows.append({
            "noise_ratio": nr,
            "true_half_life": true_half_life,
            "naive_AR1_half_life": float(np.median(naive)) if naive else np.nan,
            "naive_bias_pct": (100.0 * (np.median(naive) - true_half_life)
                               / true_half_life) if naive else np.nan,
            "ARp_IRF_half_life": float(np.median(robust)) if robust else np.nan,
            "ARp_bias_pct": (100.0 * (np.median(robust) - true_half_life)
                             / true_half_life) if robust else np.nan,
            "ARMA11_half_life": float(np.median(arma)) if arma else np.nan,
            "ARMA11_bias_pct": (100.0 * (np.median(arma) - true_half_life)
                                / true_half_life) if arma else np.nan,
            "n_reps": n_reps,
        })
    return pd.DataFrame(rows)
