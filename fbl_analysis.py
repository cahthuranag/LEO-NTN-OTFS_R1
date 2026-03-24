"""
Finite-Blocklength BLER Analysis (Section III)
===============================================
Implements finite-blocklength error probability analysis for
LEO NTN OTFS systems with diversity transforms.

Provides:
  - PPV normal approximation (Polyanskiy-Poor-Verdu)
  - Conditional BLER epsilon(gamma) — Eq. (41)
  - Closed-form BLER with shifted gamma distribution — Proposition 1 / Eq. (47)
  - High-SNR approximation revealing diversity order — Proposition 2 / Eq. (49)
  - Semi-analytical BLER for no-diversity and fixed diversity
  - Vectorized PPV bounds for batch computation

Author: Research implementation
Date: 2026-02-21
"""

import numpy as np
import math
from typing import Optional

from scipy.special import erfc


# ============================================================================
#  Helper functions
# ============================================================================

def _gammainc_series(a, x):
    """Series for lower regularised incomplete gamma P(a, x)."""
    if x <= 0:
        return 0.0
    if a <= 0:
        return 1.0
    try:
        log_prefix = -x + a * math.log(x) - math.lgamma(a)
    except (ValueError, OverflowError):
        return 0.0
    if log_prefix < -700:
        return 0.0
    term = 1.0 / a
    total = term
    for k in range(1, 500):
        term *= x / (a + k)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    result = total * math.exp(log_prefix)
    return max(0.0, min(1.0, result))


def regularised_lower_inc_gamma(a: float, x: float) -> float:
    """P(a, x) = gamma(a,x)/Gamma(a) — regularised lower incomplete gamma."""
    if x <= 0:
        return 0.0
    return _gammainc_series(a, x)


def q_function(x) -> np.ndarray:
    """Gaussian Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))."""
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2.0))


def q_inv(p: float) -> float:
    """Inverse Q-function via bisection (scalar)."""
    if p <= 0:
        return np.inf
    if p >= 1:
        return -np.inf
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(q_function(mid)) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ============================================================================
#  Core AWGN capacity and dispersion
# ============================================================================

def capacity_awgn(snr_linear: float) -> float:
    """AWGN channel capacity C(gamma) = log2(1 + gamma) [bits/channel use]."""
    return np.log2(1.0 + snr_linear)


def channel_dispersion(snr_linear: float) -> float:
    """Channel dispersion V(gamma) = (1 - 1/(1+gamma)^2)(log2 e)^2 — Eq. (42)."""
    if snr_linear <= 0:
        return 0.0
    return (1.0 - 1.0 / (1.0 + snr_linear) ** 2) * (np.log2(np.e)) ** 2


# ============================================================================
#  Conditional BLER — PPV normal approximation
# ============================================================================

def conditional_bler(gamma: float, n_c: int, R_c: float) -> float:
    """
    Conditional BLER epsilon(gamma) for given SNR — Eq. (41).

    Uses the Polyanskiy-Poor-Verdu normal approximation:
      epsilon approx Q( sqrt(n_c) * (C(gamma) - R_c) / sqrt(V(gamma)) )

    Args:
        gamma: instantaneous SNR (linear)
        n_c: blocklength
        R_c: coding rate [bits/channel use]

    Returns:
        Conditional block error probability.
    """
    if gamma <= 0:
        return 1.0
    C = capacity_awgn(gamma)
    V = channel_dispersion(gamma)
    if V <= 0 or n_c <= 0:
        return 0.0 if C >= R_c else 1.0
    arg = (np.sqrt(n_c) * (C - R_c)
           + 0.5 * np.log2(n_c) / np.sqrt(n_c)) / np.sqrt(V)
    return float(q_function(arg))


def finite_blocklength_bler(snr_linear: float, N: int, K: int) -> float:
    """
    Finite blocklength BLER via PPV normal approximation.

    Args:
        snr_linear: channel SNR (linear)
        N: block length
        K: information bits

    Returns:
        Block error probability.
    """
    if snr_linear <= 0:
        return 1.0
    R = K / N
    C = capacity_awgn(snr_linear)
    V = channel_dispersion(snr_linear)
    if V <= 0:
        return 1.0 if R > C else 0.0
    if C > R:
        # PPV normal approximation with log correction — Eq. (41)
        arg = (np.sqrt(N) * (C - R)
               + 0.5 * np.log2(N) / np.sqrt(N)) / np.sqrt(V)
        bler = float(q_function(arg))
    else:
        bler = 1.0
    return np.clip(bler, 0, 1)


def finite_blocklength_bler_vec(snr_linear, N, K):
    """
    Vectorized finite blocklength BLER approximation.

    All arguments can be numpy arrays (broadcastable).
    """
    snr_linear = np.asarray(snr_linear, dtype=float)
    N = np.asarray(N, dtype=float)
    K = np.asarray(K, dtype=float)
    R = K / N
    C = np.log2(np.maximum(1 + snr_linear, 1e-30))
    V = (1 - 1 / (1 + snr_linear) ** 2) * (np.log2(np.e)) ** 2
    # PPV normal approximation with log correction — Eq. (41)
    arg = np.where(
        (V > 0) & (C > R),
        (np.sqrt(N) * (C - R) + 0.5 * np.log2(N) / np.sqrt(N))
        / np.sqrt(np.maximum(V, 1e-30)),
        0.0,
    )
    bler = np.where(
        (V > 0) & (C > R),
        0.5 * erfc(arg / np.sqrt(2)),
        np.where(snr_linear > 0, 1.0, 1.0),
    )
    return np.clip(bler, 0, 1)


def polar_sc_bler(snr_dB: float, N: int, K: int,
                  sc_loss_dB: float = 0.5) -> float:
    """Polar code BLER with SC decoding (0.5 dB loss model)."""
    effective_snr_dB = snr_dB - sc_loss_dB
    snr_linear = 10 ** (effective_snr_dB / 10)
    return finite_blocklength_bler(snr_linear, N, K)


def polar_sc_bler_vec(snr_dB, N, K, sc_loss_dB=0.5):
    """Vectorized Polar code BLER with SC decoding."""
    snr_dB = np.asarray(snr_dB, dtype=float)
    snr_linear = 10 ** ((snr_dB - sc_loss_dB) / 10)
    return finite_blocklength_bler_vec(snr_linear, N, K)


# ============================================================================
#  Closed-form and high-SNR BLER approximations (Propositions 1 & 2)
# ============================================================================

class FiniteBlocklengthBLER:
    """
    Closed-form and high-SNR BLER approximations.

    Implements Proposition 1 (closed-form with shifted gamma distribution)
    and Proposition 2 (high-SNR approximation revealing diversity order).
    """

    @staticmethod
    def capacity(gamma: float) -> float:
        """C(gamma) = log2(1+gamma)."""
        return capacity_awgn(gamma)

    @staticmethod
    def dispersion(gamma: float) -> float:
        """V(gamma) — Eq. (42)."""
        return channel_dispersion(gamma)

    @classmethod
    def conditional_bler(cls, gamma: float, n_c: int, R_c: float) -> float:
        """Conditional BLER epsilon(gamma) — Q-function approximation (Eq. 41)."""
        return conditional_bler(gamma, n_c, R_c)

    @classmethod
    def average_bler_mc(cls, gamma_bar_SU: float, P_LOS: float,
                        P_NLOS_taps: np.ndarray, K_LOS: float,
                        n_c: int, R_c: float,
                        gamma_bar_GS: float = np.inf,
                        n_samples: int = 10000,
                        rng: np.random.Generator = None) -> float:
        """
        Monte Carlo estimate of average BLER — Eq. (43).

        Averages the conditional BLER over the combined SNR distribution
        (shifted gamma from LOS + NLOS paths).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        L_nlos = len(P_NLOS_taps)
        gamma_det = gamma_bar_SU * P_LOS

        bler_sum = 0.0
        for _ in range(n_samples):
            gamma_nlos = 0.0
            for p in range(L_nlos):
                fading = rng.exponential(1.0)
                gamma_nlos += gamma_bar_SU * P_NLOS_taps[p] * fading
            gamma_combined = gamma_det + gamma_nlos
            if np.isfinite(gamma_bar_GS):
                gamma_combined = (gamma_bar_GS * gamma_combined) / \
                                 (gamma_bar_GS + gamma_combined + 1.0)
            bler_sum += cls.conditional_bler(gamma_combined, n_c, R_c)

        return bler_sum / n_samples

    @staticmethod
    def bler_closed_form(gamma_th: float, gamma_det: float,
                         gamma_bar_NLOS: float, L_i: int,
                         n_c: int = None, R_c: float = None) -> float:
        """
        Closed-form FBL BLER approximation — Proposition 1.

        From the proof (Appendix B):
          epsilon = beta * sqrt(n_c) * integral_{phi}^{delta} F_gamma(g) dg

        where F_gamma is the CDF of the shifted-gamma distribution,
        phi = psi - 1/(2*beta*sqrt(n_c)),
        delta = psi + 1/(2*beta*sqrt(n_c)).

        When n_c is None, falls back to the midpoint approximation F(psi).
        """
        if gamma_bar_NLOS <= 1e-30:
            if gamma_th > gamma_det:
                return 1.0
            return 0.0

        a = L_i - 1  # shape parameter

        def _cdf(gamma_val):
            """CDF of gamma = gamma_det + Gamma(a, gamma_bar_NLOS)."""
            if gamma_val <= gamma_det:
                return 0.0
            x = (gamma_val - gamma_det) / gamma_bar_NLOS
            if a <= 0:
                return 1.0 - math.exp(-min(x, 700))
            return float(regularised_lower_inc_gamma(a, x))

        # Fall back to midpoint (outage) approximation if no blocklength
        if n_c is None:
            return _cdf(gamma_th)

        # Piecewise linear parameters (Eq. 44-47)
        psi = gamma_th
        R_val = R_c if R_c is not None else math.log2(1.0 + gamma_th)
        beta = 1.0 / math.sqrt(2.0 * math.pi * (2.0 ** (2.0 * R_val) - 1.0))
        sqrt_nc = math.sqrt(n_c)
        half_width = 1.0 / (2.0 * beta * sqrt_nc)
        phi = psi - half_width
        delta = psi + half_width

        # Numerical integration of F_gamma over [phi, delta]
        # using 16-point Gauss-Legendre quadrature
        n_quad = 16
        # Gauss-Legendre nodes and weights on [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        # Map to [phi, delta]
        mid = 0.5 * (phi + delta)
        half = 0.5 * (delta - phi)
        integral = 0.0
        for j in range(n_quad):
            g = mid + half * nodes[j]
            integral += weights[j] * _cdf(g)
        integral *= half

        eps = beta * sqrt_nc * integral
        return max(0.0, min(1.0, eps))

    @staticmethod
    def bler_explicit(gamma_th: float, gamma_det: float,
                      gamma_bar_NLOS: float, L_i: int) -> float:
        """Explicit series BLER — Eq. (48)."""
        if gamma_th <= gamma_det:
            return 0.0
        if gamma_bar_NLOS <= 1e-30:
            return 1.0
        xi = (gamma_th - gamma_det) / gamma_bar_NLOS
        if xi > 700:
            return 1.0
        s = sum(xi ** m / math.factorial(m) for m in range(L_i - 1))
        return 1.0 - math.exp(-xi) * s

    @staticmethod
    def bler_high_snr(gamma_th: float, gamma_det: float,
                      gamma_bar_NLOS: float, L_i: int) -> float:
        """High-SNR BLER approximation — Proposition 2 / Eq. (49)."""
        if gamma_th <= gamma_det:
            return 0.0
        if gamma_bar_NLOS <= 1e-30:
            return 1.0
        xi = (gamma_th - gamma_det) / gamma_bar_NLOS
        a = L_i - 1
        if a <= 0:
            return 1.0 - math.exp(-min(xi, 700))
        if xi > 700:
            return 0.0
        return xi ** a * math.exp(-xi) / math.factorial(a)

    @staticmethod
    def bler_rayleigh(gamma_th: float, gamma_bar_br: float,
                      L_i: int) -> float:
        """Pure Rayleigh fading BLER (no LOS) — Eq. (51)."""
        if gamma_bar_br <= 0:
            return 1.0
        x = gamma_th / gamma_bar_br
        return regularised_lower_inc_gamma(L_i, x)

    @staticmethod
    def piecewise_conditional_bler(gamma: float, psi: float, beta: float,
                                   n_c: int) -> float:
        """Piecewise linear approximation of conditional BLER — Eq. (44)."""
        phi_val = psi - 1.0 / (2.0 * beta * np.sqrt(n_c))
        delta = psi + 1.0 / (2.0 * beta * np.sqrt(n_c))
        if gamma <= phi_val:
            return 1.0
        elif gamma >= delta:
            return 0.0
        else:
            return 0.5 - beta * np.sqrt(n_c) * (gamma - psi)


# ============================================================================
#  Semi-analytical BLER with diversity transforms (Section III)
# ============================================================================

def bler_no_diversity(gamma_per_sub: np.ndarray, erased_mask: np.ndarray,
                      k_c: int, N: int, K: int) -> float:
    """
    Semi-analytical BLER without diversity transform — Section III.

    Without a diversity transform, each subchannel independently carries
    its own Polar(N_sub, K_sub) codeword with N_sub = N/n_s, K_sub = K/n_s.
    The system block fails if ANY subchannel's decode fails:

      epsilon_sys = 1 - prod_j (1 - epsilon_j)

    where epsilon_j = conditional_bler(gamma_j, N_sub, R) is the
    per-subchannel FBL BLER.

    Args:
        gamma_per_sub: (n_s,) per-subchannel SNR (linear)
        erased_mask: (n_s,) bool, True = erased
        k_c: MDS code dimension (unused, kept for interface compatibility)
        N: total block length (split across subchannels)
        K: total information bits (split across subchannels)

    Returns:
        System BLER at this channel realization.
    """
    n_s = len(gamma_per_sub)
    N_sub = N // n_s
    R = K / N  # per-subchannel rate = K_sub / N_sub = (K/n_s) / (N/n_s) = K/N

    # System succeeds only if ALL subchannels decode successfully
    p_all_success = 1.0
    for j in range(n_s):
        eps_j = conditional_bler(gamma_per_sub[j], N_sub, R)
        p_all_success *= (1.0 - eps_j)

    return 1.0 - p_all_success


def _ppv_from_capacity_dispersion(C_eff: float, V_eff: float,
                                   N: int, R: float) -> float:
    """Compute PPV BLER directly from effective capacity and dispersion.

    Uses the normal approximation (Theorem 1, eq. 15):
      epsilon = Q( (sqrt(N)*(C_eff - R) + log2(N)/(2*sqrt(N))) / sqrt(V_eff) )
    """
    if V_eff <= 0 or N <= 0:
        return 0.0 if C_eff >= R else 1.0
    arg = (np.sqrt(N) * (C_eff - R)
           + 0.5 * np.log2(N) / np.sqrt(N)) / np.sqrt(V_eff)
    return float(q_function(arg))


def bler_fixed_diversity(gamma_per_sub: np.ndarray, erased_mask: np.ndarray,
                         k_c: int, N: int, K: int) -> float:
    """
    Semi-analytical BLER with fixed diversity transform (G_FIX) — Section III.

    The MDS diversity transform allows recovery from up to (n_s - k_c) erased
    subchannels. G_FIX distributes coded bits uniformly across subchannels,
    so the effective capacity and dispersion are the unweighted means of the
    per-subchannel values:
        C_eff = mean_j C(gamma_j),   V_eff = mean_j V(gamma_j)

    The PPV bound is then evaluated with these effective quantities.
    Returns 1.0 if fewer than k_c subchannels survive (unrecoverable).

    Args:
        gamma_per_sub: (n_s,) per-subchannel SNR (linear)
        erased_mask: (n_s,) bool, True = erased
        k_c: MDS code dimension (minimum surviving subchannels needed)
        N: polar code block length
        K: information bits

    Returns:
        BLER at this channel realization.
    """
    surviving = gamma_per_sub[~erased_mask]
    if len(surviving) < k_c:
        return 1.0
    R = K / N
    C_vals = np.array([capacity_awgn(g) for g in surviving])
    V_vals = np.array([channel_dispersion(g) for g in surviving])
    C_eff = float(np.mean(C_vals))
    V_eff = float(np.mean(V_vals))
    return _ppv_from_capacity_dispersion(C_eff, V_eff, N, R)


