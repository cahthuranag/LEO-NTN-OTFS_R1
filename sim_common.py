"""
Shared infrastructure for LEO NTN OTFS simulations.
====================================================
Constants, link budget, channel models, signal paths, MC trials,
and IEEE-style plotting helpers.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple
from itertools import combinations

from channel import (
    SNRCalculator, SPEED_OF_LIGHT, linear_to_db, db_to_linear,
    LargeScalePathLoss, AtmosphericParams, EnvironmentParams,
    OrbitalParams, LinkBudgetParams, OTFSParams, TDLProfile,
    DDChannel, EphemerisPredictor, SmallScaleFading,
)
from fbl_analysis import (
    finite_blocklength_bler, conditional_bler, capacity_awgn,
    bler_no_diversity, bler_fixed_diversity,
)
from polar_code import PolarCode
from diversity_transform import (
    DiversityTransformConfig, gf2_rank, gf2_matmul,
    generate_candidate_T,
)

# ============================================================================
#  IEEE-style figure formatting
# ============================================================================

def ieee_setup():
    """Configure matplotlib for IEEE paper figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'legend.framealpha': 1.0,
        'legend.edgecolor': '0.8',
        'figure.figsize': (3.5, 2.8),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def save_figure(fig, basename):
    """Save figure in PNG and EPS formats."""
    fig.savefig(f'{basename}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{basename}.eps', format='eps', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {basename}.png, {basename}.eps")


# ============================================================================
#  Module-level constants
# ============================================================================

# Diversity transform configuration
DT_CONFIG = DiversityTransformConfig(k_c=4, n_s=6, m=3)
G_FIX = DT_CONFIG.G_FIX
M_GF = DT_CONFIG.m
N_S = DT_CONFIG.n_s
RHO = DT_CONFIG.rho       # 12
N_OUT = DT_CONFIG.n_out    # 18

# ---- Link budget parameters ----
F_FEEDER_HZ = 20e9
G_TX_GW_DBI = 50.0
G_RX_SAT_DBI = 35.0
NF_SAT_DB = 3.0
D_FEEDER_M = 800e3

F_ACCESS_HZ = 2e9
G_TX_SAT_DBI = 30.0
G_RX_UE_DBI = 0.0
NF_UE_DB = 7.0

BW_HZ = 5e6
R_EARTH_M = 6371e3
H_ORB_M = 600e3
ALPHA_POWER = 0.5

kB = 1.38e-23
T0 = 290.0
LAMBDA_FEEDER = SPEED_OF_LIGHT / F_FEEDER_HZ
LAMBDA_ACCESS = SPEED_OF_LIGHT / F_ACCESS_HZ
FSPL_FEEDER_DB = 20.0 * np.log10(4.0 * np.pi * D_FEEDER_M / LAMBDA_FEEDER)
N0_SAT_DBW = 10.0 * np.log10(kB * T0 * BW_HZ) + NF_SAT_DB
N0_UE_DBW = 10.0 * np.log10(kB * T0 * BW_HZ) + NF_UE_DB
FEEDER_GAIN_DB = G_TX_GW_DBI + G_RX_SAT_DBI - FSPL_FEEDER_DB - N0_SAT_DBW
ACCESS_GAIN_EXCL_FSPL_DB = G_TX_SAT_DBI + G_RX_UE_DBI - N0_UE_DBW

# Atmospheric / environment models
ATM_PARAMS = AtmosphericParams(
    rho_wv=7.5, H_gas_m=6000.0, LWC=0.05, H_cf_m=1000.0,
    rain_rate_mmh=2.0, H_rain_m=3000.0, kappa=0.0101, beta=1.276,
)
ENV_PARAMS = EnvironmentParams(env_type="suburban")
PATH_LOSS_MODEL = LargeScalePathLoss(ATM_PARAMS, ENV_PARAMS)


def p_block_3gpp(elevation_angles_deg, p_block_min=0.05):
    """Per-subchannel blockage probability from 3GPP TR 38.811 P_LOS model.

    P_block(epsilon) = 1 - P_LOS(epsilon) where P_LOS is the elevation-
    dependent LOS probability for the configured environment (Eq. 16).

    A minimum floor p_block_min is enforced so that even high-elevation
    subchannels retain a non-zero blockage probability, reflecting
    residual obstructions not captured by the 3GPP P_LOS saturation.
    """
    angles = np.atleast_1d(elevation_angles_deg)
    p_block = np.array([1.0 - PATH_LOSS_MODEL.los_probability(e) for e in angles])
    p_block = np.maximum(p_block, p_block_min)
    return p_block

# Multipath TDL-D profile
L_TAPS = 2
TAP_POWERS_DB = np.array([0.0, -3.0])
TAP_POWERS_LIN = db_to_linear(TAP_POWERS_DB)
TAP_POWERS_LIN = TAP_POWERS_LIN / np.sum(TAP_POWERS_LIN)

# Pre-generate all 2^rho input vectors for MAP soft diversity demapping
N_CW = 1 << RHO
X_ALL = np.zeros((RHO, N_CW), dtype=np.int8)
for _i_cw in range(N_CW):
    for _b in range(RHO):
        X_ALL[_b, _i_cw] = (_i_cw >> _b) & 1

Z_ALL_FIX = np.mod(G_FIX.T.astype(int) @ X_ALL.astype(int), 2).astype(np.int8)
S_ALL_FIX = 1.0 - 2.0 * Z_ALL_FIX.astype(np.float64)

X_MASK_0 = [X_ALL[i, :] == 0 for i in range(RHO)]
X_MASK_1 = [X_ALL[i, :] == 1 for i in range(RHO)]


def iterative_map_demapper(corr, X_ALL_arr, rho, n_pos):
    """
    Iterative (successive-cancellation) MAP demapper for diversity transform.

    Paper Section V, Eq. (llr_compute): LLR for each input bit is computed
    conditioned on prior hard decisions, progressively halving the candidate
    set.  This mirrors the SC decoding structure of Polar codes.

    Args:
        corr:      (N_CW, n_pos) correlation matrix = (S_ALL.T @ llr_z) * 0.5
        X_ALL_arr: (rho, N_CW)   binary matrix of all input codewords
        rho:       number of input bits
        n_pos:     number of parallel coded-bit positions

    Returns:
        X_llr: (rho, n_pos) LLR for each input bit at each position
    """
    N_CW = corr.shape[0]
    X_llr = np.zeros((rho, n_pos))

    # active_mask tracks which candidates survive per position
    active_mask = np.ones((N_CW, n_pos), dtype=bool)

    for i in range(rho):
        bit_i = X_ALL_arr[i, :, np.newaxis]  # (N_CW, 1)

        mask_0 = active_mask & (bit_i == 0)   # (N_CW, n_pos)
        mask_1 = active_mask & (bit_i == 1)

        # Max-log-MAP over surviving candidates for each branch
        corr_0 = np.where(mask_0, corr, -np.inf)
        corr_1 = np.where(mask_1, corr, -np.inf)

        max_0 = np.max(corr_0, axis=0)  # (n_pos,)
        max_1 = np.max(corr_1, axis=0)

        X_llr[i, :] = max_0 - max_1

        # Hard decision: keep candidates consistent with the decision
        decide_0 = (X_llr[i, :] > 0)  # (n_pos,) bool
        active_mask = np.where(decide_0[np.newaxis, :], mask_0, mask_1)

    return X_llr


def subword_diversity_transmit_receive(info_bits, G_DIV, polar_sub,
                                       gamma_e2e_per_sub, erased_mask, eta,
                                       rng):
    """
    Paper's Kronecker encoding + max-log-MAP demapper (Section V).

    Architecture:
      1. Partition K info bits into ρ independent subwords
      2. Each subword independently Polar-encoded
      3. Diversity transform: Z = G_DIV.T @ X
      4. QPSK modulation per subchannel + AWGN
      5. Iterative: SC MAP demapper → Polar decode each subword → feedback

    This matches the paper's G_PD = G_POL ⊗ G_FIX structure and the
    iterative demapping architecture of Section V.
    """
    K_sub = polar_sub.K
    N_sub = polar_sub.N
    rho = RHO
    n_out = N_OUT

    # --- Transmit: partition → Polar encode → diversity transform ---
    X = np.zeros((rho, N_sub), dtype=np.int8)
    for i in range(rho):
        sub_info = info_bits[i * K_sub:(i + 1) * K_sub]
        X[i, :] = polar_sub.encode(sub_info)

    Z = np.mod(G_DIV.T.astype(int) @ X.astype(int), 2).astype(np.int8)

    # --- Channel: QPSK per subchannel + AWGN ---
    llr_z = np.zeros((n_out, N_sub))
    for j in range(N_S):
        gamma_j = gamma_e2e_per_sub[j]
        eta_j = eta[j]
        if erased_mask[j]:
            llr_z[j * M_GF:(j + 1) * M_GF, :] = 0.0
        else:
            sub_bits = Z[j * M_GF:(j + 1) * M_GF, :].flatten()
            qpsk_syms = qpsk_modulate(sub_bits)
            noise_std = np.sqrt(0.5 / max(gamma_j, 1e-10))
            noise = noise_std * (rng.normal(0, 1, len(qpsk_syms))
                                 + 1j * rng.normal(0, 1, len(qpsk_syms)))
            y = qpsk_syms + noise
            llr_bits = qpsk_soft_demod(y, gamma_j)
            llr_bits *= eta_j
            llr_z[j * M_GF:(j + 1) * M_GF, :] = llr_bits.reshape(M_GF, N_sub)

    if np.all(erased_mask):
        return None

    # --- Compute S_ALL for this G_DIV ---
    is_fixed = (G_DIV is G_FIX) or np.array_equal(G_DIV, G_FIX)
    if is_fixed:
        S_ALL = S_ALL_FIX
    else:
        Z_ALL = np.mod(G_DIV.T.astype(int) @ X_ALL.astype(int), 2)
        S_ALL = 1.0 - 2.0 * Z_ALL.astype(np.float64)

    corr = (S_ALL.T @ llr_z) * 0.5  # (N_CW, N_sub)

    # Paper Section V: max-log-MAP diversity demapper
    X_llr = np.zeros((rho, N_sub))
    for i in range(rho):
        X_llr[i, :] = (np.max(corr[X_MASK_0[i], :], axis=0)
                        - np.max(corr[X_MASK_1[i], :], axis=0))

    # Polar decode each subword independently
    all_info = np.zeros(rho * K_sub, dtype=np.int8)
    for i in range(rho):
        all_info[i * K_sub:(i + 1) * K_sub] = polar_sub.decode(X_llr[i, :])

    return all_info


def statistical_eta(gamma_su_arr, gamma_gs_arr, K_rice, R, n_subchannels):
    """
    Paper Eq. (llr_scaling) + Proposition 1: statistical reliability η̂_i.

    Averages the conditional BLER over the shadow fading distribution
    (Gauss-Hermite quadrature) to compute per-subchannel outage probability
    from channel statistics. This produces genuinely different per-subchannel
    weights because it depends on the statistical channel parameters
    (elevation-dependent path loss, shadow fading variance, K-factor)
    rather than the instantaneous realized SNR.
    """
    gamma_th = 2.0 ** R - 1.0  # SNR threshold for target rate
    mu_sh_dB = ENV_PARAMS.mu_sh_dB
    sigma_sh_dB = ENV_PARAMS.sigma_sh_dB

    gamma_su = np.broadcast_to(np.asarray(gamma_su_arr, dtype=float),
                               (n_subchannels,))
    gamma_gs = np.broadcast_to(np.asarray(gamma_gs_arr, dtype=float),
                               (n_subchannels,))

    # Gauss-Hermite quadrature for E_X[f(X)], X ~ N(0, sigma^2)
    n_quad = 16
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_quad)
    x_samples = np.sqrt(2.0) * sigma_sh_dB * gh_nodes
    w_samples = gh_weights / np.sqrt(np.pi)

    eta = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        # Average outage probability over shadow fading realizations
        p_outage = 0.0
        for k in range(n_quad):
            L_sh_dB = mu_sh_dB + x_samples[k]
            gamma_su_sh = gamma_su[j] * db_to_linear(-L_sh_dB)

            # MRC combined access-link SNR (all taps at this shadow level)
            gamma_su_mrc = gamma_su_sh * np.sum(TAP_POWERS_LIN)
            gamma_e2e_k = SNRCalculator.cascaded_snr(gamma_gs[j], gamma_su_mrc)

            # Outage: combined SNR below threshold
            if gamma_e2e_k < gamma_th:
                p_outage += w_samples[k]

        eta[j] = max(1.0 - p_outage, 0.0)

    return eta


# OTFS DD-domain grid parameters
OTFS_PARAMS = OTFSParams(N_D=16, M_D=64, delta_f_Hz=15e3)
M_SUB = 10


# ============================================================================
#  DD-grid helpers
# ============================================================================

def build_subchannel_map(N_D=OTFS_PARAMS.N_D, M_D=OTFS_PARAMS.M_D,
                         n_s=N_S, m_sub=M_SUB):
    sc_map = {}
    for j in range(n_s):
        positions = []
        for d in range(N_D):
            for l in range(j * m_sub, (j + 1) * m_sub):
                positions.append((d, l))
        sc_map[j] = positions
    return sc_map


def map_bpsk_to_dd(bpsk_per_sub, subchannel_map, N_D, M_D):
    x_dd = np.zeros((N_D, M_D), dtype=complex)
    for j, symbols in bpsk_per_sub.items():
        positions = subchannel_map[j]
        n_sym = len(symbols)
        for idx in range(min(n_sym, len(positions))):
            d, l = positions[idx]
            x_dd[d, l] = symbols[idx]
    return x_dd


def extract_from_dd(y_dd, subchannel_map, n_symbols_per_sub=None):
    result = {}
    for j, positions in subchannel_map.items():
        n = len(positions) if n_symbols_per_sub is None else n_symbols_per_sub.get(j, len(positions))
        syms = np.zeros(n, dtype=complex)
        for idx in range(n):
            d, l = positions[idx]
            syms[idx] = y_dd[d, l]
        result[j] = syms
    return result


SUBCHANNEL_MAP = build_subchannel_map()


# ============================================================================
#  QAM modulation / demodulation  (paper Sec. II-C: QAM on DD grid)
# ============================================================================

def qpsk_modulate(bits):
    """Gray-coded QPSK: pairs of bits -> unit-energy complex symbols.

    Mapping (b0, b1) -> ((1-2*b0) + j*(1-2*b1)) / sqrt(2).
    If len(bits) is odd, a zero-bit is appended.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) % 2 != 0:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.int8)])
    I = 1.0 - 2.0 * bits[0::2].astype(np.float64)
    Q = 1.0 - 2.0 * bits[1::2].astype(np.float64)
    return (I + 1j * Q) / np.sqrt(2.0)


def qpsk_soft_demod(y, snr):
    """QPSK soft demapper: received symbols + per-symbol SNR -> per-bit LLRs.

    For Gray-coded unit-energy QPSK:
        L(b_I) = 2*sqrt(2) * gamma * Re(y)
        L(b_Q) = 2*sqrt(2) * gamma * Im(y)

    Returns LLRs interleaved as [b0_sym0, b1_sym0, b0_sym1, b1_sym1, ...].
    snr can be a scalar or per-symbol array.
    """
    snr = np.asarray(snr, dtype=np.float64)
    scale = 2.0 * np.sqrt(2.0) * snr
    llr_I = scale * np.real(y)
    llr_Q = scale * np.imag(y)
    n_sym = len(y)
    llr = np.zeros(2 * n_sym)
    llr[0::2] = llr_I
    llr[1::2] = llr_Q
    return llr


def extract_snr_from_dd(snr_post, subchannel_map, n_symbols_per_sub=None):
    """Extract per-position post-MMSE SNR for each subchannel."""
    result = {}
    for j, positions in subchannel_map.items():
        n = (len(positions) if n_symbols_per_sub is None
             else n_symbols_per_sub.get(j, len(positions)))
        snrs = np.zeros(n)
        for idx in range(n):
            d, l = positions[idx]
            snrs[idx] = np.real(snr_post[d, l])
        result[j] = snrs
    return result


# ============================================================================
#  Link budget helpers
# ============================================================================

def slant_range_m(elevation_deg):
    eps_rad = np.radians(elevation_deg)
    sin_eps = np.sin(eps_rad)
    return (-R_EARTH_M * sin_eps
            + np.sqrt((R_EARTH_M * sin_eps) ** 2
                      + 2 * R_EARTH_M * H_ORB_M + H_ORB_M ** 2))


def access_fspl_dB(slant_range_m_val):
    return 20.0 * np.log10(4.0 * np.pi * slant_range_m_val / LAMBDA_ACCESS)


def atmospheric_loss_dB(elevation_deg):
    epsilon_rad = np.radians(elevation_deg)
    return PATH_LOSS_MODEL.atmospheric_loss_dB(F_ACCESS_HZ, epsilon_rad)


def power_to_link_snrs(P_total_dBW, access_fspl_dB_val, atm_loss_dB=0.0):
    P_total_lin = db_to_linear(P_total_dBW)
    P_SAT_lin = ALPHA_POWER * P_total_lin
    P_GW_lin = (1.0 - ALPHA_POWER) * P_total_lin
    gamma_gs_dB = linear_to_db(P_GW_lin) + FEEDER_GAIN_DB
    gamma_su_dB = (linear_to_db(P_SAT_lin) + ACCESS_GAIN_EXCL_FSPL_DB
                   - access_fspl_dB_val - atm_loss_dB)
    return db_to_linear(gamma_gs_dB), db_to_linear(gamma_su_dB)


def subchannel_elevation_angles(ref_elevation_deg, n_subchannels, spread_deg=70.0):
    """Generate per-subchannel elevation angles for multi-satellite geometry.

    Models n_subchannels satellites spread around the reference elevation,
    reflecting the heterogeneous geometry of co-visible LEO satellites.
    Spread is clipped to keep all angles in [10, 90] degrees.
    """
    offsets = np.linspace(-spread_deg / 2, spread_deg / 2, n_subchannels)
    angles = ref_elevation_deg + offsets
    return np.clip(angles, 10.0, 90.0)


def per_subchannel_link_snrs(P_total_dBW, elevation_angles_deg):
    """Compute per-subchannel (gamma_gs, gamma_su) arrays from elevation angles."""
    n = len(elevation_angles_deg)
    gamma_gs_arr = np.zeros(n)
    gamma_su_arr = np.zeros(n)
    for j in range(n):
        slant = slant_range_m(elevation_angles_deg[j])
        fspl = access_fspl_dB(slant)
        atm = atmospheric_loss_dB(elevation_angles_deg[j])
        gamma_gs_arr[j], gamma_su_arr[j] = power_to_link_snrs(
            P_total_dBW, fspl, atm)
    return gamma_gs_arr, gamma_su_arr


def per_subchannel_eta_ephemeris(elevation_angles_deg, P_total_dBW,
                                 K_rice_dB, R, p_block_per_sub=None):
    """Compute per-subchannel predicted eta from individual elevation angles.

    Implements the paper's ephemeris-driven reliability prediction (Eqs. 52-57):
      eta_i = (1 - p_block_i) * (1 - bler_pred_i)

    The predicted BLER averages the conditional BLER over the LOS shadow
    fading distribution (Paper Eq. 18):
      bler_pred_i = E_X[ conditional_bler(gamma_e2e(X), N_sub, R) ]
    where X ~ N(0, sigma_sh^2) and the predicted SNR includes the shadow
    fading realization: gamma_su_shadowed = gamma_su * 10^(-(mu_sh + X)/10).

    This ensures per-subchannel eta values reflect the actual SNR variation
    from shadow fading, producing distinct reliabilities even for
    high-elevation subchannels with similar nominal SNR.
    """
    n = len(elevation_angles_deg)
    mu_sh_dB = ENV_PARAMS.mu_sh_dB      # 3.0 dB mean shadow margin
    sigma_sh_dB = ENV_PARAMS.sigma_sh_dB  # 4.0 dB shadow fading std
    N_sub_pred = 256 // n  # approximate per-subchannel blocklength

    # Gauss-Hermite quadrature for E_X[f(X)], X ~ N(0, sigma^2)
    n_quad = 16
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_quad)
    # Transform: X = sqrt(2) * sigma * node, weight *= 1/sqrt(pi)
    x_samples = np.sqrt(2.0) * sigma_sh_dB * gh_nodes
    w_samples = gh_weights / np.sqrt(np.pi)

    # Rician fading MC samples for inner expectation
    K_rice = db_to_linear(K_rice_dB)
    n_fading_samples = 200
    fading_rng = np.random.default_rng(123)

    eta = np.zeros(n)
    for j in range(n):
        slant = slant_range_m(elevation_angles_deg[j])
        fspl = access_fspl_dB(slant)
        atm = atmospheric_loss_dB(elevation_angles_deg[j])
        gamma_gs_j, gamma_su_j = power_to_link_snrs(P_total_dBW, fspl, atm)

        # Average BLER over shadow fading (outer) and Rician fading (inner)
        bler_avg = 0.0
        for k in range(n_quad):
            L_sh_dB = mu_sh_dB + x_samples[k]
            gamma_su_sh = gamma_su_j * db_to_linear(-L_sh_dB)
            # Inner: average over Rician small-scale fading
            bler_fading = 0.0
            for _ in range(n_fading_samples):
                # Rician fading power: |h|^2 with K-factor
                los_mean = np.sqrt(K_rice / (K_rice + 1))
                nlos_std = np.sqrt(1.0 / (2.0 * (K_rice + 1)))
                h_r = los_mean + fading_rng.normal(0, nlos_std)
                h_i = fading_rng.normal(0, nlos_std)
                fading_power = h_r**2 + h_i**2
                gamma_su_faded = gamma_su_sh * fading_power
                gamma_e2e_k = SNRCalculator.cascaded_snr(
                    gamma_gs_j, gamma_su_faded)
                bler_fading += conditional_bler(
                    gamma_e2e_k, max(N_sub_pred, 1), R)
            bler_fading /= n_fading_samples
            bler_avg += w_samples[k] * bler_fading
        eta[j] = max(1.0 - bler_avg, 0.0)

    if p_block_per_sub is not None:
        p_block_arr = np.asarray(p_block_per_sub)
        eta = (1.0 - p_block_arr) * eta
    return eta


# ============================================================================
#  Channel models
# ============================================================================

def generate_multipath_per_subchannel(gamma_avg_SU, gamma_gs, n_subchannels,
                                      K_rice, p_block_sub, shadow_loss_dB, rng):
    """Generate per-subchannel SNR with fading, blockage, and shadow fading.

    Implements the paper's Eq. (18) blockage/shadowing model:
      - B=1 (LOS):   L_blk = mu_sh + X,  X ~ N(0, sigma_sh^2)  [dB]
      - B=0 (blocked): L_blk = L_deep                            [dB]
    """
    # gamma_avg_SU and gamma_gs can be scalar or per-subchannel arrays
    gamma_su_arr = np.broadcast_to(np.asarray(gamma_avg_SU, dtype=float),
                                   (n_subchannels,))
    gamma_gs_arr = np.broadcast_to(np.asarray(gamma_gs, dtype=float),
                                   (n_subchannels,))
    gamma_mrc = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        fading_power = np.zeros(L_TAPS)
        for p in range(L_TAPS):
            if p == 0:
                los_mean = np.sqrt(K_rice / (K_rice + 1))
                nlos_std = np.sqrt(1.0 / (2.0 * (K_rice + 1)))
                h_real = los_mean + rng.normal(0, nlos_std)
                h_imag = rng.normal(0, nlos_std)
                fading_power[p] = h_real**2 + h_imag**2
            else:
                fading_power[p] = rng.exponential(1.0)
        gamma_mrc[j] = gamma_su_arr[j] * np.sum(TAP_POWERS_LIN * fading_power)

    # Blockage and shadow fading — Paper Eq. (18)
    blocked_mask = rng.random(n_subchannels) < p_block_sub
    mu_sh_dB = ENV_PARAMS.mu_sh_dB      # 3.0 dB mean shadowing margin
    sigma_sh_dB = ENV_PARAMS.sigma_sh_dB  # 4.0 dB shadow fading std
    for j in range(n_subchannels):
        if blocked_mask[j]:
            # B=0: deep blockage loss
            gamma_mrc[j] *= db_to_linear(-shadow_loss_dB)
        else:
            # B=1: LOS shadow fading  L_blk = mu_sh + X(t)
            X_sh = rng.normal(0, sigma_sh_dB)
            L_sh_dB = mu_sh_dB + X_sh
            gamma_mrc[j] *= db_to_linear(-L_sh_dB)

    gamma_e2e = SNRCalculator.cascaded_snr(gamma_gs_arr, gamma_mrc)
    return gamma_e2e, blocked_mask


def generate_dd_channel(gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
                        p_block_sub, shadow_loss_dB, rng,
                        doppler_bins=None):
    """
    Generate DD-domain channel. doppler_bins controls tap Doppler indices;
    defaults to [0, 1] if not provided.
    """
    N_D = OTFS_PARAMS.N_D
    M_D = OTFS_PARAMS.M_D

    tap_gains = np.zeros(L_TAPS, dtype=complex)
    for p in range(L_TAPS):
        if p == 0:
            los_mean = np.sqrt(K_rice / (K_rice + 1))
            nlos_std = np.sqrt(1.0 / (2.0 * (K_rice + 1)))
            h_real = los_mean + rng.normal(0, nlos_std)
            h_imag = rng.normal(0, nlos_std)
            tap_gains[p] = (h_real + 1j * h_imag) * np.sqrt(TAP_POWERS_LIN[p])
        else:
            h = (rng.normal(0, 1) + 1j * rng.normal(0, 1)) / np.sqrt(2)
            tap_gains[p] = h * np.sqrt(TAP_POWERS_LIN[p])

    tap_delays_bins = np.array([0, 1])
    if doppler_bins is None:
        tap_dopplers_bins = np.array([0, 1])
    else:
        tap_dopplers_bins = np.array(doppler_bins)

    h_dd = DDChannel.build_dd_kernel(
        tap_gains, tap_delays_bins, tap_dopplers_bins, N_D, M_D)

    # gamma_avg_SU and gamma_gs can be scalar or per-subchannel arrays
    gamma_su_arr = np.broadcast_to(np.asarray(gamma_avg_SU, dtype=float),
                                   (n_subchannels,))
    gamma_gs_arr = np.broadcast_to(np.asarray(gamma_gs, dtype=float),
                                   (n_subchannels,))

    # Noise variance uses mean access-link SNR (shared DD grid)
    gamma_su_mean = float(np.mean(gamma_su_arr))
    noise_var = 1.0 / max(gamma_su_mean, 1e-10)

    H_freq = np.fft.fft2(h_dd)
    gamma_mrc = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        positions = SUBCHANNEL_MAP[j]
        h_vals = np.array([H_freq[d, l] for d, l in positions])
        gamma_mrc[j] = gamma_su_arr[j] * np.mean(np.abs(h_vals)**2)

    # Blockage and shadow fading — Paper Eq. (18)
    blocked_mask = rng.random(n_subchannels) < p_block_sub
    mu_sh_dB = ENV_PARAMS.mu_sh_dB      # 3.0 dB mean shadowing margin
    sigma_sh_dB = ENV_PARAMS.sigma_sh_dB  # 4.0 dB shadow fading std
    for j in range(n_subchannels):
        if blocked_mask[j]:
            # B=0: deep blockage loss
            gamma_mrc[j] *= db_to_linear(-shadow_loss_dB)
        else:
            # B=1: LOS shadow fading  L_blk = mu_sh + X(t)
            X_sh = rng.normal(0, sigma_sh_dB)
            L_sh_dB = mu_sh_dB + X_sh
            gamma_mrc[j] *= db_to_linear(-L_sh_dB)

    gamma_e2e = SNRCalculator.cascaded_snr(gamma_gs_arr, gamma_mrc)
    return h_dd, noise_var, gamma_e2e, blocked_mask


def apply_blockage_post_mmse(rx_per_sub, snr_per_sub, blocked_mask,
                             gamma_e2e_per_sub, qam_per_sub, n_subchannels,
                             rng):
    """Replace post-MMSE output on blocked subchannels with AWGN observation.

    For blocked subchannels, the DD-domain h_dd does not reflect the blockage
    shadow (only gamma_e2e does).  To inject the correct degradation into the
    signal path we replace the MMSE-equalised symbols with a fresh observation
    at the link-budget SNR gamma_e2e[j]:
        y_j = x_j + n_j,   n_j ~ CN(0, 1/gamma_e2e[j])
    This ensures the Polar decoder sees LLRs consistent with the actual
    blocked-subchannel SNR.
    """
    for j in range(n_subchannels):
        if blocked_mask[j]:
            gamma_j = gamma_e2e_per_sub[j]
            tx_syms = qam_per_sub[j]
            n_syms = len(tx_syms)
            noise_std = np.sqrt(0.5 / max(gamma_j, 1e-10))
            noise = noise_std * (rng.normal(0, 1, n_syms)
                                 + 1j * rng.normal(0, 1, n_syms))
            rx_per_sub[j] = tx_syms + noise
            snr_per_sub[j] = np.full(n_syms, gamma_j)


def mmse_equalize_dd(y_dd, h_dd, noise_var):
    H_freq = np.fft.fft2(h_dd)
    Y_freq = np.fft.fft2(y_dd)
    H_sq = np.abs(H_freq)**2
    X_hat_freq = np.conj(H_freq) * Y_freq / (H_sq + noise_var)
    snr_post = H_sq / noise_var
    x_hat_dd = np.fft.ifft2(X_hat_freq)
    return x_hat_dd, snr_post


# ============================================================================
#  Signal paths
# ============================================================================

def diversity_transform_transmit_receive_otfs(coded_bits, G_DIV,
                                              h_dd, noise_var,
                                              gamma_e2e_per_sub,
                                              erased_mask, eta, rng,
                                              blocked_mask=None):
    """Diversity-transform OTFS path with QPSK and post-MMSE SNR.

    Signal chain (paper-aligned):
      Polar coded bits -> diversity transform -> QPSK mod -> DD grid
      -> DDChannel (access-link noise only) -> MMSE equalize
      -> QPSK soft demod (post-MMSE SNR) -> eta-scaled LLRs -> MAP demapper
    No additional noise injection (paper Eq. 27).
    """
    N = len(coded_bits)
    N_D = OTFS_PARAMS.N_D
    M_D = OTFS_PARAMS.M_D

    n_pos = int(np.ceil(N / RHO))
    N_pad = RHO * n_pos
    coded_padded = np.zeros(N_pad, dtype=np.int8)
    coded_padded[:N] = coded_bits

    X = coded_padded.reshape(RHO, n_pos)
    Z = np.mod(G_DIV.T @ X, 2).astype(np.int8)  # (N_OUT, n_pos) binary

    # QPSK modulation per subchannel
    qam_per_sub = {}
    n_sym_per_sub = {}
    for j in range(N_S):
        sub_bits = Z[j * M_GF:(j + 1) * M_GF, :].flatten()  # M_GF*n_pos bits
        qpsk_syms = qpsk_modulate(sub_bits)
        qam_per_sub[j] = qpsk_syms
        n_sym_per_sub[j] = len(qpsk_syms)

    # DD grid -> channel -> MMSE
    x_dd = map_bpsk_to_dd(qam_per_sub, SUBCHANNEL_MAP, N_D, M_D)
    y_dd = DDChannel.apply_fast(x_dd, h_dd, noise_var, rng)
    x_hat_dd, snr_post = mmse_equalize_dd(y_dd, h_dd, noise_var)
    rx_per_sub = extract_from_dd(x_hat_dd, SUBCHANNEL_MAP, n_sym_per_sub)
    snr_per_sub = extract_snr_from_dd(snr_post, SUBCHANNEL_MAP, n_sym_per_sub)

    # Apply blockage: replace MMSE output on blocked subchannels
    if blocked_mask is not None:
        apply_blockage_post_mmse(rx_per_sub, snr_per_sub, blocked_mask,
                                 gamma_e2e_per_sub, qam_per_sub, N_S, rng)

    if np.all(erased_mask):
        return None

    # QPSK soft demod + eta-scaling -> llr_z for MAP demapper
    llr_z = np.zeros((N_OUT, n_pos))
    for j in range(N_S):
        eta_j = eta[j]
        if erased_mask[j]:
            llr_z[j * M_GF:(j + 1) * M_GF, :] = 0.0
        else:
            # QPSK soft demod using post-MMSE SNR (no extra noise)
            llr_bits = qpsk_soft_demod(rx_per_sub[j], snr_per_sub[j])
            llr_bits *= eta_j
            n_needed = M_GF * n_pos
            llr_z[j * M_GF:(j + 1) * M_GF, :] = llr_bits[:n_needed].reshape(M_GF, n_pos)

    # Paper Section V: max-log-MAP diversity demapper
    # L(x_i) = max_{x:x_i=0} corr(x) - max_{x:x_i=1} corr(x)
    is_fixed = (G_DIV is G_FIX) or np.array_equal(G_DIV, G_FIX)
    if is_fixed:
        S_ALL = S_ALL_FIX
    else:
        Z_ALL = np.mod(G_DIV.T.astype(int) @ X_ALL.astype(int), 2)
        S_ALL = 1.0 - 2.0 * Z_ALL.astype(np.float64)

    corr = (S_ALL.T @ llr_z) * 0.5
    X_llr = np.zeros((RHO, n_pos))
    for i in range(RHO):
        X_llr[i, :] = (np.max(corr[X_MASK_0[i], :], axis=0)
                        - np.max(corr[X_MASK_1[i], :], axis=0))

    llr_vector = X_llr.flatten()[:N]
    return llr_vector


def standard_polar_otfs(info_bits, polar_code, h_dd, noise_var,
                        gamma_e2e_per_sub, n_subchannels, perm, rng,
                        blocked_mask=None):
    """Standard Polar over OTFS DD channel with QPSK modulation.

    Uses post-MMSE SNR for LLR computation — NO additional noise injection
    (paper Eq. 27: effective noise is access-link noise only under dominant
    feeder-link regime).
    """
    N = polar_code.N
    N_D = OTFS_PARAMS.N_D
    M_D = OTFS_PARAMS.M_D

    coded = polar_code.encode(info_bits)

    # Bit-level permutation (interleaver) before modulation
    if perm is not None:
        coded_perm = coded[perm]
    else:
        coded_perm = coded

    # QPSK modulation: N bits -> N/2 complex symbols
    qpsk_syms = qpsk_modulate(coded_perm)
    n_total_syms = len(qpsk_syms)

    # Distribute QPSK symbols across subchannels
    syms_per_sub = n_total_syms // n_subchannels
    qam_per_sub = {}
    n_sym_per_sub = {}
    for s in range(n_subchannels):
        start_s = s * syms_per_sub
        end_s = min((s + 1) * syms_per_sub, n_total_syms)
        qam_per_sub[s] = qpsk_syms[start_s:end_s]
        n_sym_per_sub[s] = end_s - start_s
    if syms_per_sub * n_subchannels < n_total_syms:
        extra = qpsk_syms[syms_per_sub * n_subchannels:]
        qam_per_sub[n_subchannels - 1] = np.concatenate(
            [qam_per_sub[n_subchannels - 1], extra])
        n_sym_per_sub[n_subchannels - 1] += len(extra)

    # DD grid -> channel -> MMSE equalize
    x_dd = map_bpsk_to_dd(qam_per_sub, SUBCHANNEL_MAP, N_D, M_D)
    y_dd = DDChannel.apply_fast(x_dd, h_dd, noise_var, rng)
    x_hat_dd, snr_post = mmse_equalize_dd(y_dd, h_dd, noise_var)

    # Extract equalized symbols and per-position post-MMSE SNR
    rx_per_sub = extract_from_dd(x_hat_dd, SUBCHANNEL_MAP, n_sym_per_sub)
    snr_per_sub = extract_snr_from_dd(snr_post, SUBCHANNEL_MAP, n_sym_per_sub)

    # Apply blockage: replace MMSE output on blocked subchannels
    if blocked_mask is not None:
        apply_blockage_post_mmse(rx_per_sub, snr_per_sub, blocked_mask,
                                 gamma_e2e_per_sub, qam_per_sub,
                                 n_subchannels, rng)

    # QPSK soft demod using post-MMSE SNR (single noise source)
    llr_all = np.zeros(N)
    bit_offset = 0
    for s in range(n_subchannels):
        n_syms = n_sym_per_sub[s]
        llr_bits = qpsk_soft_demod(rx_per_sub[s][:n_syms],
                                   snr_per_sub[s][:n_syms])
        n_bits = min(2 * n_syms, N - bit_offset)
        llr_all[bit_offset:bit_offset + n_bits] = llr_bits[:n_bits]
        bit_offset += n_bits

    # Inverse permutation
    if perm is not None:
        inv_perm = np.argsort(perm)
        llr_all = llr_all[inv_perm]

    return llr_all


def diversity_transform_transmit_receive(coded_bits, G_DIV, gamma_e2e_per_sub,
                                         erased_mask, eta, rng):
    """Diversity-transform AWGN path with QPSK (used for OFDM comparison)."""
    N = len(coded_bits)
    n_pos = int(np.ceil(N / RHO))
    N_pad = RHO * n_pos
    coded_padded = np.zeros(N_pad, dtype=np.int8)
    coded_padded[:N] = coded_bits

    X = coded_padded.reshape(RHO, n_pos)
    Z = np.mod(G_DIV.T @ X, 2).astype(np.int8)

    llr_z = np.zeros((N_OUT, n_pos))
    for j in range(N_S):
        gamma_j = gamma_e2e_per_sub[j]
        eta_j = eta[j]
        if erased_mask[j]:
            llr_z[j * M_GF:(j + 1) * M_GF, :] = 0.0
        else:
            # QPSK: flatten bits, modulate, add complex AWGN, soft demod
            sub_bits = Z[j * M_GF:(j + 1) * M_GF, :].flatten()
            qpsk_syms = qpsk_modulate(sub_bits)
            # Complex AWGN at per-symbol SNR gamma_j
            noise_std = np.sqrt(0.5 / max(gamma_j, 1e-10))
            noise = noise_std * (rng.normal(0, 1, len(qpsk_syms))
                                 + 1j * rng.normal(0, 1, len(qpsk_syms)))
            y = qpsk_syms + noise
            llr_bits = qpsk_soft_demod(y, gamma_j)
            llr_bits *= eta_j
            llr_z[j * M_GF:(j + 1) * M_GF, :] = llr_bits.reshape(M_GF, n_pos)

    if np.all(erased_mask):
        return None

    # Paper Section V: max-log-MAP diversity demapper
    # L(x_i) = max_{x:x_i=0} corr(x) - max_{x:x_i=1} corr(x)
    is_fixed = (G_DIV is G_FIX) or np.array_equal(G_DIV, G_FIX)
    if is_fixed:
        S_ALL = S_ALL_FIX
    else:
        Z_ALL = np.mod(G_DIV.T.astype(int) @ X_ALL.astype(int), 2)
        S_ALL = 1.0 - 2.0 * Z_ALL.astype(np.float64)

    corr = (S_ALL.T @ llr_z) * 0.5
    X_llr = np.zeros((RHO, n_pos))
    for i in range(RHO):
        X_llr[i, :] = (np.max(corr[X_MASK_0[i], :], axis=0)
                        - np.max(corr[X_MASK_1[i], :], axis=0))

    llr_vector = X_llr.flatten()[:N]
    return llr_vector


# ============================================================================
#  EphemerisPredictor factory
# ============================================================================

def create_ephemeris_predictor(elevation_deg, P_total_dBW, K_rice_dB=3.0):
    orbital = OrbitalParams(h_orb=H_ORB_M)
    feeder = LinkBudgetParams(
        P_tx_dBm=10.0 * np.log10(ALPHA_POWER * db_to_linear(P_total_dBW)) + 30.0,
        G_tx_dBi=G_TX_GW_DBI, G_rx_dBi=G_RX_SAT_DBI,
        f_c_Hz=F_FEEDER_HZ, noise_figure_dB=NF_SAT_DB,
        bandwidth_Hz=BW_HZ, T_sys_K=T0,
    )
    access = LinkBudgetParams(
        P_tx_dBm=10.0 * np.log10((1.0 - ALPHA_POWER) * db_to_linear(P_total_dBW)) + 30.0,
        G_tx_dBi=G_TX_SAT_DBI, G_rx_dBi=G_RX_UE_DBI,
        f_c_Hz=F_ACCESS_HZ, noise_figure_dB=NF_UE_DB,
        bandwidth_Hz=BW_HZ, T_sys_K=T0,
    )
    tdl = TDLProfile(
        num_taps=L_TAPS,
        relative_powers_dB=TAP_POWERS_DB.tolist(),
        delays_ns=[0, 100],
        K_factor_dB=K_rice_dB,
    )
    return EphemerisPredictor(
        orbital, feeder, access, ATM_PARAMS, ENV_PARAMS, OTFS_PARAMS, tdl,
    )


# ============================================================================
#  Monte Carlo trial (5 curves)
# ============================================================================

def mc_trial_one_snr(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                     max_erasures, K_rice, epsilon_th, p_block_sub,
                     shadow_loss_dB, polar_code, rng,
                     use_otfs=False, eta_ephemeris=None,
                     ici_variance=0.0, polar_sub=None):
    """
    Run one MC trial for all 5 curves.

    All schemes operate over n_subchannels parallel subchannels:

    Curves:
      1. PPV bound (no diversity) — per-subchannel Polar(N_sub, K_sub), fail if any fails
      2. PPV bound (fixed diversity, G_FIX) — MDS recovery across subchannels
      3. MC: Polar per-subchannel (no interleaver) — independent Polar(N_sub, K_sub) per subchannel
      4. MC: Polar per-subchannel + interleaver — same with bit interleaving within each subchannel
      5. MC: Polar + G_FIX (fixed diversity transform) — diversity transform across subchannels

    epsilon_th: BLER threshold for erasure declaration (Paper Eq. 18/Section III).
    polar_sub: PolarCode(N_sub, K_sub) for per-subchannel baselines.
               If None, created internally from N//n_subchannels, K//n_subchannels.
    """
    if use_otfs:
        h_dd, noise_var, gamma_e2e, blocked_mask = generate_dd_channel(
            gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
            p_block_sub, shadow_loss_dB, rng)
    else:
        gamma_e2e, blocked_mask = generate_multipath_per_subchannel(
            gamma_avg_SU, gamma_gs, n_subchannels, K_rice, p_block_sub,
            shadow_loss_dB, rng)
        # Apply ICI degradation for OFDM in high-Doppler
        if ici_variance > 0:
            gamma_e2e = gamma_e2e / (1.0 + gamma_e2e * ici_variance)

    # Paper Section III: erasure when per-subchannel BLER > epsilon_th
    R = K / N
    N_sub = N // n_subchannels
    K_sub = K // n_subchannels
    k_c = DT_CONFIG.k_c
    erased_mask = np.array([
        conditional_bler(gamma_e2e[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)
    ])
    n_erased = int(np.sum(erased_mask))

    # Paper Eq. (llr_scaling): η̂_i = 1 - ε_i for LLR scaling
    eta_inst = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        if erased_mask[j]:
            eta_inst[j] = 0.0
        else:
            eps_j = conditional_bler(gamma_e2e[j], N_sub, R)
            eta_inst[j] = 1.0 - eps_j

    # Semi-analytical PPV bounds
    ppv_no_div = bler_no_diversity(gamma_e2e, erased_mask, k_c, N, K)
    ppv_fix_div = bler_fixed_diversity(gamma_e2e, erased_mask, k_c, N, K)

    # Per-subchannel info bits (for baselines without diversity)
    info_bits_all = rng.integers(0, 2, size=K).astype(np.int8)

    # ---- Baselines: independent Polar(N_sub, K_sub) per subchannel ----
    # Create per-subchannel polar code if not provided
    # N_sub must be power of 2 for Polar codes
    if polar_sub is None:
        # Find largest power-of-2 <= N_sub
        N_sub_p2 = 1 << (N_sub.bit_length() - 1) if N_sub > 0 else 1
        K_sub_p2 = max(1, int(K_sub * N_sub_p2 / max(N_sub, 1)))
        K_sub_p2 = min(K_sub_p2, N_sub_p2 - 1)  # ensure K < N
        polar_sub = PolarCode(N_sub_p2, K_sub_p2, design_snr_dB=1.0, list_size=8)
    K_sub_actual = polar_sub.K

    # Curve 3: No interleaver — independent per-subchannel Polar, system fails if any fails
    N_sub_pc = polar_sub.N
    err_1 = False
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        llr_j = polar_sub.encode_and_transmit(sub_info, gamma_e2e[j], rng)
        if polar_sub.decode_check(sub_info, llr_j):
            err_1 = True
            break

    # Curve 4: Interleaver — per-subchannel Polar with bit interleaving within subchannel
    err_2 = False
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        coded_j = polar_sub.encode(sub_info)
        perm_j = rng.permutation(N_sub_pc)
        coded_perm = coded_j[perm_j]
        # Transmit through subchannel j
        s = 1.0 - 2.0 * coded_perm.astype(np.float64)
        sigma2 = 1.0 / (2.0 * max(gamma_e2e[j], 1e-10))
        noise = rng.normal(0, np.sqrt(sigma2), N_sub_pc)
        y = s + noise
        llr_perm = 4.0 * gamma_e2e[j] * y
        # Inverse permutation
        inv_perm = np.argsort(perm_j)
        llr_j = llr_perm[inv_perm]
        if polar_sub.decode_check(sub_info, llr_j):
            err_2 = True
            break

    # Curve 5: Diversity transform — Polar(N, K) + G_FIX across subchannels
    if n_erased > max_erasures:
        err_3 = True
    else:
        coded_bits = polar_code.encode(info_bits_all)
        if use_otfs:
            llr_3 = diversity_transform_transmit_receive_otfs(
                coded_bits, G_FIX, h_dd, noise_var,
                gamma_e2e, erased_mask, eta_inst, rng,
                blocked_mask=blocked_mask)
        else:
            llr_3 = diversity_transform_transmit_receive(
                coded_bits, G_FIX, gamma_e2e, erased_mask, eta_inst, rng)
        if llr_3 is None:
            err_3 = True
        else:
            decoded_3 = polar_code.decode(llr_3)
            err_3 = not np.array_equal(decoded_3, info_bits_all)

    return {
        'ppv_no_div': ppv_no_div,
        'ppv_fix_div': ppv_fix_div,
        'err_1': err_1,
        'err_2': err_2,
        'err_3': err_3,
    }


# ============================================================================
#  Variable n_s MC trial and signal path (for n_subchannels sweep)
# ============================================================================

def diversity_transform_transmit_receive_variable(
        coded_bits, G_DIV, gamma_e2e_per_sub, erased_mask, eta,
        rng, n_s, m, rho, n_out,
        X_ALL_local, S_ALL, X_MASK_0_local, X_MASK_1_local):
    N = len(coded_bits)
    n_pos = int(np.ceil(N / rho))
    padded = np.zeros(rho * n_pos, dtype=np.int8)
    padded[:N] = coded_bits

    X = padded.reshape(rho, n_pos)
    Z = np.mod(G_DIV.T.astype(int) @ X.astype(int), 2).astype(np.int8)

    llr_z = np.zeros((n_out, n_pos))
    for j in range(n_s):
        eta_j = eta[j]
        gamma_j = gamma_e2e_per_sub[j]
        if erased_mask[j]:
            for b in range(m):
                llr_z[j * m + b, :] = 0.0
        else:
            for b in range(m):
                row_idx = j * m + b
                s_row = 1.0 - 2.0 * Z[row_idx, :].astype(np.float64)
                sigma = np.sqrt(0.5 / max(gamma_j, 1e-10))
                y = s_row + rng.normal(0, sigma, n_pos)
                llr_z[row_idx, :] = eta_j * 4.0 * gamma_j * y

    if np.all(erased_mask):
        return None

    # Paper Section V: max-log-MAP diversity demapper
    corr = (S_ALL.T @ llr_z) * 0.5
    X_llr = np.zeros((rho, n_pos))
    for i in range(rho):
        X_llr[i, :] = (np.max(corr[X_MASK_0_local[i], :], axis=0)
                        - np.max(corr[X_MASK_1_local[i], :], axis=0))

    llr_coded = X_llr.flatten()[:N]
    return llr_coded


def mc_trial_variable_ns(gamma_avg_SU, gamma_gs, N, K, dt_config,
                         G_FIX_local, X_ALL_local, S_ALL_FIX_local,
                         X_MASK_0_local, X_MASK_1_local,
                         subchannel_map, n_sym_per_sub,
                         K_rice, epsilon_th, p_block_sub,
                         shadow_loss_dB, polar_code, rng,
                         eta_ephemeris=None, polar_sub=None):
    """
    MC trial with variable n_s — 5 curves.

    Curves:
      1. PPV bound (no diversity) — per-subchannel, system fails if any fails
      2. PPV bound (fixed diversity, G_FIX)
      3. MC: Polar per-subchannel (no interleaver)
      4. MC: Polar per-subchannel + interleaver
      5. MC: Polar + G_FIX (fixed diversity transform)
    """
    n_s = dt_config.n_s
    m = dt_config.m
    k_c = dt_config.k_c
    rho = dt_config.rho
    n_out = dt_config.n_out
    max_erasures = dt_config.max_erasures

    gamma_e2e, blocked_mask = generate_multipath_per_subchannel(
        gamma_avg_SU, gamma_gs, n_s, K_rice, p_block_sub, shadow_loss_dB, rng)

    R = K / N
    N_sub = N // n_s
    K_sub = K // n_s

    erased_mask = np.array([
        conditional_bler(gamma_e2e[j], N_sub, R) > epsilon_th
        for j in range(n_s)
    ])
    n_erased = int(np.sum(erased_mask))

    eta_inst = np.zeros(n_s)
    for j in range(n_s):
        if erased_mask[j]:
            eta_inst[j] = 0.0
        else:
            eps_j = conditional_bler(gamma_e2e[j], N_sub, R)
            eta_inst[j] = 1.0 - eps_j

    # PPV bounds
    ppv_no_div = bler_no_diversity(gamma_e2e, erased_mask, k_c, N, K)
    ppv_fix = bler_fixed_diversity(gamma_e2e, erased_mask, k_c, N, K)

    # Per-subchannel Polar code for baselines
    if polar_sub is None:
        N_sub_p2 = 1 << (N_sub.bit_length() - 1) if N_sub > 0 else 1
        K_sub_p2 = max(1, int(K_sub * N_sub_p2 / max(N_sub, 1)))
        K_sub_p2 = min(K_sub_p2, N_sub_p2 - 1)
        polar_sub = PolarCode(N_sub_p2, K_sub_p2, design_snr_dB=1.0, list_size=8)
    K_sub_actual = polar_sub.K
    N_sub_pc = polar_sub.N

    # Curve 3: No interleaver — independent per-subchannel Polar
    err_no_intlv = False
    for j in range(n_s):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        llr_j = polar_sub.encode_and_transmit(sub_info, gamma_e2e[j], rng)
        if polar_sub.decode_check(sub_info, llr_j):
            err_no_intlv = True
            break

    # Curve 4: Interleaver — per-subchannel Polar with bit interleaving
    err_intlv = False
    for j in range(n_s):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        coded_j = polar_sub.encode(sub_info)
        perm_j = rng.permutation(N_sub_pc)
        coded_perm = coded_j[perm_j]
        s = 1.0 - 2.0 * coded_perm.astype(np.float64)
        sigma2 = 1.0 / (2.0 * max(gamma_e2e[j], 1e-10))
        noise = rng.normal(0, np.sqrt(sigma2), N_sub_pc)
        y = s + noise
        llr_perm = 4.0 * gamma_e2e[j] * y
        inv_perm = np.argsort(perm_j)
        llr_j = llr_perm[inv_perm]
        if polar_sub.decode_check(sub_info, llr_j):
            err_intlv = True
            break

    # Curve 5: Fixed diversity transform
    info_bits = rng.integers(0, 2, size=K).astype(np.int8)
    if n_erased > max_erasures:
        err_fix = True
    else:
        coded_bits = polar_code.encode(info_bits)
        llr_fix = diversity_transform_transmit_receive_variable(
            coded_bits, G_FIX_local, gamma_e2e, erased_mask, eta_inst,
            rng, n_s, m, rho, n_out,
            X_ALL_local, S_ALL_FIX_local, X_MASK_0_local, X_MASK_1_local)
        if llr_fix is None:
            err_fix = True
        else:
            decoded = polar_code.decode(llr_fix)
            err_fix = not np.array_equal(decoded, info_bits)

    return {
        'ppv_no_div': ppv_no_div,
        'ppv_fix': ppv_fix,
        'err_no_intlv': err_no_intlv,
        'err_intlv': err_intlv,
        'err_fix': err_fix,
    }
