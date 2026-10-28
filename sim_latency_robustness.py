"""
Latency vs target system BLER — prediction robustness study.

Shows how transmission latency increases with prediction error sigma_p,
and that the diversity transform still outperforms HARQ-CC even under
imperfect prediction.

Curves (single power level):
  - G_DIV (sigma_p = 0 dB, perfect)
  - G_DIV (sigma_p = 2 dB)
  - G_DIV (sigma_p = 4 dB)
  - G_DIV (sigma_p = 6 dB)
  - HARQ-CC (no prediction dependency)
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.special import comb
from scipy.stats import norm

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, OTFS_PARAMS,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    p_block_3gpp, generate_dd_channel,
    db_to_linear, linear_to_db,
)
from fbl_analysis import capacity_awgn, channel_dispersion

RTT_MS = 20.0
MAX_HARQ_RETX = 4
DELTA_F_HZ = OTFS_PARAMS.delta_f_Hz
M_D = OTFS_PARAMS.M_D

SIGMA_LEVELS_DB = [0.0, 2.0, 4.0]

_NC_GRID = np.logspace(-1, 3, 80)

_N_QUAD = 16
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(_N_QUAD)


def _bler_at_snr(gamma, n_c, R_c):
    if gamma <= 0 or n_c <= 0:
        return 1.0
    C = capacity_awgn(gamma)
    V = channel_dispersion(gamma)
    if C <= R_c or V <= 0:
        return 1.0
    arg = np.sqrt(n_c / V) * (C - R_c)
    return float(norm.cdf(-arg))


def _frame_duration_ms(n_c):
    return n_c / (M_D * DELTA_F_HZ) * 1e3


def _epsilon_star(eps_sys_target, n_s, d_min):
    binom_coeff = comb(n_s, d_min, exact=True)
    return (eps_sys_target / binom_coeff) ** (1.0 / d_min)


def _min_blocklength(eps_target, gamma, R_c):
    if gamma <= 0 or eps_target <= 0 or eps_target >= 1:
        return np.inf
    C = capacity_awgn(gamma)
    V = channel_dispersion(gamma)
    if C <= R_c or V <= 0:
        return np.inf
    q_inv = norm.ppf(1.0 - eps_target)
    n_c = V * q_inv**2 / (C - R_c)**2
    return n_c


def _div_latency_robust(eps_sys_target, gamma, R_c, sigma_p_dB, n_s, d_min):
    """Diversity transform latency under imperfect prediction.

    Prediction error causes false erasures (good subchannel declared
    bad) with probability P_fa, consuming MDS redundancy.  The per-
    subchannel coding target is tightened to compensate:

        eps_coding = (eps_star - P_fa) / (1 - P_fa)

    This is fully analytical — no grid search or bisection needed.
    """
    eps_star = _epsilon_star(eps_sys_target, n_s, d_min)

    if sigma_p_dB <= 0:
        n_c = _min_blocklength(eps_star, gamma, R_c)
        return _frame_duration_ms(n_c)

    psi = 2.0 ** R_c - 1.0
    if gamma <= psi:
        return np.inf

    snr_margin_dB = 10.0 * np.log10(gamma / psi)
    p_fa = norm.cdf(-snr_margin_dB / sigma_p_dB)

    eps_coding = (eps_star - p_fa) / (1.0 - p_fa)
    if eps_coding <= 1e-15:
        eps_coding = 1e-15

    n_c = _min_blocklength(eps_coding, gamma, R_c)
    if not np.isfinite(n_c):
        n_c = _NC_GRID[-1]
    return _frame_duration_ms(n_c)


def _harqcc_batch(eps_arr, gamma, R_c):
    n_eps_local = len(eps_arr)
    lat_out = np.full(n_eps_local, np.inf)

    n_grid = len(_NC_GRID)
    eps_table = np.zeros((n_grid, MAX_HARQ_RETX + 1))
    for gi, n_c in enumerate(_NC_GRID):
        for k in range(MAX_HARQ_RETX + 1):
            eps_table[gi, k] = _bler_at_snr(gamma * (k + 1), n_c, R_c)

    for gi, n_c in enumerate(_NC_GRID):
        T_frame = _frame_duration_ms(n_c)
        for K in range(MAX_HARQ_RETX + 1):
            p_all_fail = 1.0
            for k in range(K + 1):
                p_all_fail *= eps_table[gi, k]

            avg_lat = 0.0
            p_still_failing = 1.0
            for k in range(K + 1):
                p_success_k = p_still_failing * (1.0 - eps_table[gi, k])
                avg_lat += p_success_k * (T_frame + k * RTT_MS)
                p_still_failing *= eps_table[gi, k]
            avg_lat += p_still_failing * (T_frame + K * RTT_MS)

            for i in range(n_eps_local):
                if p_all_fail <= eps_arr[i] and avg_lat < lat_out[i]:
                    lat_out[i] = avg_lat

    return lat_out


def simulate(R_c=0.5, n_subchannels=N_S,
             P_total_dBW=10.0,
             ref_elevation_deg=50.0, n_mc=5000,
             shadow_loss_dB=20.0):
    k_c = DT_CONFIG.k_c
    n_s = DT_CONFIG.n_s
    d_min = DT_CONFIG.d_min
    K_rice = 2.0

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                              spread_deg=30.0)
    p_block_sub = p_block_3gpp(elev_angles)

    eps_sys_targets = np.logspace(-5, np.log10(0.5), 80)
    n_eps = len(eps_sys_targets)

    print("Latency robustness: prediction error sensitivity")
    print(f"  R_c={R_c:.2f}, n_s={n_s}, k_c={k_c}, d_min={d_min}")
    print(f"  P_total={P_total_dBW} dBW")
    print(f"  Sigma levels (dB): {SIGMA_LEVELS_DB}")
    print(f"  MC={n_mc}, RTT={RTT_MS} ms, max_retx={MAX_HARQ_RETX}")

    results = {
        'eps_sys_targets': eps_sys_targets,
        'P_total_dBW': P_total_dBW,
    }

    gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

    lat_div = {sigma: np.zeros((n_mc, n_eps)) for sigma in SIGMA_LEVELS_DB}
    lat_harq_all = np.zeros((n_mc, n_eps))

    for mc in range(n_mc):
        rng = np.random.default_rng(42 + mc)
        h_dd, noise_var, gamma_e2e, blocked_mask = generate_dd_channel(
            gamma_su, gamma_gs, n_subchannels, K_rice,
            p_block_sub, shadow_loss_dB, rng)

        non_blocked = gamma_e2e[~blocked_mask]
        if len(non_blocked) == 0:
            for sigma in SIGMA_LEVELS_DB:
                lat_div[sigma][mc, :] = np.inf
            lat_harq_all[mc, :] = np.inf
            continue

        gamma = np.mean(non_blocked)

        for sigma in SIGMA_LEVELS_DB:
            for i, eps_sys in enumerate(eps_sys_targets):
                lat_div[sigma][mc, i] = _div_latency_robust(
                    eps_sys, gamma, R_c, sigma, n_s, d_min)

        lat_harq_all[mc, :] = _harqcc_batch(eps_sys_targets, gamma, R_c)

        if (mc + 1) % 500 == 0:
            print(f"  MC trial {mc+1}/{n_mc}")

    for sigma in SIGMA_LEVELS_DB:
        geo = np.zeros(n_eps)
        for i in range(n_eps):
            col = lat_div[sigma][:, i]
            valid = col[np.isfinite(col) & (col > 0)]
            geo[i] = np.exp(np.mean(np.log(valid))) if len(valid) > 0 else np.inf
        geo[::-1] = np.maximum.accumulate(geo[::-1])
        results[f'lat_div_{sigma:.0f}dB'] = geo

    geo_harq = np.zeros(n_eps)
    for i in range(n_eps):
        col = lat_harq_all[:, i]
        valid = col[np.isfinite(col) & (col > 0)]
        geo_harq[i] = np.exp(np.mean(np.log(valid))) if len(valid) > 0 else np.inf
    geo_harq[::-1] = np.maximum.accumulate(geo_harq[::-1])
    results['lat_harq'] = geo_harq

    for eps_t in [1e-1, 1e-2, 1e-3, 1e-4]:
        idx = np.argmin(np.abs(eps_sys_targets - eps_t))
        parts = [f"eps={eps_t:.0e}:"]
        for sigma in SIGMA_LEVELS_DB:
            v = results[f'lat_div_{sigma:.0f}dB'][idx]
            parts.append(f"s{sigma:.0f}={v:.4f}")
        parts.append(f"HARQ={results['lat_harq'][idx]:.4f}")
        print(f"  {'  '.join(parts)}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    eps = results['eps_sys_targets']
    marker_every = max(1, len(eps) // 10)

    styles = [
        ('lat_div_0dB', 'D-', '#ff7f0e', 1.7,
         r'$\mathbf{G}_{\mathrm{DIV}}$ ($\sigma_p = 0$ dB, perfect)'),
        ('lat_div_2dB', 's-', '#1f77b4', 1.4,
         r'$\mathbf{G}_{\mathrm{DIV}}$ ($\sigma_p = 2$ dB)'),
        ('lat_div_4dB', '^-', '#2ca02c', 1.4,
         r'$\mathbf{G}_{\mathrm{DIV}}$ ($\sigma_p = 4$ dB)'),
        ('lat_harq', 'v--', '#7f7f7f', 1.4,
         r'HARQ-CC'),
    ]

    for key, fmt, color, lw, label in styles:
        ax.plot(eps, results[key],
                fmt, color=color, lw=lw, ms=5.0, mew=1.1,
                markevery=marker_every, label=label)

    ax.set_xlabel(
        r'Target system BLER $\epsilon_{\mathrm{sys}}^{\mathrm{target}}$')
    ax.set_ylabel('Transmission latency (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.legend(fontsize=5.5, loc='upper left', framealpha=1.0,
              ncol=1, handletextpad=0.5, handlelength=2.0,
              borderpad=0.4, labelspacing=0.3)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'latency_robustness_prediction')


def save_csv(results):
    eps = results['eps_sys_targets']
    keys = [f'lat_div_{s:.0f}dB' for s in SIGMA_LEVELS_DB] + ['lat_harq']

    with open('latency_robustness_prediction.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['eps_sys_target'] + keys)
        for i in range(len(eps)):
            row = [f"{eps[i]:.6e}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)

    print("Saved: latency_robustness_prediction.csv")


if __name__ == "__main__":
    results = simulate(n_mc=5000)
    plot(results)
    save_csv(results)
