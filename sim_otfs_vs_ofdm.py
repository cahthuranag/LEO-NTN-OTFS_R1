"""
OTFS vs OFDM — BLER comparison under high-Doppler LEO NTN channel.

OTFS resolves Doppler in the delay-Doppler domain via MMSE equalization,
maintaining performance across a wide range of Doppler spreads.
OFDM suffers inter-carrier interference (ICI) that scales with normalized
Doppler f_D / Delta_f, creating an SINR ceiling:
    sigma^2_ICI = (pi * f_D_norm)^2 / 3
    gamma_eff = gamma / (1 + gamma * sigma^2_ICI)
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS, OTFS_PARAMS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    per_subchannel_eta_ephemeris, p_block_3gpp,
    create_ephemeris_predictor, mc_trial_one_snr,
    generate_multipath_per_subchannel, generate_dd_channel,
    diversity_transform_transmit_receive, diversity_transform_transmit_receive_otfs,
    mmse_equalize_dd,
    db_to_linear, linear_to_db, SNRCalculator,
    G_FIX, RHO, N_OUT, M_GF, N_CW, X_ALL, S_ALL_FIX, X_MASK_0, X_MASK_1,
    conditional_bler, bler_fixed_diversity,
    DDChannel, SUBCHANNEL_MAP, map_bpsk_to_dd, extract_from_dd,
    TAP_POWERS_LIN,
)
from polar_code import PolarCode


def _ofdm_ici_variance(f_D_norm):
    return (np.pi * f_D_norm) ** 2 / 3.0


def _mc_trial_otfs_ofdm(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                         max_erasures, K_rice, epsilon_th,
                         p_block_sub, shadow_loss_dB,
                         polar_code, polar_sub, rng, f_D_norm,
                         eta_ephemeris=None):
    """
    MC trial comparing OTFS and OFDM at a given normalized Doppler.
    6 curves: OTFS/OFDM x (no-interleaver, interleaver, G_FIX).
    """
    # --- Shared channel realization ---
    gamma_e2e_base, blocked_mask = generate_multipath_per_subchannel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
        p_block_sub, shadow_loss_dB, rng)

    R = K / N
    N_sub = N // n_subchannels
    k_c = DT_CONFIG.k_c
    K_sub_actual = polar_sub.K
    N_sub_pc = polar_sub.N

    # --- OFDM: apply ICI degradation ---
    sigma2_ici = _ofdm_ici_variance(f_D_norm)
    gamma_e2e_ofdm = gamma_e2e_base / (1.0 + gamma_e2e_base * sigma2_ici)

    erased_mask_ofdm = np.array([
        conditional_bler(gamma_e2e_ofdm[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)
    ])
    n_erased_ofdm = int(np.sum(erased_mask_ofdm))

    # --- OTFS: DD-domain channel ---
    N_D = OTFS_PARAMS.N_D
    max_doppler_bin = max(1, int(round(f_D_norm * N_D)))
    max_doppler_bin = min(max_doppler_bin, N_D // 2)
    doppler_bins = [0, max_doppler_bin]

    h_dd, noise_var, gamma_e2e_otfs_dd, blocked_mask_otfs = generate_dd_channel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
        p_block_sub, shadow_loss_dB, rng, doppler_bins=doppler_bins)
    erased_mask_otfs = np.array([
        conditional_bler(gamma_e2e_otfs_dd[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)
    ])
    n_erased_otfs = int(np.sum(erased_mask_otfs))

    # --- Eta for LLR scaling ---
    eta_inst_otfs = np.zeros(n_subchannels)
    eta_inst_ofdm = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        if not erased_mask_otfs[j]:
            eta_inst_otfs[j] = 1.0 - conditional_bler(gamma_e2e_otfs_dd[j], N_sub, R)
        if not erased_mask_ofdm[j]:
            eta_inst_ofdm[j] = 1.0 - conditional_bler(gamma_e2e_ofdm[j], N_sub, R)

    info_bits = rng.integers(0, 2, size=K).astype(np.int8)

    # ====================== OTFS CURVES ======================

    # OTFS: Per-subchannel no interleaver (uses OTFS SNR, no ICI)
    err_otfs_no_intlv = False
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        llr_j = polar_sub.encode_and_transmit(sub_info, gamma_e2e_otfs_dd[j], rng)
        if polar_sub.decode_check(sub_info, llr_j):
            err_otfs_no_intlv = True
            break

    # OTFS: Per-subchannel with interleaver
    err_otfs_intlv = False
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        coded_j = polar_sub.encode(sub_info)
        perm_j = rng.permutation(N_sub_pc)
        s = 1.0 - 2.0 * coded_j[perm_j].astype(np.float64)
        sigma2 = 1.0 / (2.0 * max(gamma_e2e_otfs_dd[j], 1e-10))
        y = s + rng.normal(0, np.sqrt(sigma2), N_sub_pc)
        llr_j = (4.0 * gamma_e2e_otfs_dd[j] * y)[np.argsort(perm_j)]
        if polar_sub.decode_check(sub_info, llr_j):
            err_otfs_intlv = True
            break

    # OTFS: Fixed diversity transform
    if n_erased_otfs > max_erasures:
        err_otfs_fix = True
    else:
        coded_bits = polar_code.encode(info_bits)
        llr_otfs_fix = diversity_transform_transmit_receive_otfs(
            coded_bits, G_FIX, h_dd, noise_var,
            gamma_e2e_otfs_dd, erased_mask_otfs, eta_inst_otfs, rng,
            blocked_mask=blocked_mask_otfs)
        if llr_otfs_fix is None:
            err_otfs_fix = True
        else:
            err_otfs_fix = not np.array_equal(polar_code.decode(llr_otfs_fix), info_bits)

    # ====================== OFDM CURVES ======================

    # OFDM: Per-subchannel no interleaver (uses ICI-degraded SNR)
    err_ofdm_no_intlv = False
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        llr_j = polar_sub.encode_and_transmit(sub_info, gamma_e2e_ofdm[j], rng)
        if polar_sub.decode_check(sub_info, llr_j):
            err_ofdm_no_intlv = True
            break

    # OFDM: Per-subchannel with interleaver
    err_ofdm_intlv = False
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub_actual).astype(np.int8)
        coded_j = polar_sub.encode(sub_info)
        perm_j = rng.permutation(N_sub_pc)
        s = 1.0 - 2.0 * coded_j[perm_j].astype(np.float64)
        sigma2 = 1.0 / (2.0 * max(gamma_e2e_ofdm[j], 1e-10))
        y = s + rng.normal(0, np.sqrt(sigma2), N_sub_pc)
        llr_j = (4.0 * gamma_e2e_ofdm[j] * y)[np.argsort(perm_j)]
        if polar_sub.decode_check(sub_info, llr_j):
            err_ofdm_intlv = True
            break

    # OFDM: Fixed diversity transform
    if n_erased_ofdm > max_erasures:
        err_ofdm_fix = True
    else:
        coded_bits_ofdm = polar_code.encode(info_bits)
        llr_ofdm_fix = diversity_transform_transmit_receive(
            coded_bits_ofdm, G_FIX, gamma_e2e_ofdm, erased_mask_ofdm,
            eta_inst_ofdm, rng)
        if llr_ofdm_fix is None:
            err_ofdm_fix = True
        else:
            err_ofdm_fix = not np.array_equal(polar_code.decode(llr_ofdm_fix), info_bits)

    return {
        'otfs_no_intlv': err_otfs_no_intlv,
        'otfs_intlv': err_otfs_intlv,
        'otfs_fixed': err_otfs_fix,
        'ofdm_no_intlv': err_ofdm_no_intlv,
        'ofdm_intlv': err_ofdm_intlv,
        'ofdm_fixed': err_ofdm_fix,
    }


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=20.0, ref_elevation_deg=50.0,
             f_D_norm_range=(0.0, 0.35), n_points=10, n_mc=2000):
    rng = np.random.default_rng(2030)
    f_D_arr = np.linspace(f_D_norm_range[0], f_D_norm_range[1], n_points)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                                  spread_deg=30.0)
    gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    epsilon_th = 0.1
    shadow_loss_dB = 20.0
    p_block_sub = p_block_3gpp(elev_angles)
    R = K / N

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    # Per-subchannel Polar code
    N_sub = N // n_subchannels
    K_sub = K // n_subchannels
    N_sub_p2 = 1 << (N_sub.bit_length() - 1) if N_sub > 0 else 1
    K_sub_p2 = max(1, int(K_sub * N_sub_p2 / max(N_sub, 1)))
    K_sub_p2 = min(K_sub_p2, N_sub_p2 - 1)
    polar_sub = PolarCode(N_sub_p2, K_sub_p2, design_snr_dB=1.0, list_size=8)

    eta_ephemeris = per_subchannel_eta_ephemeris(
        elev_angles, P_total_dBW, K_rice_dB, R, p_block_per_sub=p_block_sub)

    print(f"OTFS vs OFDM: BLER vs Normalized Doppler (6 curves)")
    print(f"  P_total={P_total_dBW} dBW, elev={ref_elevation_deg}deg, MC={n_mc}")
    print(f"  Polar_sub({N_sub_p2},{K_sub_p2}), Polar_div({N},{K})")

    curve_keys = ['otfs_no_intlv', 'otfs_intlv', 'otfs_fixed',
                  'ofdm_no_intlv', 'ofdm_intlv', 'ofdm_fixed']
    results = {'f_D_norm': f_D_arr}
    for key in curve_keys:
        results[key] = np.zeros(n_points)

    for idx, f_D_norm in enumerate(f_D_arr):
        err_counts = {k: 0 for k in curve_keys}

        for _ in range(n_mc):
            res = _mc_trial_otfs_ofdm(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, epsilon_th, p_block_sub, shadow_loss_dB,
                polar_code, polar_sub, rng, f_D_norm,
                eta_ephemeris=eta_ephemeris)
            for k in curve_keys:
                err_counts[k] += int(res[k])

        for k in curve_keys:
            results[k][idx] = err_counts[k] / n_mc

        delta_f = OTFS_PARAMS.delta_f_Hz
        f_D_Hz = f_D_norm * delta_f
        print(f"  f_D={f_D_norm:.3f} ({f_D_Hz:.0f}Hz): "
              f"OTFS(NI={results['otfs_no_intlv'][idx]:.2e}, "
              f"I={results['otfs_intlv'][idx]:.2e}, "
              f"Fix={results['otfs_fixed'][idx]:.2e}) "
              f"OFDM(NI={results['ofdm_no_intlv'][idx]:.2e}, "
              f"Fix={results['ofdm_fixed'][idx]:.2e})")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    f_D = results['f_D_norm']

    # Use distinct colors: blue family for OTFS, red family for OFDM
    # Line weight hierarchy: G_DIV (thick) > intlv > no-intlv (thin)

    # OTFS curves (solid lines, filled markers)
    ax.semilogy(f_D, np.maximum(results['otfs_no_intlv'], 1e-8),
                'v-', color='#1f77b4', lw=0.8, ms=5,
                label=r'OTFS: Polar (no intlv)')
    ax.semilogy(f_D, np.maximum(results['otfs_intlv'], 1e-8),
                's-', color='#2ca02c', lw=0.8, ms=2.5,
                label=r'OTFS: Polar + intlv')
    ax.semilogy(f_D, np.maximum(results['otfs_fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.5, ms=5,
                label=r'OTFS: Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    # OFDM curves (dashed lines, open markers)
    ax.semilogy(f_D, np.maximum(results['ofdm_no_intlv'], 1e-8),
                'v--', color='#1f77b4', lw=0.8, ms=5, mfc='none', mew=0.8,
                label=r'OFDM: Polar (no intlv)')
    ax.semilogy(f_D, np.maximum(results['ofdm_intlv'], 1e-8),
                's--', color='#2ca02c', lw=0.8, ms=2.5, mfc='none', mew=0.8,
                label=r'OFDM: Polar + intlv')
    ax.semilogy(f_D, np.maximum(results['ofdm_fixed'], 1e-8),
                'D--', color='#d62728', lw=1.5, ms=5, mfc='none', mew=1.0,
                label=r'OFDM: Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Normalized Doppler $f_D / \Delta f$')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')

    # Two-column legend: OTFS left, OFDM right — grouped for clarity
    ax.legend(fontsize=5.5, loc='lower right', framealpha=1.0, ncol=2,
              columnspacing=0.8, handletextpad=0.4, handlelength=2.0,
              borderpad=0.4, labelspacing=0.35)
    ax.set_ylim(1e-2, 1.1)
    ax.set_xlim(f_D[0], f_D[-1])
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'otfs_vs_ofdm')


def save_csv(results):
    keys = ['otfs_no_intlv', 'otfs_intlv', 'otfs_fixed',
            'ofdm_no_intlv', 'ofdm_intlv', 'ofdm_fixed']
    with open('otfs_vs_ofdm.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['f_D_norm'] + keys)
        for i in range(len(results['f_D_norm'])):
            row = [f"{results['f_D_norm'][i]:.4f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: otfs_vs_ofdm.csv")


if __name__ == "__main__":
    results = simulate(n_mc=10000, n_points=12)
    plot(results)
    save_csv(results)
