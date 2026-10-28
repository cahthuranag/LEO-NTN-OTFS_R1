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


def _build_dd_kernel_fractional(tap_gains, tap_delays_bins, tap_dopplers,
                                N_D, M_D):
    """DD kernel supporting fractional Doppler indices.

    For integer Doppler, energy is concentrated in a single DD bin.
    For fractional Doppler, energy leaks across all Doppler bins
    following a Dirichlet kernel (aliased sinc), modelling the
    inter-Doppler interference that degrades MMSE equalization.
    """
    h_dd = np.zeros((N_D, M_D), dtype=complex)
    n = np.arange(N_D)
    for p in range(len(tap_gains)):
        l_p = int(tap_delays_bins[p]) % M_D
        nu_p = tap_dopplers[p]
        h_time_p = tap_gains[p] * np.exp(1j * 2 * np.pi * nu_p * n / N_D)
        h_dd[:, l_p] += np.fft.fft(h_time_p) / N_D
    return h_dd


def _mc_trial_otfs_ofdm(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                         max_erasures, K_rice, epsilon_th,
                         p_block_sub, shadow_loss_dB,
                         polar_code, polar_sub, rng, f_D_norm,
                         eta_ephemeris=None):
    """
    MC trial comparing OTFS and OFDM at a given normalized Doppler.
    8 curves: 6 original + 2 new (fractional Doppler, frac+imperfect CSI).
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

    # OTFS: Fixed diversity transform (integer Doppler, perfect CSI)
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

    # =========== OTFS: Fractional Doppler + Imperfect CSI ===========
    # Analytical SINR degradation model:
    #   OTFS fractional ICI ~ (pi * alpha)^2 / (3 * N_D)
    #   Imperfect CSI MSE ~ sigma_csi^2
    alpha_frac = 0.5
    sigma_frac_sq = (np.pi * alpha_frac) ** 2 / (3.0 * N_D)
    sigma_csi_sq = 0.01

    gamma_e2e_frac = gamma_e2e_otfs_dd / (1.0 + gamma_e2e_otfs_dd * sigma_frac_sq)
    gamma_e2e_frac_icsi = gamma_e2e_otfs_dd / (
        1.0 + gamma_e2e_otfs_dd * (sigma_frac_sq + sigma_csi_sq))

    erased_frac = np.array([
        conditional_bler(gamma_e2e_frac[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)])
    erased_frac_icsi = np.array([
        conditional_bler(gamma_e2e_frac_icsi[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)])

    eta_frac = np.array([0.0 if erased_frac[j] else
                         1.0 - conditional_bler(gamma_e2e_frac[j], N_sub, R)
                         for j in range(n_subchannels)])
    eta_frac_icsi = np.array([0.0 if erased_frac_icsi[j] else
                              1.0 - conditional_bler(gamma_e2e_frac_icsi[j], N_sub, R)
                              for j in range(n_subchannels)])

    # Fractional Doppler, perfect CSI
    if int(np.sum(erased_frac)) > max_erasures:
        err_otfs_frac = True
    else:
        coded_bits_f = polar_code.encode(info_bits)
        llr_frac = diversity_transform_transmit_receive(
            coded_bits_f, G_FIX, gamma_e2e_frac, erased_frac, eta_frac, rng)
        if llr_frac is None:
            err_otfs_frac = True
        else:
            err_otfs_frac = not np.array_equal(polar_code.decode(llr_frac), info_bits)

    # Fractional Doppler + imperfect CSI
    if int(np.sum(erased_frac_icsi)) > max_erasures:
        err_otfs_frac_icsi = True
    else:
        llr_icsi = diversity_transform_transmit_receive(
            coded_bits_f, G_FIX, gamma_e2e_frac_icsi, erased_frac_icsi,
            eta_frac_icsi, rng)
        if llr_icsi is None:
            err_otfs_frac_icsi = True
        else:
            err_otfs_frac_icsi = not np.array_equal(polar_code.decode(llr_icsi), info_bits)

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
        'otfs_frac': err_otfs_frac,
        'otfs_frac_icsi': err_otfs_frac_icsi,
        'ofdm_no_intlv': err_ofdm_no_intlv,
        'ofdm_intlv': err_ofdm_intlv,
        'ofdm_fixed': err_ofdm_fix,
    }


def _worker_doppler(args):
    """Worker for one Doppler point."""
    (idx, f_D_norm, n_mc, seed, gamma_su, gamma_gs, N, K,
     n_subchannels, max_erasures, K_rice, epsilon_th,
     p_block_sub, shadow_loss_dB, eta_ephemeris) = args

    rng = np.random.default_rng(seed)
    R = K / N

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)
    N_sub = N // n_subchannels
    K_sub = K // n_subchannels
    N_sub_p2 = 1 << (N_sub.bit_length() - 1) if N_sub > 0 else 1
    K_sub_p2 = max(1, int(K_sub * N_sub_p2 / max(N_sub, 1)))
    K_sub_p2 = min(K_sub_p2, N_sub_p2 - 1)
    polar_sub = PolarCode(N_sub_p2, K_sub_p2, design_snr_dB=1.0, list_size=8)

    curve_keys = ['otfs_no_intlv', 'otfs_intlv', 'otfs_fixed',
                  'otfs_frac', 'otfs_frac_icsi',
                  'ofdm_no_intlv', 'ofdm_intlv', 'ofdm_fixed']
    err_counts = {k: 0 for k in curve_keys}

    for _ in range(n_mc):
        res = _mc_trial_otfs_ofdm(
            gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
            K_rice, epsilon_th, p_block_sub, shadow_loss_dB,
            polar_code, polar_sub, rng, f_D_norm,
            eta_ephemeris=eta_ephemeris)
        for k in curve_keys:
            err_counts[k] += int(res[k])

    return idx, {k: v / n_mc for k, v in err_counts.items()}


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=20.0, ref_elevation_deg=50.0,
             f_D_norm_range=(0.0, 0.35), n_points=10, n_mc=2000,
             n_workers=None):
    import multiprocessing as mp
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_points)

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

    eta_ephemeris = per_subchannel_eta_ephemeris(
        elev_angles, P_total_dBW, K_rice_dB, R, p_block_per_sub=p_block_sub)

    print(f"OTFS vs OFDM: BLER vs Normalized Doppler (8 curves, {n_workers} workers)")
    print(f"  P_total={P_total_dBW} dBW, elev={ref_elevation_deg}deg, MC={n_mc}")

    curve_keys = ['otfs_no_intlv', 'otfs_intlv', 'otfs_fixed',
                  'otfs_frac', 'otfs_frac_icsi',
                  'ofdm_no_intlv', 'ofdm_intlv', 'ofdm_fixed']
    results = {'f_D_norm': f_D_arr}
    for key in curve_keys:
        results[key] = np.zeros(n_points)

    base_seed = 2030
    worker_args = []
    for idx, f_D_norm in enumerate(f_D_arr):
        worker_args.append((
            idx, float(f_D_norm), n_mc, base_seed + idx * 1000,
            gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
            K_rice, epsilon_th, p_block_sub, shadow_loss_dB, eta_ephemeris
        ))

    with mp.Pool(n_workers) as pool:
        for idx, bler_dict in pool.imap_unordered(_worker_doppler, worker_args):
            f_D_norm = f_D_arr[idx]
            for k, v in bler_dict.items():
                results[k][idx] = v
            delta_f = OTFS_PARAMS.delta_f_Hz
            f_D_Hz = f_D_norm * delta_f
            print(f"  f_D={f_D_norm:.3f} ({f_D_Hz:.0f}Hz): "
                  f"OTFS(Fix={results['otfs_fixed'][idx]:.2e}, "
                  f"Frac={results['otfs_frac'][idx]:.2e}, "
                  f"Frac+iCSI={results['otfs_frac_icsi'][idx]:.2e}) "
                  f"OFDM(Fix={results['ofdm_fixed'][idx]:.2e})")

    return results


def plot(results):
    from scipy.signal import savgol_filter
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    f_D = results['f_D_norm']

    def smooth(y, window=7, poly=2):
        return savgol_filter(np.maximum(y, 1e-8), window, poly)

    # Stagger markers so overlapping curves remain distinguishable
    # OTFS curves (solid lines, filled markers)
    ax.semilogy(f_D, np.maximum(results['otfs_no_intlv'], 1e-8),
                'o-', color='#1f77b4', lw=1.1, ms=5, markevery=(0, 3),
                label=r'OTFS (no div.)')
    ax.semilogy(f_D, np.maximum(results['otfs_intlv'], 1e-8),
                's-', color='#2ca02c', lw=1.1, ms=5, markevery=(1, 3),
                label=r'OTFS + intlv')
    ax.semilogy(f_D, smooth(results['otfs_fixed']),
                'D-', color='#ff7f0e', lw=1.7, ms=5.5, markevery=(0, 3),
                label=r'OTFS + $\mathbf{G}_{\mathrm{DIV}}$')
    ax.semilogy(f_D, smooth(results['otfs_frac']),
                '^-', color='#9467bd', lw=1.4, ms=5.5, markevery=(1, 3),
                label=r'OTFS + $\mathbf{G}_{\mathrm{DIV}}$ (frac.)')
    ax.semilogy(f_D, smooth(results['otfs_frac_icsi']),
                'v-', color='#8c564b', lw=1.4, ms=5.5, markevery=(2, 3),
                label=r'OTFS + $\mathbf{G}_{\mathrm{DIV}}$ (frac. + imp. CSI)')

    # OFDM curves (dashed lines, open markers — distinct colours)
    ax.semilogy(f_D, np.maximum(results['ofdm_no_intlv'], 1e-8),
                'o--', color='#d62728', lw=1.1, ms=5, mfc='none', mew=1.2, markevery=(2, 3),
                label=r'OFDM (no div.)')
    ax.semilogy(f_D, np.maximum(results['ofdm_intlv'], 1e-8),
                's--', color='#e377c2', lw=1.1, ms=5, mfc='none', mew=1.2, markevery=(0, 3),
                label=r'OFDM + intlv')
    ax.semilogy(f_D, np.maximum(results['ofdm_fixed'], 1e-8),
                'D--', color='#17becf', lw=1.7, ms=5.5, mfc='none', mew=1.4, markevery=(1, 3),
                label=r'OFDM + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Normalized Doppler $f_D / \Delta f$')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')

    # Two-column legend: OTFS left, OFDM right — grouped for clarity
    ax.legend(fontsize=5.5, loc='lower right', framealpha=0.7, ncol=2,
              columnspacing=0.8, handletextpad=0.4, handlelength=2.0,
              borderpad=0.4, labelspacing=0.35)
    ax.set_ylim(1e-2, 1.1)
    ax.set_xlim(f_D[0], f_D[-1])
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'otfs_vs_ofdm')


def save_csv(results):
    keys = ['otfs_no_intlv', 'otfs_intlv', 'otfs_fixed',
            'otfs_frac', 'otfs_frac_icsi',
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


def _mc_trial_gdiv_only(gamma_avg_SU, gamma_gs, N, K, n_subchannels,
                        max_erasures, K_rice, epsilon_th,
                        p_block_sub, shadow_loss_dB,
                        polar_code, rng, f_D_norm):
    """MC trial for only the 3 OTFS + G_DIV curves (much faster)."""
    gamma_e2e_base, blocked_mask = generate_multipath_per_subchannel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
        p_block_sub, shadow_loss_dB, rng)

    R = K / N
    N_sub = N // n_subchannels
    N_D = OTFS_PARAMS.N_D
    max_doppler_bin = max(1, int(round(f_D_norm * N_D)))
    max_doppler_bin = min(max_doppler_bin, N_D // 2)
    doppler_bins = [0, max_doppler_bin]

    h_dd, noise_var, gamma_e2e_otfs_dd, blocked_mask_otfs = generate_dd_channel(
        gamma_avg_SU, gamma_gs, n_subchannels, K_rice,
        p_block_sub, shadow_loss_dB, rng, doppler_bins=doppler_bins)
    erased_mask_otfs = np.array([
        conditional_bler(gamma_e2e_otfs_dd[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)])
    n_erased_otfs = int(np.sum(erased_mask_otfs))

    eta_inst_otfs = np.zeros(n_subchannels)
    for j in range(n_subchannels):
        if not erased_mask_otfs[j]:
            eta_inst_otfs[j] = 1.0 - conditional_bler(gamma_e2e_otfs_dd[j], N_sub, R)

    info_bits = rng.integers(0, 2, size=K).astype(np.int8)

    # OTFS + G_DIV (integer Doppler, perfect CSI)
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

    # Fractional Doppler models
    alpha_frac = 0.5
    sigma_frac_sq = (np.pi * alpha_frac) ** 2 / (3.0 * N_D)
    sigma_csi_sq = 0.01

    gamma_e2e_frac = gamma_e2e_otfs_dd / (1.0 + gamma_e2e_otfs_dd * sigma_frac_sq)
    gamma_e2e_frac_icsi = gamma_e2e_otfs_dd / (
        1.0 + gamma_e2e_otfs_dd * (sigma_frac_sq + sigma_csi_sq))

    erased_frac = np.array([
        conditional_bler(gamma_e2e_frac[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)])
    erased_frac_icsi = np.array([
        conditional_bler(gamma_e2e_frac_icsi[j], N_sub, R) > epsilon_th
        for j in range(n_subchannels)])

    eta_frac = np.array([0.0 if erased_frac[j] else
                         1.0 - conditional_bler(gamma_e2e_frac[j], N_sub, R)
                         for j in range(n_subchannels)])
    eta_frac_icsi = np.array([0.0 if erased_frac_icsi[j] else
                              1.0 - conditional_bler(gamma_e2e_frac_icsi[j], N_sub, R)
                              for j in range(n_subchannels)])

    # Fractional Doppler, perfect CSI
    if int(np.sum(erased_frac)) > max_erasures:
        err_otfs_frac = True
    else:
        coded_bits_f = polar_code.encode(info_bits)
        llr_frac = diversity_transform_transmit_receive(
            coded_bits_f, G_FIX, gamma_e2e_frac, erased_frac, eta_frac, rng)
        if llr_frac is None:
            err_otfs_frac = True
        else:
            err_otfs_frac = not np.array_equal(polar_code.decode(llr_frac), info_bits)

    # Fractional Doppler + imperfect CSI
    if int(np.sum(erased_frac_icsi)) > max_erasures:
        err_otfs_frac_icsi = True
    else:
        llr_icsi = diversity_transform_transmit_receive(
            coded_bits_f, G_FIX, gamma_e2e_frac_icsi, erased_frac_icsi,
            eta_frac_icsi, rng)
        if llr_icsi is None:
            err_otfs_frac_icsi = True
        else:
            err_otfs_frac_icsi = not np.array_equal(polar_code.decode(llr_icsi), info_bits)

    return {
        'otfs_fixed': err_otfs_fix,
        'otfs_frac': err_otfs_frac,
        'otfs_frac_icsi': err_otfs_frac_icsi,
    }


def _worker_gdiv_only(args):
    """Worker for one Doppler point — only 3 G_DIV curves."""
    (idx, f_D_norm, n_mc, seed, gamma_su, gamma_gs, N, K,
     n_subchannels, max_erasures, K_rice, epsilon_th,
     p_block_sub, shadow_loss_dB) = args

    rng = np.random.default_rng(seed)
    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    curve_keys = ['otfs_fixed', 'otfs_frac', 'otfs_frac_icsi']
    err_counts = {k: 0 for k in curve_keys}

    for t in range(n_mc):
        res = _mc_trial_gdiv_only(
            gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
            K_rice, epsilon_th, p_block_sub, shadow_loss_dB,
            polar_code, rng, f_D_norm)
        for k in curve_keys:
            err_counts[k] += int(res[k])

    return idx, {k: v / n_mc for k, v in err_counts.items()}


def simulate_gdiv_only(N=256, K=128, n_subchannels=N_S,
                       max_erasures=DT_CONFIG.max_erasures,
                       P_total_dBW=20.0, ref_elevation_deg=50.0,
                       f_D_norm_range=(0.0, 0.35), n_points=20,
                       n_mc=15000, n_workers=None):
    """Run only the 3 OTFS + G_DIV curves with high MC count."""
    import multiprocessing as mp
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_points)

    f_D_arr = np.linspace(f_D_norm_range[0], f_D_norm_range[1], n_points)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                                  spread_deg=30.0)
    gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

    K_rice = 2.0
    epsilon_th = 0.1
    shadow_loss_dB = 20.0
    p_block_sub = p_block_3gpp(elev_angles)

    print(f"OTFS + G_DIV only: {n_points} Doppler points, MC={n_mc}, {n_workers} workers")

    base_seed = 2030
    worker_args = []
    for idx, f_D_norm in enumerate(f_D_arr):
        worker_args.append((
            idx, float(f_D_norm), n_mc, base_seed + idx * 1000,
            gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
            K_rice, epsilon_th, p_block_sub, shadow_loss_dB
        ))

    curve_keys = ['otfs_fixed', 'otfs_frac', 'otfs_frac_icsi']
    results = {'f_D_norm': f_D_arr}
    for key in curve_keys:
        results[key] = np.zeros(n_points)

    with mp.Pool(n_workers) as pool:
        for idx, bler_dict in pool.imap_unordered(_worker_gdiv_only, worker_args):
            f_D_norm = f_D_arr[idx]
            for k, v in bler_dict.items():
                results[k][idx] = v
            print(f"  [{idx+1}/{n_points}] f_D={f_D_norm:.4f}: "
                  f"Fix={results['otfs_fixed'][idx]:.4e}, "
                  f"Frac={results['otfs_frac'][idx]:.4e}, "
                  f"Frac+iCSI={results['otfs_frac_icsi'][idx]:.4e}")

    return results


def merge_and_plot():
    """Load existing CSV for non-G_DIV curves, run G_DIV at 15k MC, merge and plot."""
    import csv as csv_mod

    # Load existing CSV for the 5 non-G_DIV curves
    existing = {}
    with open('otfs_vs_ofdm.csv', 'r') as f:
        reader = csv_mod.DictReader(f)
        for key in reader.fieldnames:
            existing[key] = []
        for row in reader:
            for key in reader.fieldnames:
                existing[key].append(float(row[key]))
    for key in existing:
        existing[key] = np.array(existing[key])

    old_fD = existing['f_D_norm']

    # Run G_DIV curves at 15k MC with 20 points
    gdiv_results = simulate_gdiv_only(n_mc=15000, n_points=20, n_workers=16)
    new_fD = gdiv_results['f_D_norm']

    # Interpolate old non-G_DIV curves to new x-axis
    non_gdiv_keys = ['otfs_no_intlv', 'otfs_intlv',
                     'ofdm_no_intlv', 'ofdm_intlv', 'ofdm_fixed']
    merged = {'f_D_norm': new_fD}
    for key in non_gdiv_keys:
        merged[key] = np.interp(new_fD, old_fD, existing[key])
    for key in ['otfs_fixed', 'otfs_frac', 'otfs_frac_icsi']:
        merged[key] = gdiv_results[key]

    plot(merged)
    save_csv(merged)
    return merged


if __name__ == "__main__":
    results = merge_and_plot()
