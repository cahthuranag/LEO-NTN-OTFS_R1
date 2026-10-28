"""
BLER vs Total Transmit Power — 5-curve comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS, OTFS_PARAMS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    create_ephemeris_predictor, mc_trial_one_snr,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    per_subchannel_eta_ephemeris, p_block_3gpp,
    db_to_linear, linear_to_db, ALPHA_POWER,
    F_FEEDER_HZ, G_TX_GW_DBI, G_RX_SAT_DBI, D_FEEDER_M,
    F_ACCESS_HZ, G_TX_SAT_DBI, G_RX_UE_DBI,
)
from polar_code import PolarCode


def _worker_power_point(args):
    """Worker for one power point — runs n_mc trials."""
    (idx, p_dBW, n_mc, seed, N, K, n_subchannels, max_erasures,
     elev_angles, K_rice, K_rice_dB, epsilon_th, shadow_loss_dB,
     p_block_sub, eta_ephemeris, gamma_gs, gamma_su) = args

    rng = np.random.default_rng(seed)
    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    ppv_no_acc, ppv_fix_acc = 0.0, 0.0
    err_counts = [0, 0, 0]

    for _ in range(n_mc):
        res = mc_trial_one_snr(
            gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
            K_rice, epsilon_th, p_block_sub, shadow_loss_dB,
            polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris)
        ppv_no_acc += res['ppv_no_div']
        ppv_fix_acc += res['ppv_fix_div']
        err_counts[0] += int(res['err_1'])
        err_counts[1] += int(res['err_2'])
        err_counts[2] += int(res['err_3'])

    return idx, {
        'ppv_no_diversity': ppv_no_acc / n_mc,
        'ppv_fixed_diversity': ppv_fix_acc / n_mc,
        'no_interleaver': err_counts[0] / n_mc,
        'interleaver': err_counts[1] / n_mc,
        'fixed': err_counts[2] / n_mc,
    }


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             power_range_dBW=(-6, 25), ref_elevation_deg=40.0,
             n_points=16, n_mc=5000, n_workers=None):
    import multiprocessing as mp
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_points)

    power_dBW_arr = np.linspace(power_range_dBW[0], power_range_dBW[1], n_points)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                                  spread_deg=30.0)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    epsilon_th = 0.1
    shadow_loss_dB = 20.0
    p_block_sub = p_block_3gpp(elev_angles)
    R = K / N

    print(f"BLER vs P_total (5 curves, {n_workers} workers)")
    print(f"  Polar({N},{K}), Rate={R:.2f}, MC={n_mc}")

    result_keys = ['ppv_no_diversity', 'ppv_fixed_diversity',
                   'no_interleaver', 'interleaver', 'fixed']
    results = {key: np.zeros(n_points) for key in result_keys}
    results['P_total_dBW'] = power_dBW_arr

    base_seed = 2026
    worker_args = []
    for idx, p_dBW in enumerate(power_dBW_arr):
        gamma_gs, gamma_su = per_subchannel_link_snrs(p_dBW, elev_angles)
        eta_ephemeris = per_subchannel_eta_ephemeris(
            elev_angles, p_dBW, K_rice_dB, R, p_block_per_sub=p_block_sub)
        worker_args.append((
            idx, float(p_dBW), n_mc, base_seed + idx * 1000,
            N, K, n_subchannels, max_erasures,
            elev_angles, K_rice, K_rice_dB, epsilon_th, shadow_loss_dB,
            p_block_sub, eta_ephemeris, gamma_gs, gamma_su
        ))

    with mp.Pool(n_workers) as pool:
        for idx, bler_dict in pool.imap_unordered(_worker_power_point, worker_args):
            for k in result_keys:
                results[k][idx] = bler_dict[k]
            p_dBW = power_dBW_arr[idx]
            print(f"  [{idx+1}/{n_points}] P={p_dBW:5.1f}dBW: "
                  f"NoIntlv={bler_dict['no_interleaver']:.3e}, "
                  f"Intlv={bler_dict['interleaver']:.3e}, "
                  f"Fix={bler_dict['fixed']:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p = results['P_total_dBW']

    ax.semilogy(p, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=1.0, ms=4,
                label=r'PPV bound (no diversity)')
    ax.semilogy(p, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=1.0, ms=4,
                label=r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    ax.semilogy(p, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label=r'Polar (no interleaver)')
    ax.semilogy(p, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label=r'Polar + interleaver')
    ax.semilogy(p, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label=r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.legend(fontsize=6.5, loc='lower left', ncol=2, framealpha=1.0,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.5,
              borderpad=0.4, labelspacing=0.3)
    ax.set_xlim(0, 26)
    ax.set_ylim(1e-2, 1.1)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'bler_comparison')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity',
            'no_interleaver', 'interleaver', 'fixed']
    with open('bler_vs_power.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_total_dBW'] + keys)
        for i in range(len(results['P_total_dBW'])):
            row = [f"{results['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_power.csv")


if __name__ == "__main__":
    results = simulate(n_mc=50000, n_workers=16)
    plot(results)
    save_csv(results)
