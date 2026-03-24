"""
BLER vs Per-Subchannel Blockage Probability — 5-curve comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, Tuple

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    per_subchannel_eta_ephemeris, p_block_3gpp,
    create_ephemeris_predictor, mc_trial_one_snr,
    db_to_linear, linear_to_db,
)
from polar_code import PolarCode


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=8.0, ref_elevation_deg=50.0,
             p_block_range=(0.001, 0.35), n_points=8, n_mc=2000):
    rng = np.random.default_rng(2028)
    p_block_arr = np.linspace(p_block_range[0], p_block_range[1], n_points)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                                  spread_deg=30.0)
    gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    epsilon_th = 0.1
    shadow_loss_dB = 20.0

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)
    R = K / N

    print(f"BLER vs Blockage (5 curves, P_total={P_total_dBW} dBW, MC={n_mc})")

    results = {key: np.zeros(n_points) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity',
        'no_interleaver', 'interleaver', 'fixed']}
    results['p_block'] = p_block_arr

    for idx, p_blk in enumerate(p_block_arr):
        p_blk_arr = np.full(n_subchannels, p_blk)
        eta_ephemeris = per_subchannel_eta_ephemeris(
            elev_angles, P_total_dBW, K_rice_dB, R, p_block_per_sub=p_blk_arr)

        ppv_no_acc, ppv_fix_acc = 0.0, 0.0
        err_counts = [0, 0, 0]

        for _ in range(n_mc):
            res = mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, epsilon_th, p_blk, shadow_loss_dB,
                polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris)
            ppv_no_acc += res['ppv_no_div']
            ppv_fix_acc += res['ppv_fix_div']
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])

        results['ppv_no_diversity'][idx] = ppv_no_acc / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_fix_acc / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc

        print(f"  p_block={p_blk:.3f}: PPV_no={results['ppv_no_diversity'][idx]:.3e}, "
              f"NoIntlv={results['no_interleaver'][idx]:.3e}, "
              f"Intlv={results['interleaver'][idx]:.3e}, "
              f"Fix={results['fixed'][idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p_blk = results['p_block']

    ax.semilogy(p_blk, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=1.0, ms=4,
                label=r'PPV bound (no diversity)')
    ax.semilogy(p_blk, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=1.0, ms=4,
                label=r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    ax.semilogy(p_blk, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label=r'Polar (no interleaver)')
    ax.semilogy(p_blk, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label=r'Polar + interleaver')
    ax.semilogy(p_blk, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label=r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel('Per-subchannel blockage probability')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.legend(fontsize=6, loc='lower right', framealpha=1.0)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    fig.tight_layout()
    save_figure(fig, 'bler_vs_blockage')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity',
            'no_interleaver', 'interleaver', 'fixed']
    with open('bler_vs_blockage.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['p_block'] + keys)
        for i in range(len(results['p_block'])):
            row = [f"{results['p_block'][i]:.4f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_blockage.csv")


if __name__ == "__main__":
    results = simulate(n_mc=10000)
    plot(results)
    save_csv(results)
