"""
BLER vs Blocklength — finite-blocklength performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, L_TAPS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    per_subchannel_eta_ephemeris, p_block_3gpp,
    create_ephemeris_predictor, mc_trial_one_snr,
    db_to_linear, linear_to_db,
)
from polar_code import PolarCode


def simulate(K_msg=32, blocklengths=None, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=10.0, ref_elevation_deg=50.0, n_mc=2000,
             shadow_loss_dB=20.0):
    if blocklengths is None:
        blocklengths = [64, 128, 256, 512]

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                                  spread_deg=30.0)
    gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    epsilon_th = 0.1
    # 3GPP TR 38.811 elevation-dependent blockage (Eq. 16)
    p_block_sub = p_block_3gpp(elev_angles)

    n_pts = len(blocklengths)
    results = {key: np.zeros(n_pts) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity',
        'no_interleaver', 'interleaver', 'fixed']}
    results['blocklength'] = np.array(blocklengths)

    print(f"BLER vs Blocklength (K={K_msg}, P_total={P_total_dBW} dBW, MC={n_mc})")

    for idx, N in enumerate(blocklengths):
        rng = np.random.default_rng(2029 + idx)

        K = K_msg
        R = K / N
        polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

        eta_ephemeris = per_subchannel_eta_ephemeris(
            elev_angles, P_total_dBW, K_rice_dB, R, p_block_per_sub=p_block_sub)

        ppv_acc = [0.0, 0.0]
        err_counts = [0, 0, 0]

        for _ in range(n_mc):
            res = mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, epsilon_th, p_block_sub, shadow_loss_dB,
                polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris)
            ppv_acc[0] += res['ppv_no_div']
            ppv_acc[1] += res['ppv_fix_div']
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])

        results['ppv_no_diversity'][idx] = ppv_acc[0] / n_mc
        results['ppv_fixed_diversity'][idx] = ppv_acc[1] / n_mc
        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc

        print(f"  N={N:5d}, K={K:4d}, R={R:.2f}: Fix={results['fixed'][idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bl = results['blocklength']

    ax.semilogy(bl, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=1.0, ms=4,
                label=r'PPV bound (no diversity)')
    ax.semilogy(bl, np.maximum(results['ppv_fixed_diversity'], 1e-8),
                'p--', color='#17becf', lw=1.0, ms=4,
                label=r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    ax.semilogy(bl, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label=r'Polar (no interleaver)')
    ax.semilogy(bl, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label=r'Polar + interleaver')
    ax.semilogy(bl, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label=r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Blocklength $N$')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.legend(fontsize=6, loc='lower left', framealpha=1.0)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(results['blocklength'])
    ax.set_xticklabels([str(int(x)) for x in results['blocklength']])

    save_figure(fig, 'bler_vs_blocklength')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity',
            'no_interleaver', 'interleaver', 'fixed']
    with open('bler_vs_blocklength.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['blocklength'] + keys)
        for i in range(len(results['blocklength'])):
            row = [str(int(results['blocklength'][i]))]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_blocklength.csv")


if __name__ == "__main__":
    results = simulate(n_mc=10000)
    plot(results)
    save_csv(results)
