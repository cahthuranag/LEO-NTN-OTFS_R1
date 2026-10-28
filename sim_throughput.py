"""
Throughput-Reliability Trade-off — effective throughput accounting for
diversity transform rate overhead.

Baselines:  η = R_c × (1 − BLER)
Diversity:  η = R_c × (ρ/n_out) × (1 − BLER)
MRC/SC:     η = max_R { (R / n_s) × (1 − ε(γ_combined, N, R)) }

MRC/SC repeat the same codeword on all n_s subchannels. Their code
rate is optimised at each operating point to maximise throughput.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, N_S, L_TAPS, DT_CONFIG, RHO, N_OUT,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    SNRCalculator, capacity_awgn,
    generate_multipath_per_subchannel,
    subchannel_elevation_angles, per_subchannel_link_snrs, p_block_3gpp,
    db_to_linear,
)
from fbl_analysis import conditional_bler
from sim_bler_vs_power import simulate as simulate_bler_vs_power


def _optimal_mrc_sc_throughput(gamma_combined, N, n_s):
    """Find optimal code rate R that maximises (R/n_s)*(1-ε(γ,N,R))."""
    best_eta = 0.0
    rate_candidates = np.linspace(0.05, 0.95, 50)
    for R in rate_candidates:
        eps = conditional_bler(gamma_combined, N, R)
        eta = (R / n_s) * (1.0 - eps)
        if eta > best_eta:
            best_eta = eta
    return best_eta


def simulate_mrc_sc_throughput(N=256, n_subchannels=N_S,
                               power_range_dBW=(-6, 25),
                               ref_elevation_deg=40.0,
                               n_points=16, n_mc=5000):
    rng = np.random.default_rng(2027)
    power_dBW_arr = np.linspace(power_range_dBW[0], power_range_dBW[1], n_points)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                              spread_deg=30.0)
    K_rice = 2.0
    shadow_loss_dB = 20.0
    p_block_sub = p_block_3gpp(elev_angles)

    tp_mrc = np.zeros(n_points)
    tp_sc = np.zeros(n_points)

    for idx, p_dBW in enumerate(power_dBW_arr):
        gamma_gs, gamma_su = per_subchannel_link_snrs(p_dBW, elev_angles)
        eta_mrc_sum, eta_sc_sum = 0.0, 0.0

        for _ in range(n_mc):
            gamma_e2e, blocked = generate_multipath_per_subchannel(
                gamma_su, gamma_gs, n_subchannels, K_rice,
                p_block_sub, shadow_loss_dB, rng)

            non_blocked_snr = gamma_e2e[~blocked]
            if len(non_blocked_snr) == 0:
                continue

            gamma_mrc = np.sum(non_blocked_snr)
            eta_mrc_sum += _optimal_mrc_sc_throughput(gamma_mrc, N, n_subchannels)

            gamma_sc = np.max(non_blocked_snr)
            eta_sc_sum += _optimal_mrc_sc_throughput(gamma_sc, N, n_subchannels)

        tp_mrc[idx] = eta_mrc_sum / n_mc
        tp_sc[idx] = eta_sc_sum / n_mc
        print(f"  P={p_dBW:5.1f}dBW: MRC_tp={tp_mrc[idx]:.4f}, "
              f"SC_tp={tp_sc[idx]:.4f}")

    return {'P_total_dBW': power_dBW_arr, 'mrc': tp_mrc, 'sc': tp_sc}


def simulate(results_power=None, N=256, K=128):
    if results_power is None:
        results_power = simulate_bler_vs_power(N=N, K=K, n_mc=10000)

    R = K / N
    R_dt = RHO / N_OUT  # diversity transform rate = 12/18 = 2/3
    R_eff_div = R * R_dt  # effective rate for diversity schemes

    p_arr = results_power['P_total_dBW']
    n_pts = len(p_arr)

    throughput = {'P_total_dBW': p_arr}

    # Baselines: effective rate = R_c
    for key in ['no_interleaver', 'interleaver']:
        throughput[key] = R * (1.0 - np.clip(results_power[key], 0, 1))

    # Diversity schemes: effective rate = R_c × (ρ/n_out)
    for key in ['fixed']:
        throughput[key] = R_eff_div * (1.0 - np.clip(results_power[key], 0, 1))

    # PPV bounds with same rate correction
    throughput['ppv_no_diversity'] = R * (1.0 - np.clip(
        results_power['ppv_no_diversity'], 0, 1))
    throughput['ppv_fixed_diversity'] = R_eff_div * (1.0 - np.clip(
        results_power['ppv_fixed_diversity'], 0, 1))

    # MRC/SC: rate-optimised throughput via dedicated MC
    print("Running rate-optimised MRC/SC throughput simulation...")
    mrc_sc = simulate_mrc_sc_throughput(N=N, n_points=n_pts,
                                        power_range_dBW=(p_arr[0], p_arr[-1]),
                                        n_mc=5000)
    throughput['mrc'] = mrc_sc['mrc']
    throughput['sc'] = mrc_sc['sc']

    ref_slant = slant_range_m(50.0)
    ref_fspl = access_fspl_dB(ref_slant)
    ref_atm_loss = atmospheric_loss_dB(50.0)

    capacity_ref = np.zeros(n_pts)
    for i, p_dBW in enumerate(p_arr):
        gamma_gs, gamma_su = power_to_link_snrs(p_dBW, ref_fspl, ref_atm_loss)
        gamma_e2e_avg = SNRCalculator.cascaded_snr(gamma_gs, gamma_su)
        capacity_ref[i] = capacity_awgn(gamma_e2e_avg)
    throughput['capacity'] = capacity_ref

    print(f"Throughput-Reliability (R_c={R:.2f}, R_dt={R_dt:.4f}, "
          f"R_eff_div={R_eff_div:.4f}):")
    for i in range(0, n_pts, max(1, n_pts // 6)):
        print(f"  P={p_arr[i]:5.1f}dBW: Fix={throughput['fixed'][i]:.4f}, "
              f"NoIntlv={throughput['no_interleaver'][i]:.4f}, "
              f"MRC={throughput['mrc'][i]:.4f}, SC={throughput['sc'][i]:.4f}, "
              f"C={capacity_ref[i]:.4f}")

    return throughput


def plot(throughput):
    from scipy.signal import savgol_filter
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p_arr = throughput['P_total_dBW']

    def smooth(y, window=7, poly=2):
        y = np.asarray(y, dtype=float)
        if len(y) < window:
            window = max(3, len(y) | 1)
        return savgol_filter(y, window, poly)

    ax.plot(p_arr, smooth(throughput['ppv_fixed_diversity']),
            'p--', color='#17becf', lw=1.0, ms=4,
            label=r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    ax.plot(p_arr, throughput['no_interleaver'],
            'v-', color='#7f7f7f', lw=1.0, ms=3,
            label=r'Polar (no interleaver)')
    ax.plot(p_arr, throughput['interleaver'],
            '^-', color='#d62728', lw=1.0, ms=3,
            label=r'Polar + interleaver')
    ax.plot(p_arr, smooth(throughput['fixed']),
            'D-', color='#ff7f0e', lw=1.2, ms=4,
            label=r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$')
    ax.plot(p_arr, throughput['mrc'],
            's-', color='#1f77b4', lw=1.0, ms=3,
            label=r'MRC (rate-optimised)')
    ax.plot(p_arr, throughput['sc'],
            'o-', color='#2ca02c', lw=1.0, ms=3,
            label=r'SC (rate-optimised)')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel('Throughput (bits/channel use)', fontsize=8)
    ax.legend(fontsize=5.5, loc='upper left', framealpha=1.0, ncol=2,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.5)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, None)
    ax.grid(True, ls='--', alpha=0.3)

    save_figure(fig, 'throughput_reliability')


def save_csv(throughput):
    keys = ['capacity', 'ppv_no_diversity', 'ppv_fixed_diversity',
            'no_interleaver', 'interleaver', 'fixed', 'mrc', 'sc']
    with open('throughput_reliability.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_total_dBW'] + keys)
        for i in range(len(throughput['P_total_dBW'])):
            row = [f"{throughput['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{throughput[k][i]:.6e}")
            w.writerow(row)
    print("Saved: throughput_reliability.csv")


if __name__ == "__main__":
    throughput = simulate()
    plot(throughput)
    save_csv(throughput)
