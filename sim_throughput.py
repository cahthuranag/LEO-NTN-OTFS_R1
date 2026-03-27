"""
Throughput-Reliability Trade-off — effective throughput accounting for
diversity transform rate overhead.

Baselines:  η = R_c × (1 − BLER)
Diversity:  η = R_c × (ρ/n_out) × (1 − BLER)

where ρ/n_out = k_c/n_s is the diversity transform code rate.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, N_S, L_TAPS, DT_CONFIG, RHO, N_OUT,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    SNRCalculator, capacity_awgn,
)
from sim_bler_vs_power import simulate as simulate_bler_vs_power


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
              f"C={capacity_ref[i]:.4f}")

    return throughput


def plot(throughput):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p_arr = throughput['P_total_dBW']

    ax.plot(p_arr, throughput['ppv_fixed_diversity'],
            'p--', color='#17becf', lw=1.0, ms=4,
            label=r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    ax.plot(p_arr, throughput['no_interleaver'],
            'v-', color='#7f7f7f', lw=1.0, ms=3,
            label=r'Polar (no interleaver)')
    ax.plot(p_arr, throughput['interleaver'],
            '^-', color='#d62728', lw=1.0, ms=3,
            label=r'Polar + interleaver')
    ax.plot(p_arr, throughput['fixed'],
            'D-', color='#ff7f0e', lw=1.2, ms=4,
            label=r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel('Throughput (bits/channel use)', fontsize=8)
    ax.legend(fontsize=6, loc='upper left', framealpha=1.0)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, None)
    ax.grid(True, ls='--', alpha=0.3)

    save_figure(fig, 'throughput_reliability')


def save_csv(throughput):
    keys = ['capacity', 'ppv_no_diversity', 'ppv_fixed_diversity',
            'no_interleaver', 'interleaver', 'fixed']
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
