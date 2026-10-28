"""
BLER vs Total Transmit Power — Prediction robustness study.

Shows how the proposed diversity-transform scheme degrades when the
ephemeris-driven prediction is imperfect.  The combined prediction
error sigma_p (dB) captures:
  - Ephemeris / orbital uncertainty  (~0.5 dB per 5 deg elevation error)
  - UE location uncertainty          (~0.1-0.5 dB)
  - Atmospheric model mismatch       (~1-3 dB)

Curves:
  - Proposed (sigma_p = 0 dB)  — perfect prediction (reference)
  - Proposed (sigma_p = 2 dB)  — small combined error
  - Proposed (sigma_p = 4 dB)  — medium combined error
  - Proposed (sigma_p = 6 dB)  — large combined error
  - Polar (no diversity)       — baseline unaffected by prediction
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    p_block_3gpp, mc_trial_prediction_robustness,
)
from polar_code import PolarCode


SIGMA_LEVELS_DB = [0.0, 2.0, 4.0]


def simulate(N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             power_range_dBW=(-6, 25), ref_elevation_deg=40.0,
             n_points=16, n_mc=5000):
    rng = np.random.default_rng(2026)
    power_dBW_arr = np.linspace(power_range_dBW[0], power_range_dBW[1],
                                n_points)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg,
                                              n_subchannels,
                                              spread_deg=30.0)

    K_rice = 2.0
    epsilon_th = 0.1
    shadow_loss_dB = 20.0
    p_block_sub = p_block_3gpp(elev_angles)

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    N_sub = N // n_subchannels
    K_sub = K // n_subchannels
    N_sub_p2 = 1 << (N_sub.bit_length() - 1) if N_sub > 0 else 1
    K_sub_p2 = max(1, int(K_sub * N_sub_p2 / max(N_sub, 1)))
    K_sub_p2 = min(K_sub_p2, N_sub_p2 - 1)
    polar_sub = PolarCode(N_sub_p2, K_sub_p2, design_snr_dB=1.0, list_size=8)

    print("Prediction robustness: BLER vs P_total")
    print(f"  Polar({N},{K}), Rate={K/N:.2f}, MC={n_mc}")
    print(f"  Elevation angles: {elev_angles}")
    print(f"  Sigma levels (dB): {SIGMA_LEVELS_DB}")

    results = {'P_total_dBW': power_dBW_arr}
    for sigma in SIGMA_LEVELS_DB:
        results[f'proposed_{sigma:.0f}dB'] = np.zeros(n_points)
    results['baseline'] = np.zeros(n_points)

    for idx, p_dBW in enumerate(power_dBW_arr):
        gamma_gs, gamma_su = per_subchannel_link_snrs(p_dBW, elev_angles)

        counts = {f'proposed_{sigma:.0f}dB': 0 for sigma in SIGMA_LEVELS_DB}
        count_baseline = 0

        for _ in range(n_mc):
            seed = int(rng.integers(0, 2**32))
            for sigma in SIGMA_LEVELS_DB:
                trial_rng = np.random.default_rng(seed)
                res = mc_trial_prediction_robustness(
                    gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                    K_rice, epsilon_th, p_block_sub, shadow_loss_dB,
                    polar_code, trial_rng,
                    sigma_pred_dB=sigma,
                    polar_sub=polar_sub)
                counts[f'proposed_{sigma:.0f}dB'] += int(res['err_proposed'])
                if sigma == 0.0:
                    count_baseline += int(res['err_baseline'])

        for sigma in SIGMA_LEVELS_DB:
            key = f'proposed_{sigma:.0f}dB'
            results[key][idx] = counts[key] / n_mc
        results['baseline'][idx] = count_baseline / n_mc

        line = f"  P={p_dBW:5.1f}dBW:"
        for sigma in SIGMA_LEVELS_DB:
            key = f'proposed_{sigma:.0f}dB'
            line += f"  {sigma:.0f}dB={results[key][idx]:.3e}"
        line += f"  base={results['baseline'][idx]:.3e}"
        print(line)

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    p = results['P_total_dBW']

    styles = [
        ('proposed_0dB', 'D-',  '#ff7f0e', 1.7,
         r'Proposed ($\sigma_p = 0$ dB, perfect)'),
        ('proposed_2dB', 's-',  '#1f77b4', 1.4,
         r'Proposed ($\sigma_p = 2$ dB)'),
        ('proposed_4dB', '^-',  '#2ca02c', 1.4,
         r'Proposed ($\sigma_p = 4$ dB)'),
        ('baseline',     'v--', '#7f7f7f', 1.4,
         r'Polar (no diversity)'),
    ]

    for key, fmt, color, lw, label in styles:
        ax.semilogy(p, np.maximum(results[key], 1e-8),
                    fmt, color=color, lw=lw, ms=5.6, label=label)

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.legend(fontsize=6, loc='lower left', framealpha=1.0,
              handletextpad=0.4, handlelength=1.5,
              borderpad=0.4, labelspacing=0.3)
    ax.set_xlim(0, 25)
    ax.set_ylim(0.08, 1.1)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'bler_robustness_prediction')


def save_csv(results):
    keys = [f'proposed_{s:.0f}dB' for s in SIGMA_LEVELS_DB] + ['baseline']
    with open('bler_robustness_prediction.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_total_dBW'] + keys)
        for i in range(len(results['P_total_dBW'])):
            row = [f"{results['P_total_dBW'][i]:.2f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_robustness_prediction.csv")


if __name__ == "__main__":
    results = simulate(n_mc=10000)
    plot(results)
    save_csv(results)
