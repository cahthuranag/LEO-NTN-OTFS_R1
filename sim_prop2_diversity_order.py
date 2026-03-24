"""
Proposition 2 Validation: High-SNR Diversity Order.

Uses pure Rayleigh fading (K=0, no LOS) per paper's Eq. (51)-(52):
  ε ≈ Υ(L, γ_th / γ̄_br)   with diversity order L

Shows that the BLER slope in log-log reveals diversity order = L
for different numbers of multipath taps L=1,2,3,4.

MC outage, Prop 1 closed-form, and Prop 2 high-SNR are compared,
with measured slopes annotated.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from sim_common import ieee_setup, save_figure, db_to_linear
from fbl_analysis import FiniteBlocklengthBLER


def simulate(R=0.5, snr_range_dB=(-5, 35), n_points=80,
             n_mc=500000, n_mc_points=20, L_values=None):
    """
    Pure Rayleigh fading: K=0, all L paths are NLOS with equal power.
    γ ~ Gamma(L, γ̄/L), outage = Υ(L, γ_th × L / γ̄).
    """
    if L_values is None:
        L_values = [1, 2, 3, 4]

    rng = np.random.default_rng(2041)
    gamma_th = 2.0 ** R - 1.0

    # Dense grid for analytical curves
    snr_dB_arr = np.linspace(snr_range_dB[0], snr_range_dB[1], n_points)
    snr_lin_arr = db_to_linear(snr_dB_arr)

    # Coarser grid for MC points
    snr_dB_mc = np.linspace(snr_range_dB[0], snr_range_dB[1], n_mc_points)
    snr_lin_mc = db_to_linear(snr_dB_mc)

    results = {'snr_dB': snr_dB_arr, 'snr_dB_mc': snr_dB_mc}

    print(f"Prop 2 Validation: Diversity Order — Pure Rayleigh (K=0)")
    print(f"  R={R:.2f}, γ_th={gamma_th:.4f}, MC={n_mc}")

    for L in L_values:
        bler_closed = np.zeros(n_points)
        bler_high_snr = np.zeros(n_points)
        bler_mc = np.zeros(n_mc_points)

        print(f"\n  L={L} taps (expected diversity order = {L}):")

        # Analytical curves (dense)
        for idx, (snr_dB, gamma_bar) in enumerate(zip(snr_dB_arr, snr_lin_arr)):
            gamma_bar_br = gamma_bar / L

            bler_closed[idx] = FiniteBlocklengthBLER.bler_rayleigh(
                gamma_th, gamma_bar_br, L)

            x = gamma_th / gamma_bar_br
            if x < 700:
                bler_high_snr[idx] = x**L * math.exp(-x) / math.factorial(L)
            else:
                bler_high_snr[idx] = 0.0

        # MC simulation (coarse grid)
        for idx, (snr_dB, gamma_bar) in enumerate(zip(snr_dB_mc, snr_lin_mc)):
            gamma_bar_br = gamma_bar / L
            gamma_samples = np.sum(
                rng.exponential(gamma_bar_br, size=(n_mc, L)), axis=1)
            bler_mc[idx] = np.mean(gamma_samples < gamma_th)

        results[f'closed_L{L}'] = bler_closed
        results[f'highsnr_L{L}'] = bler_high_snr
        results[f'mc_L{L}'] = bler_mc

        # Estimate slope from closed-form in the high-SNR region
        high_mask = (bler_closed > 1e-6) & (bler_closed < 0.05) & (snr_lin_arr > 3)
        if np.sum(high_mask) >= 3:
            log_snr = np.log10(snr_lin_arr[high_mask])
            log_bler = np.log10(bler_closed[high_mask])
            slope = np.polyfit(log_snr, log_bler, 1)[0]
            results[f'slope_L{L}'] = -slope
            print(f"    Measured slope: {-slope:.2f} (expected: {L})")
        else:
            results[f'slope_L{L}'] = float('nan')
            print(f"    Insufficient data for slope")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    snr = results['snr_dB']
    snr_mc = results['snr_dB_mc']

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    mkrs = ['o', 's', '^', 'D']

    # Annotation positions for L labels — placed in clear areas between curves
    L_annot = {
        1: (25, 5e-3),
        2: (20, 3e-4),
        3: (14, 5e-5),
        4: (3, 2e-3),
    }
    bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=1.0)

    for i, L in enumerate([1, 2, 3, 4]):
        cf_key = f'closed_L{L}'
        hs_key = f'highsnr_L{L}'
        mc_key = f'mc_L{L}'
        if cf_key not in results:
            continue

        cf = results[cf_key]
        hs = results[hs_key]
        mc = results[mc_key]

        # Corollary 1 (dashed, no label per L)
        mask_cf = cf > 1e-8
        if np.any(mask_cf):
            ax.semilogy(snr[mask_cf], cf[mask_cf],
                        linestyle='--', color=colors[i], lw=1.5)

        # Prop. 2 (solid, no label per L)
        mask_hs = (hs > 1e-8) & (hs < 1.0)
        if np.any(mask_hs):
            ax.semilogy(snr[mask_hs], hs[mask_hs],
                        linestyle='-', color=colors[i], lw=1.5)

        # MC (markers, no label per L)
        mask_mc = mc > 1e-7
        if np.any(mask_mc):
            ax.semilogy(snr_mc[mask_mc], mc[mask_mc],
                        linestyle='none', marker=mkrs[i],
                        color=colors[i], ms=4, mew=1.0,
                        mfc='none')

        # Annotate L value near the curves
        if L in L_annot:
            ax.text(L_annot[L][0], L_annot[L][1],
                    f'$L\\!={L}$', fontsize=9, color=colors[i],
                    ha='center', va='center', fontweight='bold',
                    bbox=bbox_props)

    # Grouped legend: one entry per type
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], linestyle='--', color='black', lw=1.5,
               label='Corollary 1'),
        Line2D([0], [0], linestyle='-', color='black', lw=1.5,
               label='Prop. 2'),
        Line2D([0], [0], linestyle='none', marker='o', color='black',
               ms=4, mew=1.0, mfc='none',
               label='MC simulation'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right',
              framealpha=1.0)

    ax.set_xlabel(r'Average SNR $\bar{\gamma}$ (dB)')
    ax.set_ylabel(r'BLER $\epsilon_i^{\mathrm{E2E}}$')
    ax.set_xlim(results['snr_dB'][0], results['snr_dB'][-1])
    ax.set_ylim(1e-6, 1)
    ax.set_xticks(np.arange(-5, 40, 5))
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'prop2_diversity_order')


def save_csv(results):
    L_values = [1, 2, 3, 4]

    # Save analytical curves (dense grid)
    ana_keys = []
    for L in L_values:
        for prefix in ['closed', 'highsnr']:
            k = f'{prefix}_L{L}'
            if k in results:
                ana_keys.append(k)

    with open('prop2_diversity_order.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_dB'] + ana_keys)
        for i in range(len(results['snr_dB'])):
            row = [f"{results['snr_dB'][i]:.2f}"]
            for k in ana_keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)

    # Save MC data (coarse grid)
    mc_keys = [f'mc_L{L}' for L in L_values if f'mc_L{L}' in results]
    with open('prop2_diversity_order_mc.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_dB'] + mc_keys)
        for i in range(len(results['snr_dB_mc'])):
            row = [f"{results['snr_dB_mc'][i]:.2f}"]
            for k in mc_keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)

    print("Saved: prop2_diversity_order.csv, prop2_diversity_order_mc.csv")


if __name__ == "__main__":
    results = simulate()
    plot(results)
    save_csv(results)
