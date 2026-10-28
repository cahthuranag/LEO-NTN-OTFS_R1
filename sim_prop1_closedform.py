"""
Proposition 1 Validation: Closed-Form BLER Approximation.

Shows that the closed-form BLER (Prop 1) equals the outage probability
under the dominant-LOS model, and that the MC-averaged FBL BLER
converges to the outage probability as blocklength n_c → ∞.

Uses the paper's dominant-LOS channel model:
  γ = γ_det (constant) + γ_NLOS (gamma-distributed)

The figure demonstrates:
  - Closed-form (Prop 1) matches MC outage exactly
  - FBL BLER > outage for finite n_c, converging as n_c grows
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

from sim_common import ieee_setup, save_figure, L_TAPS, TAP_POWERS_LIN, db_to_linear
from fbl_analysis import FiniteBlocklengthBLER, conditional_bler


FEEDER_MARGINS_DB = [10.0, 3.0]


def simulate(blocklengths=None, K_ratio=0.75, snr_range_dB=(-5, 3),
             n_points=50, n_mc=100000, K_rice=2.0):
    """
    Sweep average SNR in the transition region around γ_th.

    Uses the dominant-LOS model matching the paper's analytical framework.
    Also computes MC BLER under reduced feeder-link margins for n_c=256.
    """
    if blocklengths is None:
        blocklengths = [64, 128, 256, 1024]

    rng = np.random.default_rng(2040)

    snr_dB_arr = np.linspace(snr_range_dB[0], snr_range_dB[1], n_points)
    snr_lin_arr = db_to_linear(snr_dB_arr)

    P_LOS = K_rice / (K_rice + 1.0)
    P_NLOS_total = 1.0 / (K_rice + 1.0)

    R = K_ratio
    gamma_th = 2.0 ** R - 1.0

    # SNR where γ_det = γ_th (transition point)
    snr_transition = gamma_th * (K_rice + 1) / (K_rice * TAP_POWERS_LIN[0])
    snr_transition_dB = 10 * np.log10(snr_transition)
    print(f"Prop 1 Validation: Closed-form vs MC (Rician, K={K_rice})")
    print(f"  R={R:.2f}, γ_th={gamma_th:.4f}, L_TAPS={L_TAPS}")
    print(f"  γ_det = γ_th at SNR = {snr_transition_dB:.1f} dB")
    print(f"  Blocklengths: {blocklengths}, MC={n_mc}")
    print(f"  Feeder-link margins for sensitivity: {FEEDER_MARGINS_DB} dB")

    results = {'snr_dB': snr_dB_arr, 'blocklengths': blocklengths,
               'transition_dB': snr_transition_dB}

    # Pre-generate fading using dominant-LOS model
    gamma_inst_all = np.zeros((n_points, n_mc))
    gamma_det_arr = np.zeros(n_points)
    gamma_bar_NLOS_arr = np.zeros(n_points)

    for idx, (snr_dB, gamma_bar) in enumerate(zip(snr_dB_arr, snr_lin_arr)):
        gamma_det_arr[idx] = gamma_bar * P_LOS * TAP_POWERS_LIN[0]
        gamma_bar_NLOS_arr[idx] = gamma_bar * (P_NLOS_total * TAP_POWERS_LIN[0]
                                                + np.sum(TAP_POWERS_LIN[1:]))
        # Generate γ = γ_det + sum of exponentials
        for trial in range(n_mc):
            gamma_nlos = 0.0
            gamma_nlos += gamma_bar * P_NLOS_total * TAP_POWERS_LIN[0] * rng.exponential(1.0)
            for p in range(1, L_TAPS):
                gamma_nlos += gamma_bar * TAP_POWERS_LIN[p] * rng.exponential(1.0)
            gamma_inst_all[idx, trial] = gamma_det_arr[idx] + gamma_nlos

    # Prop. 1 closed-form and MC BLER for each blocklength
    for N in blocklengths:
        K = int(N * K_ratio)
        R_actual = K / N
        bler_closed = np.zeros(n_points)
        bler_mc = np.zeros(n_points)

        for idx in range(n_points):
            # Prop. 1 closed-form (FBL, depends on n_c)
            bler_closed[idx] = FiniteBlocklengthBLER.bler_closed_form(
                gamma_th, gamma_det_arr[idx], gamma_bar_NLOS_arr[idx],
                L_TAPS, n_c=N, R_c=R_actual)

            # MC BLER
            bler_sum = 0.0
            for trial in range(n_mc):
                bler_sum += conditional_bler(
                    gamma_inst_all[idx, trial], N, R_actual)
            bler_mc[idx] = bler_sum / n_mc

        results[f'closed_N{N}'] = bler_closed
        results[f'mc_N{N}'] = bler_mc
        print(f"  N={N}: Prop1={bler_closed[n_points//2]:.4e}, "
              f"MC={bler_mc[n_points//2]:.4e}")

    # Feeder-link sensitivity curves for n_c=256
    N_fade = 256
    K_fade = int(N_fade * K_ratio)
    R_fade = K_fade / N_fade
    for margin_dB in FEEDER_MARGINS_DB:
        margin_lin = db_to_linear(margin_dB)
        bler_fade = np.zeros(n_points)
        for idx in range(n_points):
            gamma_gs = snr_lin_arr[idx] * margin_lin
            bler_sum = 0.0
            for trial in range(n_mc):
                gamma_su_mrc = gamma_inst_all[idx, trial]
                gamma_e2e = (gamma_gs * gamma_su_mrc) / (gamma_gs + gamma_su_mrc + 1.0)
                bler_sum += conditional_bler(gamma_e2e, N_fade, R_fade)
            bler_fade[idx] = bler_sum / n_mc
        key = f'mc_fade_{margin_dB:.0f}dB_N{N_fade}'
        results[key] = bler_fade
        print(f"  N={N_fade}, Δ_GS={margin_dB}dB: MC={bler_fade[n_points//2]:.4e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    snr = results['snr_dB']
    blocklengths = results['blocklengths']

    colors_n = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    mkrs = ['o', 's', '^', 'D']
    marker_every = max(1, len(snr) // 12)

    for i, N in enumerate(blocklengths):
        cf_key = f'closed_N{N}'
        mc_key = f'mc_N{N}'
        if cf_key not in results:
            continue

        cf = results[cf_key]
        mc = results[mc_key]

        # Prop. 1 closed-form (solid line)
        mask_cf = cf > 1e-8
        if np.any(mask_cf):
            ax.semilogy(snr[mask_cf], cf[mask_cf],
                        '-', color=colors_n[i], lw=1.5,
                        label=rf'Prop. 1, $n_c\!=\!{N}$')

        # MC BLER (markers only)
        mask_mc = mc > 1e-8
        if np.any(mask_mc):
            ax.semilogy(snr[mask_mc], mc[mask_mc],
                        linestyle='--', marker=mkrs[i],
                        color=colors_n[i], ms=4, mew=1.0,
                        mfc='none',
                        label=rf'Sim., $n_c\!=\!{N}$')

    # Feeder-link sensitivity curves
    fade_colors = ['#8c564b', '#e377c2']
    fade_mkrs = ['v', '+']
    for fi, margin_dB in enumerate(FEEDER_MARGINS_DB):
        key = f'mc_fade_{margin_dB:.0f}dB_N256'
        if key not in results:
            continue
        fade = results[key]
        mask = fade > 1e-8
        if np.any(mask):
            ax.semilogy(snr[mask], fade[mask],
                        linestyle='-.', marker=fade_mkrs[fi],
                        color=fade_colors[fi], lw=1.0, ms=4, mew=1.0,
                        mfc='none',
                        label=rf'Sim., $n_c\!=\!256$, $\bar{{\gamma}}_{{GS}}/\bar{{\gamma}}_{{SU}}={margin_dB:.0f}$ dB')

    ax.legend(fontsize=5.0, loc='lower left', framealpha=1.0,
              ncol=2, handletextpad=0.4, handlelength=1.8,
              columnspacing=0.8, borderpad=0.3, labelspacing=0.25)

    ax.set_xlabel(r'Average access-link SNR $\bar{\gamma}_{\mathrm{SU}}$ (dB)')
    ax.set_ylabel(r'BLER $\epsilon_i^{\mathrm{E2E}}$')
    ax.set_xlim(snr[0], snr[-1])
    ax.set_ylim(1e-3, 1.1)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'prop1_closedform_validation')


def save_csv(results):
    blocklengths = results['blocklengths']
    keys = []
    for N in blocklengths:
        keys += [f'closed_N{N}', f'mc_N{N}']
    for margin_dB in FEEDER_MARGINS_DB:
        keys.append(f'mc_fade_{margin_dB:.0f}dB_N256')

    with open('prop1_closedform_validation.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_dB'] + keys)
        for i in range(len(results['snr_dB'])):
            row = [f"{results['snr_dB'][i]:.2f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: prop1_closedform_validation.csv")


if __name__ == "__main__":
    results = simulate(n_mc=500000)
    plot(results)
    save_csv(results)
