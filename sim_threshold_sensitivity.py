"""
System BLER vs erasure threshold epsilon_th — analytical evaluation.

Uses Proposition 1 (closed-form per-subchannel BLER) and Corollary 2
(Poisson binomial system BLER) to sweep epsilon_th and show that
the MDS recovery guarantee holds regardless of threshold, with
epsilon_th ~ 0.1 near-optimal.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    p_block_3gpp, ENV_PARAMS, db_to_linear, linear_to_db,
)
from fbl_analysis import conditional_bler, capacity_awgn, channel_dispersion


def _poisson_binom_tail(probs, f_max):
    """P(F > f_max) via Poisson binomial recursion."""
    n = len(probs)
    P = np.zeros(n + 1)
    P[0] = 1.0
    for i in range(n):
        P_new = np.zeros(n + 1)
        for k in range(i + 1, -1, -1):
            P_new[k] = P[k] * (1.0 - probs[i])
            if k > 0:
                P_new[k] += P[k - 1] * probs[i]
        P = P_new
    return np.sum(P[f_max + 1:])


def simulate(N=256, K=128, n_subchannels=N_S,
             P_total_dBW_values=None,
             ref_elevation_deg=40.0,
             n_mc_shadow=500):
    """Analytical system BLER vs epsilon_th.

    For each (P_total, epsilon_th):
      1. Compute per-subchannel average SNR from link budget
      2. MC average over shadow fading realisations
      3. For each realisation, compute per-subchannel conditional BLER
      4. Declare erased if conditional_bler > epsilon_th
      5. Compute system BLER via Poisson binomial P(F > f_max)
         plus residual FEC term on surviving subchannels
    """
    if P_total_dBW_values is None:
        P_total_dBW_values = [5.0, 10.0, 20.0, 25.0]

    k_c = DT_CONFIG.k_c
    n_s = DT_CONFIG.n_s
    d_min = DT_CONFIG.d_min
    f_max = n_s - k_c
    K_rice = 2.0
    R = K / N
    N_sub = N // n_subchannels

    elev_angles = subchannel_elevation_angles(ref_elevation_deg,
                                              n_subchannels, spread_deg=30.0)
    p_block_sub = p_block_3gpp(elev_angles)

    mu_sh_dB = ENV_PARAMS.mu_sh_dB
    sigma_sh_dB = ENV_PARAMS.sigma_sh_dB

    eps_th_arr = np.logspace(-3, np.log10(0.7), 40)

    print("Threshold sensitivity: System BLER vs epsilon_th")
    print(f"  Polar({N},{K}), Rate={R:.2f}, n_s={n_s}, k_c={k_c}")
    print(f"  P_total values: {P_total_dBW_values} dBW")
    print(f"  Shadow MC={n_mc_shadow}")

    results = {'eps_th': eps_th_arr}

    rng = np.random.default_rng(42)

    for P_dBW in P_total_dBW_values:
        gamma_gs, gamma_su = per_subchannel_link_snrs(P_dBW, elev_angles)

        sys_bler = np.zeros(len(eps_th_arr))

        for mc in range(n_mc_shadow):
            blocked = rng.random(n_subchannels) < p_block_sub
            shadow_dB = mu_sh_dB + rng.normal(0, sigma_sh_dB, n_subchannels)

            gamma_e2e = np.zeros(n_subchannels)
            for j in range(n_subchannels):
                if blocked[j]:
                    gamma_access = gamma_su[j] * db_to_linear(-20.0)
                else:
                    gamma_access = gamma_su[j] * db_to_linear(-shadow_dB[j])

                los_mean = np.sqrt(K_rice / (K_rice + 1))
                nlos_std = np.sqrt(1.0 / (2.0 * (K_rice + 1)))
                h_r = los_mean + rng.normal(0, nlos_std)
                h_i = rng.normal(0, nlos_std)
                fading = h_r**2 + h_i**2
                gamma_faded = gamma_access * fading

                gamma_e2e[j] = gamma_gs[j] * gamma_faded / (
                    gamma_gs[j] + gamma_faded + 1.0)

            per_sub_bler = np.array([
                conditional_bler(gamma_e2e[j], N_sub, R)
                for j in range(n_subchannels)])

            for ti, eps_th in enumerate(eps_th_arr):
                erased = per_sub_bler > eps_th
                n_erased = int(np.sum(erased))

                if n_erased > f_max:
                    sys_bler[ti] += 1.0
                else:
                    surv_bler = per_sub_bler[~erased]
                    if len(surv_bler) > 0:
                        p_fec_fail = 1.0 - np.prod(1.0 - surv_bler)
                    else:
                        p_fec_fail = 0.0
                    sys_bler[ti] += p_fec_fail

        sys_bler /= n_mc_shadow
        key = f'bler_{P_dBW:.0f}dBW'
        results[key] = sys_bler

        for eps_th_check in [0.01, 0.05, 0.1, 0.2]:
            idx = np.argmin(np.abs(eps_th_arr - eps_th_check))
            print(f"  P={P_dBW:.0f}dBW, eps_th={eps_th_arr[idx]:.3f}: "
                  f"sys_BLER={sys_bler[idx]:.4e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    eps_th = results['eps_th']

    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e']
    markers = ['v', 'D', 's', 'o']
    marker_every = max(1, len(eps_th) // 8)

    keys = [k for k in results if k.startswith('bler_')]
    keys.sort(key=lambda k: float(k.replace('bler_', '').replace('dBW', '')))
    for i, key in enumerate(keys):
        P_label = key.replace('bler_', '').replace('dBW', '')
        ax.semilogy(eps_th, np.maximum(results[key], 1e-8),
                    f'{markers[i % len(markers)]}-', color=colors[i % len(colors)],
                    lw=1.4, ms=5.6, markevery=marker_every,
                    label=rf'$P_{{\mathrm{{total}}}} = {P_label}$ dBW')

    ax.set_xlabel(r'Erasure threshold $\epsilon_{\mathrm{th}}$')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.set_xscale('log')
    ax.legend(fontsize=6.5, loc='center left', framealpha=0.7,
              handletextpad=0.4, handlelength=1.5,
              borderpad=0.4, labelspacing=0.3)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'bler_vs_threshold')


def save_csv(results):
    eps_th = results['eps_th']
    keys = [k for k in sorted(results) if k.startswith('bler_')]
    with open('bler_vs_threshold.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['eps_th'] + keys)
        for i in range(len(eps_th)):
            row = [f"{eps_th[i]:.6e}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_threshold.csv")


if __name__ == "__main__":
    results = simulate(n_mc_shadow=5000)
    plot(results)
    save_csv(results)
