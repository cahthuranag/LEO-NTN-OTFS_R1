"""
Latency-reliability trade-off: Diversity Transform vs HARQ-CC (Corollary 3).

Monte Carlo validation using the same OTFS DD-domain channel model,
QPSK modulation, Rician fading, 3GPP blockage, and shadow fading as
the other simulations.

HARQ-CC model includes chase combining gain: after k retransmissions
the receiver combines all k+1 copies, yielding effective SNR =
gamma * (k+1). Persistent blockage is assumed across retransmissions.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.special import comb
from scipy.stats import norm

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, OTFS_PARAMS,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    p_block_3gpp, generate_dd_channel,
    db_to_linear, linear_to_db,
)
from fbl_analysis import capacity_awgn, channel_dispersion

# Physical constants
RTT_MS = 20.0        # 3GPP LEO 600km bent-pipe RTT
MAX_HARQ_RETX = 4
DELTA_F_HZ = OTFS_PARAMS.delta_f_Hz
M_D = OTFS_PARAMS.M_D


def _epsilon_star(eps_sys_target, n_s, d_min):
    """Relaxed per-subchannel target — Eq. (53)."""
    binom_coeff = comb(n_s, d_min, exact=True)
    return (eps_sys_target / binom_coeff) ** (1.0 / d_min)


def _min_blocklength(eps_target, gamma, R_c):
    """Minimum blocklength — Eq. (43)."""
    if gamma <= 0 or eps_target <= 0 or eps_target >= 1:
        return np.inf
    C = capacity_awgn(gamma)
    V = channel_dispersion(gamma)
    if C <= R_c or V <= 0:
        return np.inf
    q_inv = norm.ppf(1.0 - eps_target)
    n_c = V * q_inv**2 / (C - R_c)**2
    return n_c


def _frame_duration_ms(n_c):
    """OTFS frame duration T_frame = n_c / (M_D * delta_f) in ms."""
    return n_c / (M_D * DELTA_F_HZ) * 1e3


def _bler_at_snr(gamma, n_c, R_c):
    """PPV BLER at given SNR and blocklength."""
    if gamma <= 0 or n_c <= 0:
        return 1.0
    C = capacity_awgn(gamma)
    V = channel_dispersion(gamma)
    if C <= R_c or V <= 0:
        return 1.0
    arg = np.sqrt(n_c / V) * (C - R_c)
    return float(norm.cdf(-arg))


_NC_GRID = np.logspace(-1, 3, 80)  # precomputed blocklength grid


def _harqcc_latency(eps_sys_target, gamma, R_c, max_retx=MAX_HARQ_RETX):
    """HARQ-CC with chase combining gain.

    After attempt k (0-indexed), effective SNR = gamma * (k+1).
    BLER at attempt k = bler(gamma*(k+1), n_c, R_c).
    Blocklength n_c is fixed across attempts.

    System BLER after K+1 attempts = product of per-attempt BLERs.
    Searches over blocklengths to find minimum average latency.
    """
    best_latency = np.inf
    best_K = 0

    # Precompute BLER for all (n_c, k) combinations at this gamma
    n_grid = len(_NC_GRID)
    eps_table = np.zeros((n_grid, max_retx + 1))
    for gi, n_c in enumerate(_NC_GRID):
        for k in range(max_retx + 1):
            eps_table[gi, k] = _bler_at_snr(gamma * (k + 1), n_c, R_c)

    for gi, n_c in enumerate(_NC_GRID):
        T_frame = _frame_duration_ms(n_c)

        for K in range(max_retx + 1):
            # System BLER = prod of per-attempt BLERs
            p_all_fail = 1.0
            for k in range(K + 1):
                p_all_fail *= eps_table[gi, k]

            if p_all_fail > eps_sys_target:
                continue

            # Average latency
            avg_lat = 0.0
            p_still_failing = 1.0
            for k in range(K + 1):
                p_success_k = p_still_failing * (1.0 - eps_table[gi, k])
                avg_lat += p_success_k * (T_frame + k * RTT_MS)
                p_still_failing *= eps_table[gi, k]
            avg_lat += p_still_failing * (T_frame + K * RTT_MS)

            if avg_lat < best_latency:
                best_latency = avg_lat
                best_K = K

    return best_latency, best_K


def simulate(R_c=0.5, n_subchannels=N_S,
             P_total_dBW_values=None,
             ref_elevation_deg=50.0, n_mc=5000,
             shadow_loss_dB=20.0):
    """MC-based latency vs target system BLER.

    Uses the same OTFS channel model as other simulations:
      - DD-domain channel with Rician fading (K_rice=2.0)
      - 3GPP TR 38.811 elevation-dependent blockage
      - Log-normal shadow fading (suburban)
      - QPSK modulation parameters
      - Same link budget (per-subchannel elevation angles, atmospheric loss)

    Compares:
      - Diversity transform: single-shot, relaxed eps* target
      - HARQ-CC: chase combining with SNR accumulation across retx
    """
    if P_total_dBW_values is None:
        P_total_dBW_values = [5.0, 10.0]

    # Precompute HARQ-CC latency lookup: for a given gamma, compute
    # latency for all eps targets at once (avoids redundant grid search)
    def _harqcc_batch(eps_arr, gamma, R_c):
        """Compute HARQ-CC latency for all eps targets at given gamma."""
        n_eps_local = len(eps_arr)
        lat_out = np.full(n_eps_local, np.inf)

        # Build BLER table for this gamma
        n_grid = len(_NC_GRID)
        eps_table = np.zeros((n_grid, MAX_HARQ_RETX + 1))
        for gi, n_c in enumerate(_NC_GRID):
            for k in range(MAX_HARQ_RETX + 1):
                eps_table[gi, k] = _bler_at_snr(gamma * (k + 1), n_c, R_c)

        for gi, n_c in enumerate(_NC_GRID):
            T_frame = _frame_duration_ms(n_c)

            for K in range(MAX_HARQ_RETX + 1):
                p_all_fail = 1.0
                for k in range(K + 1):
                    p_all_fail *= eps_table[gi, k]

                avg_lat = 0.0
                p_still_failing = 1.0
                for k in range(K + 1):
                    p_success_k = p_still_failing * (1.0 - eps_table[gi, k])
                    avg_lat += p_success_k * (T_frame + k * RTT_MS)
                    p_still_failing *= eps_table[gi, k]
                avg_lat += p_still_failing * (T_frame + K * RTT_MS)

                # Apply to all eps targets this config satisfies
                for i in range(n_eps_local):
                    if p_all_fail <= eps_arr[i] and avg_lat < lat_out[i]:
                        lat_out[i] = avg_lat

        return lat_out

    k_c = DT_CONFIG.k_c
    n_s = DT_CONFIG.n_s
    d_min = DT_CONFIG.d_min
    K_rice = 2.0

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                              spread_deg=30.0)
    p_block_sub = p_block_3gpp(elev_angles)

    eps_sys_targets = np.logspace(-5, np.log10(0.5), 80)
    n_eps = len(eps_sys_targets)

    print(f"Latency-reliability trade-off (MC={n_mc}, OTFS channel)")
    print(f"  R_c={R_c:.2f}, n_s={n_s}, k_c={k_c}, d_min={d_min}")
    print(f"  P_total values: {P_total_dBW_values} dBW")
    print(f"  ref_elevation={ref_elevation_deg} deg, spread=30 deg")
    print(f"  K_rice={K_rice:.1f}, shadow_loss={shadow_loss_dB} dB")
    print(f"  RTT={RTT_MS} ms, max_retx={MAX_HARQ_RETX}")
    print(f"  HARQ-CC with chase combining gain")

    results = {
        'eps_sys_targets': eps_sys_targets,
        'P_total_dBW_values': np.array(P_total_dBW_values, dtype=float),
    }

    for s_idx, P_total_dBW in enumerate(P_total_dBW_values):
        gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

        lat_div_all = np.zeros((n_mc, n_eps))
        lat_harq_all = np.zeros((n_mc, n_eps))
        lat_mrc_all = np.zeros((n_mc, n_eps))
        lat_sc_all = np.zeros((n_mc, n_eps))

        print(f"\n  P_total={P_total_dBW:.0f} dBW:")

        for mc in range(n_mc):
            rng = np.random.default_rng(42 + mc)

            h_dd, noise_var, gamma_e2e, blocked_mask = generate_dd_channel(
                gamma_su, gamma_gs, n_subchannels, K_rice,
                p_block_sub, shadow_loss_dB, rng)

            non_blocked = gamma_e2e[~blocked_mask]
            if len(non_blocked) == 0:
                lat_div_all[mc, :] = np.inf
                lat_harq_all[mc, :] = np.inf
                lat_mrc_all[mc, :] = np.inf
                lat_sc_all[mc, :] = np.inf
                continue

            gamma = np.mean(non_blocked)
            gamma_mrc = np.sum(non_blocked)
            gamma_sc = np.max(non_blocked)

            # Diversity transform (single shot, relaxed target)
            for i, eps_sys in enumerate(eps_sys_targets):
                eps_star = _epsilon_star(eps_sys, n_s, d_min)
                n_c_div = _min_blocklength(eps_star, gamma, R_c)
                lat_div_all[mc, i] = _frame_duration_ms(n_c_div)

            # MRC: single shot, combined SNR
            # MRC carries K = n_c*R_c per frame (repeated codeword),
            # while DT carries K = k_c*n_c*R_c. Scale by k_c for same K.
            for i, eps_sys in enumerate(eps_sys_targets):
                n_c_mrc = _min_blocklength(eps_sys, gamma_mrc, R_c)
                lat_mrc_all[mc, i] = k_c * _frame_duration_ms(n_c_mrc)

            # SC: single shot, best subchannel SNR (same K scaling)
            for i, eps_sys in enumerate(eps_sys_targets):
                n_c_sc = _min_blocklength(eps_sys, gamma_sc, R_c)
                lat_sc_all[mc, i] = k_c * _frame_duration_ms(n_c_sc)

            # HARQ-CC with combining gain (batch all eps targets)
            lat_harq_all[mc, :] = _harqcc_batch(eps_sys_targets, gamma, R_c)

            if (mc + 1) % 500 == 0:
                print(f"    MC trial {mc+1}/{n_mc}")

        # Geometric mean (exp of mean of log) — smooth on log-scale
        for tag, arr in [('div', lat_div_all), ('harq', lat_harq_all),
                         ('mrc', lat_mrc_all), ('sc', lat_sc_all)]:
            geo = np.zeros(n_eps)
            for i in range(n_eps):
                col = arr[:, i]
                valid = col[np.isfinite(col) & (col > 0)]
                if len(valid) > 0:
                    geo[i] = np.exp(np.mean(np.log(valid)))
                else:
                    geo[i] = np.inf
            results[f'latency_{tag}_S{s_idx}'] = geo

        for eps_t in [1e-1, 1e-2, 1e-3, 1e-4]:
            idx = np.argmin(np.abs(eps_sys_targets - eps_t))
            print(f"    eps_sys={eps_t:.0e}: "
                  f"DIV={results[f'latency_div_S{s_idx}'][idx]:.4f}ms | "
                  f"MRC={results[f'latency_mrc_S{s_idx}'][idx]:.4f}ms | "
                  f"SC={results[f'latency_sc_S{s_idx}'][idx]:.4f}ms | "
                  f"HARQ-CC={results[f'latency_harq_S{s_idx}'][idx]:.4f}ms")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    eps = results['eps_sys_targets']
    P_values = results['P_total_dBW_values']

    scheme_cfg = {
        'div':  {'color': '#ff7f0e', 'marker': 'o', 'lw': 1.8, 'ms': 5.25},
        'mrc':  {'color': '#1f77b4', 'marker': 's', 'lw': 1.5, 'ms': 4.5},
        'sc':   {'color': '#2ca02c', 'marker': '^', 'lw': 1.5, 'ms': 4.5},
        'harq': {'color': '#d62728', 'marker': 'x', 'lw': 1.5, 'ms': 5.25},
    }
    scheme_labels = {
        'div':  r'$\mathbf{{G}}_{{\mathrm{{DIV}}}}$',
        'mrc':  'MRC',
        'sc':   'SC',
        'harq': 'HARQ-CC',
    }
    power_ls = ['-', '--']
    power_mfc = ['fill', 'none']
    marker_every = max(1, len(eps) // 10)

    for s_idx, P_dBW in enumerate(P_values):
        P_label = rf'{P_dBW:.0f}'
        ls = power_ls[s_idx]

        for tag in ['div', 'mrc', 'sc', 'harq']:
            sc = scheme_cfg[tag]
            mfc = sc['color'] if power_mfc[s_idx] == 'fill' else 'none'
            ax.plot(eps, results[f'latency_{tag}_S{s_idx}'],
                    ls, color=sc['color'], lw=sc['lw'],
                    marker=sc['marker'], ms=sc['ms'], mew=0.8,
                    mfc=mfc, markevery=marker_every,
                    label=rf'{scheme_labels[tag]}, $P\!=\!{P_label}$ dBW')

    ax.set_xlabel(r'Target system BLER $\epsilon_{\mathrm{sys}}^{\mathrm{target}}$')
    ax.set_ylabel('Transmission latency (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.legend(fontsize=4.5, loc='lower right', framealpha=1.0,
              ncol=2, handletextpad=0.4, handlelength=2.0,
              borderpad=0.4, labelspacing=0.25, columnspacing=0.8)
    ax.grid(True, which='both', ls='--', alpha=0.3)

    save_figure(fig, 'latency_comparison')


def save_csv(results):
    eps = results['eps_sys_targets']
    P_values = results['P_total_dBW_values']

    with open('latency_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f)
        header = ['eps_sys_target']
        for s_idx, P_dBW in enumerate(P_values):
            header += [f'latency_div_{P_dBW:.0f}dBW',
                       f'latency_mrc_{P_dBW:.0f}dBW',
                       f'latency_sc_{P_dBW:.0f}dBW',
                       f'latency_harq_{P_dBW:.0f}dBW']
        w.writerow(header)

        for i in range(len(eps)):
            row = [f"{eps[i]:.6e}"]
            for s_idx in range(len(P_values)):
                row.append(f"{results[f'latency_div_S{s_idx}'][i]:.6e}")
                row.append(f"{results[f'latency_mrc_S{s_idx}'][i]:.6e}")
                row.append(f"{results[f'latency_sc_S{s_idx}'][i]:.6e}")
                row.append(f"{results[f'latency_harq_S{s_idx}'][i]:.6e}")
            w.writerow(row)

    print("Saved: latency_comparison.csv")


if __name__ == "__main__":
    results = simulate(n_mc=1000)
    plot(results)
    save_csv(results)
