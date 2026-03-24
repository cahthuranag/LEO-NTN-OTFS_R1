"""
BLER vs Number of Subchannels n_s — 5-curve comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, M_GF, OTFS_PARAMS,
    slant_range_m, access_fspl_dB, atmospheric_loss_dB, power_to_link_snrs,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    per_subchannel_eta_ephemeris, p_block_3gpp,
    create_ephemeris_predictor, mc_trial_variable_ns,
    build_subchannel_map,
    db_to_linear, linear_to_db,
    gf2_rank, generate_candidate_T, gf2_matmul,
)
from polar_code import PolarCode
from diversity_transform import DiversityTransformConfig


def simulate(N=256, K=128, ns_values=None,
             P_total_dBW=3.0, ref_elevation_deg=50.0, n_mc=1500):
    if ns_values is None:
        ns_values = [3, 4, 5, 6, 7]
    m = 3

    rng = np.random.default_rng(2031)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    epsilon_th = 0.1
    shadow_loss_dB = 20.0
    R = K / N

    n_pts = len(ns_values)
    results = {key: np.zeros(n_pts) for key in [
        'ppv_no_diversity', 'ppv_fixed',
        'no_interleaver', 'interleaver', 'fixed']}
    results['n_subchannels'] = np.array(ns_values)

    print(f"BLER vs n_subchannels (5 curves, P_total={P_total_dBW} dBW, MC={n_mc})")

    for idx, n_s_val in enumerate(ns_values):
        k_c_val = 2
        if k_c_val >= n_s_val:
            k_c_val = n_s_val - 1

        dt_config = DiversityTransformConfig(k_c=k_c_val, n_s=n_s_val, m=m)
        rho = dt_config.rho
        n_out = dt_config.n_out
        G_FIX_local = dt_config.G_FIX

        N_CW_local = 1 << rho
        X_ALL_local = np.zeros((rho, N_CW_local), dtype=np.int8)
        for i_cw in range(N_CW_local):
            for b in range(rho):
                X_ALL_local[b, i_cw] = (i_cw >> b) & 1

        Z_ALL_FIX_local = np.mod(
            G_FIX_local.T.astype(int) @ X_ALL_local.astype(int), 2
        ).astype(np.int8)
        S_ALL_FIX_local = 1.0 - 2.0 * Z_ALL_FIX_local.astype(np.float64)

        X_MASK_0_local = [X_ALL_local[i, :] == 0 for i in range(rho)]
        X_MASK_1_local = [X_ALL_local[i, :] == 1 for i in range(rho)]

        m_sub = OTFS_PARAMS.M_D // n_s_val
        sc_map = build_subchannel_map(n_s=n_s_val, m_sub=m_sub)
        n_sym = {j: len(sc_map[j]) for j in range(n_s_val)}

        polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

        # Per-subchannel Polar code
        N_sub = N // n_s_val
        K_sub = K // n_s_val
        N_sub_p2 = 1 << (N_sub.bit_length() - 1) if N_sub > 0 else 1
        K_sub_p2 = max(1, int(K_sub * N_sub_p2 / max(N_sub, 1)))
        K_sub_p2 = min(K_sub_p2, N_sub_p2 - 1)
        polar_sub = PolarCode(N_sub_p2, K_sub_p2, design_snr_dB=1.0, list_size=8)

        elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_s_val,
                                                          spread_deg=30.0)
        gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)
        p_block_sub = p_block_3gpp(elev_angles)
        eta_ephemeris = per_subchannel_eta_ephemeris(
            elev_angles, P_total_dBW, K_rice_dB, R, p_block_per_sub=p_block_sub)

        ppv_no_acc, ppv_fix_acc = 0.0, 0.0
        err_no_intlv_count, err_intlv_count, err_fix_count = 0, 0, 0

        print(f"  n_s={n_s_val}, k_c={k_c_val}, rho={rho}, "
              f"max_erasures={dt_config.max_erasures}, "
              f"Polar_sub({N_sub_p2},{K_sub_p2})...")

        for _ in range(n_mc):
            res = mc_trial_variable_ns(
                gamma_su, gamma_gs, N, K, dt_config,
                G_FIX_local, X_ALL_local, S_ALL_FIX_local,
                X_MASK_0_local, X_MASK_1_local,
                sc_map, n_sym,
                K_rice, epsilon_th, p_block_sub,
                shadow_loss_dB, polar_code, rng,
                eta_ephemeris=eta_ephemeris,
                polar_sub=polar_sub)
            ppv_no_acc += res['ppv_no_div']
            ppv_fix_acc += res['ppv_fix']
            err_no_intlv_count += int(res['err_no_intlv'])
            err_intlv_count += int(res['err_intlv'])
            err_fix_count += int(res['err_fix'])

        results['ppv_no_diversity'][idx] = ppv_no_acc / n_mc
        results['ppv_fixed'][idx] = ppv_fix_acc / n_mc
        results['no_interleaver'][idx] = err_no_intlv_count / n_mc
        results['interleaver'][idx] = err_intlv_count / n_mc
        results['fixed'][idx] = err_fix_count / n_mc

        print(f"    PPV_no={results['ppv_no_diversity'][idx]:.3e}, "
              f"NoIntlv={results['no_interleaver'][idx]:.3e}, "
              f"Intlv={results['interleaver'][idx]:.3e}, "
              f"Fix={results['fixed'][idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ns = results['n_subchannels']

    ax.semilogy(ns, np.maximum(results['ppv_no_diversity'], 1e-8),
                'h--', color='#9467bd', lw=0.8, ms=4,
                label=r'PPV bound (no diversity)')
    ax.semilogy(ns, np.maximum(results['ppv_fixed'], 1e-8),
                'p--', color='#17becf', lw=0.8, ms=4,
                label=r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    ax.semilogy(ns, np.maximum(results['no_interleaver'], 1e-8),
                'v-', color='#7f7f7f', lw=1.0, ms=4,
                label=r'Polar (no interleaver)')
    ax.semilogy(ns, np.maximum(results['interleaver'], 1e-8),
                '^-', color='#d62728', lw=1.0, ms=4,
                label=r'Polar + interleaver')
    ax.semilogy(ns, np.maximum(results['fixed'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=5,
                label=r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Number of subchannels $n_s$')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.legend(fontsize=6, loc='lower left', framealpha=1.0)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xticks(ns)

    save_figure(fig, 'bler_vs_n_subchannels')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed',
            'no_interleaver', 'interleaver', 'fixed']
    with open('bler_vs_n_subchannels.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_subchannels'] + keys)
        for i in range(len(results['n_subchannels'])):
            row = [str(int(results['n_subchannels'][i]))]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_n_subchannels.csv")


if __name__ == "__main__":
    results = simulate(n_mc=10000)
    plot(results)
    save_csv(results)
