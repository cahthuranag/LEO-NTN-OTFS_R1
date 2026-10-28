"""
BLER vs Blockage Correlation (rho_blk) at fixed blockage probability.

X-axis: rho_blk from 0 to 1
Y-axis: System BLER for each scheme
Fixed p_block = 0.2 (mid-range operating point from Fig. 6)
"""
import numpy as np
import csv

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    per_subchannel_eta_ephemeris, mc_trial_one_snr,
    generate_dd_channel, correlated_blockage_mask,
)
from fbl_analysis import (
    conditional_bler, bler_no_diversity, bler_fixed_diversity,
)
from polar_code import PolarCode


def simulate(rho_values, N=256, K=128, n_subchannels=N_S,
             max_erasures=DT_CONFIG.max_erasures,
             P_total_dBW=8.0, ref_elevation_deg=50.0,
             p_block=0.2, n_mc=10000, n_mc_ppv=50000):
    rng = np.random.default_rng(2028)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                              spread_deg=30.0)
    gamma_gs, gamma_su = per_subchannel_link_snrs(P_total_dBW, elev_angles)

    K_rice = 2.0
    K_rice_dB = 10.0 * np.log10(K_rice)
    epsilon_th = 0.1
    shadow_loss_dB = 20.0
    R = K / N
    k_c = DT_CONFIG.k_c
    N_sub = N // n_subchannels

    polar_code = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    p_blk_per_sub = np.full(n_subchannels, p_block)
    eta_ephemeris = per_subchannel_eta_ephemeris(
        elev_angles, P_total_dBW, K_rice_dB, R,
        p_block_per_sub=p_blk_per_sub)

    results = {key: np.zeros(len(rho_values)) for key in [
        'ppv_no_diversity', 'ppv_fixed_diversity',
        'no_interleaver', 'interleaver', 'fixed']}
    results['rho_blk'] = np.array(rho_values)

    print(f"BLER vs rho_blk (p_block={p_block}, P_total={P_total_dBW} dBW, "
          f"MC_polar={n_mc}, MC_ppv={n_mc_ppv})")

    for idx, rho in enumerate(rho_values):
        # --- PPV bounds: channel-only MC with more samples (fast) ---
        ppv_rng = np.random.default_rng(2028 + idx * 7919)
        ppv_no_acc, ppv_fix_acc = 0.0, 0.0
        for _ in range(n_mc_ppv):
            _, _, gamma_e2e, _ = generate_dd_channel(
                gamma_su, gamma_gs, n_subchannels, K_rice,
                p_block, shadow_loss_dB, ppv_rng, rho_blk=rho)
            erased_mask = np.array([
                conditional_bler(gamma_e2e[j], N_sub, R) > epsilon_th
                for j in range(n_subchannels)])
            ppv_no_acc += bler_no_diversity(gamma_e2e, erased_mask, k_c, N, K)
            ppv_fix_acc += bler_fixed_diversity(gamma_e2e, erased_mask, k_c, N, K)
        results['ppv_no_diversity'][idx] = ppv_no_acc / n_mc_ppv
        results['ppv_fixed_diversity'][idx] = ppv_fix_acc / n_mc_ppv

        # --- MC Polar curves: full pipeline ---
        err_counts = [0, 0, 0]
        for _ in range(n_mc):
            res = mc_trial_one_snr(
                gamma_su, gamma_gs, N, K, n_subchannels, max_erasures,
                K_rice, epsilon_th, p_block, shadow_loss_dB,
                polar_code, rng, use_otfs=True, eta_ephemeris=eta_ephemeris,
                rho_blk=rho)
            err_counts[0] += int(res['err_1'])
            err_counts[1] += int(res['err_2'])
            err_counts[2] += int(res['err_3'])

        results['no_interleaver'][idx] = err_counts[0] / n_mc
        results['interleaver'][idx] = err_counts[1] / n_mc
        results['fixed'][idx] = err_counts[2] / n_mc

        print(f"  rho={rho:.2f}: PPV_no={results['ppv_no_diversity'][idx]:.3e}, "
              f"PPV_fix={results['ppv_fixed_diversity'][idx]:.3e}, "
              f"NoIntlv={results['no_interleaver'][idx]:.3e}, "
              f"Intlv={results['interleaver'][idx]:.3e}, "
              f"DivTx={results['fixed'][idx]:.3e}")

    return results


def plot(results):
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    rho = results['rho_blk']
    rho_smooth = np.linspace(rho[0], rho[-1], 200)

    def smooth_semilogy(ax, x, y, marker, color, lw, ms, label, linestyle='--'):
        y_safe = np.maximum(y, 1e-8)
        log_y = np.log10(y_safe)
        spl = make_interp_spline(x, log_y, k=3)
        log_y_smooth = spl(rho_smooth)
        ax.semilogy(rho_smooth, 10**log_y_smooth, linestyle, color=color, lw=lw)
        ax.semilogy(x, y_safe, marker, color=color, ms=ms, label=label,
                    linestyle='none')

    smooth_semilogy(ax, rho, results['ppv_no_diversity'],
                    'h', '#9467bd', 1.0, 4, r'PPV bound (no diversity)')
    smooth_semilogy(ax, rho, results['ppv_fixed_diversity'],
                    'p', '#17becf', 1.0, 4, r'PPV ($\mathbf{G}_{\mathrm{DIV}}$)')
    smooth_semilogy(ax, rho, results['no_interleaver'],
                    'v', '#7f7f7f', 1.0, 4, r'Polar (no interleaver)', '-')
    smooth_semilogy(ax, rho, results['interleaver'],
                    '^', '#d62728', 1.0, 4, r'Polar + interleaver', '-')
    smooth_semilogy(ax, rho, results['fixed'],
                    'D', '#ff7f0e', 1.2, 4,
                    r'Proposed Polar + $\mathbf{G}_{\mathrm{DIV}}$', '-')

    ax.set_xlabel(r'Blockage correlation $\rho_{\mathrm{blk}}$')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.legend(fontsize=5.5, loc='lower left', framealpha=1.0, ncol=2,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.5)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([1e-1, 1.1])

    fig.tight_layout()
    save_figure(fig, 'bler_vs_correlated_blockage')


def save_csv(results):
    keys = ['ppv_no_diversity', 'ppv_fixed_diversity',
            'no_interleaver', 'interleaver', 'fixed']
    with open('bler_vs_correlated_blockage.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rho_blk'] + keys)
        for i in range(len(results['rho_blk'])):
            row = [f"{results['rho_blk'][i]:.2f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print("Saved: bler_vs_correlated_blockage.csv")


if __name__ == "__main__":
    rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = simulate(rho_values, n_mc=10000, n_mc_ppv=50000)
    plot(results)
    save_csv(results)
