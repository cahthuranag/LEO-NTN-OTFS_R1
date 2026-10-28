"""
FEC comparison: Polar vs LDPC with proposed diversity transform.

Demonstrates FEC-agnostic operation at three code rates (R≈1/4, 1/2, 3/4).
No-diversity baselines: per-subchannel Polar (independent on each subchannel,
system fails if any subchannel fails) — matching Fig. 5 definition.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
from functools import partial
from pyldpc import make_ldpc, encode as ldpc_encode_bpsk, decode as ldpc_decode_bp, get_message

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, G_FIX, RHO, N_OUT, M_GF,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    generate_dd_channel, conditional_bler,
    diversity_transform_transmit_receive,
    qpsk_modulate, qpsk_soft_demod,
)
from polar_code import PolarCode


class LDPCCode:
    def __init__(self, n=252, d_v=3, d_c=6, max_iter=50):
        self.H, self.G = make_ldpc(n, d_v, d_c, systematic=True,
                                   sparse=True)
        self.N = n
        self.K = self.G.shape[1]
        self.max_iter = max_iter

    def encode(self, info_bits):
        coded_bpsk = ldpc_encode_bpsk(self.G, info_bits, snr=100)
        return (coded_bpsk < 0).astype(np.int8)

    def decode_llr(self, llr):
        mean_abs = np.mean(np.abs(llr))
        snr_est_linear = max(mean_abs / 2, 0.1)
        snr_est_dB = 10 * np.log10(snr_est_linear)
        var_est = 1.0 / snr_est_linear
        y_est = llr * var_est / 2
        decoded = ldpc_decode_bp(self.H, y_est, snr_est_dB,
                                 maxiter=self.max_iter)
        return get_message(self.G, decoded)


N_DT_PAD = 288


def _dt_trial(code, info_bits, gamma_e2e, erased_mask,
              eta_inst, max_erasures, n_erased, rng, code_N):
    if n_erased > max_erasures:
        return True
    coded_raw = code.encode(info_bits)
    coded = np.zeros(N_DT_PAD, dtype=np.int8)
    coded[:len(coded_raw)] = coded_raw
    llr = diversity_transform_transmit_receive(
        coded, G_FIX, gamma_e2e, erased_mask, eta_inst, rng)
    if llr is None:
        return True
    if hasattr(code, 'decode_llr'):
        dec = code.decode_llr(llr[:code_N])
    else:
        dec = code.decode(llr[:code_N])
    return not np.array_equal(dec, info_bits)


def _nodiv_subchannel_trial(polar_sub, gamma_e2e, n_subchannels, rng):
    """Per-subchannel independent Polar, system fails if ANY subchannel fails.
    Matches Fig. 5 baseline definition."""
    K_sub = polar_sub.K
    N_sub = polar_sub.N
    for j in range(n_subchannels):
        sub_info = rng.integers(0, 2, size=K_sub).astype(np.int8)
        coded_j = polar_sub.encode(sub_info)
        s = 1.0 - 2.0 * coded_j.astype(np.float64)
        gamma_j = max(gamma_e2e[j], 1e-10)
        sigma2 = 1.0 / (2.0 * gamma_j)
        noise = rng.normal(0, np.sqrt(sigma2), N_sub)
        y = s + noise
        llr_j = 4.0 * gamma_j * y
        dec = polar_sub.decode(llr_j)
        if not np.array_equal(dec, sub_info):
            return True
    return False


def _worker(args):
    """Worker for one power point. Returns dict of error counts."""
    (p_idx, P_dBW, n_mc, seed, elev_angles, p_block, K_rice,
     shadow_loss_dB, epsilon_th, n_subchannels, max_erasures) = args

    rng = np.random.default_rng(seed)

    # Build codes inside worker (not picklable across processes)
    polar_r14 = PolarCode(256, 64, design_snr_dB=1.0, list_size=32)
    polar_r12 = PolarCode(256, 128, design_snr_dB=1.0, list_size=32)
    polar_r34 = PolarCode(256, 192, design_snr_dB=1.0, list_size=32)
    ldpc_r14 = LDPCCode(n=252, d_v=3, d_c=4)
    ldpc_r12 = LDPCCode(n=252, d_v=3, d_c=6)
    ldpc_r34 = LDPCCode(n=252, d_v=3, d_c=12)

    schemes = [
        ('polar_r14', polar_r14, 64,  256),
        ('polar_r12', polar_r12, 128, 256),
        ('polar_r34', polar_r34, 192, 256),
        ('ldpc_r14',  ldpc_r14,  ldpc_r14.K, ldpc_r14.N),
        ('ldpc_r12',  ldpc_r12,  ldpc_r12.K, ldpc_r12.N),
        ('ldpc_r34',  ldpc_r34,  ldpc_r34.K, ldpc_r34.N),
    ]

    # No-diversity per-subchannel polar codes (matching Fig. 5)
    # N_sub = N // n_subchannels, round up to power of 2
    N_sub_raw = 256 // n_subchannels
    N_sub_p2 = 1
    while N_sub_p2 < N_sub_raw:
        N_sub_p2 *= 2
    K_sub_r14 = max(N_sub_p2 // 4, 2)
    K_sub_r12 = max(N_sub_p2 // 2, 2)
    K_sub_r34 = max(3 * N_sub_p2 // 4, 2)
    polar_sub_r14 = PolarCode(N_sub_p2, K_sub_r14, design_snr_dB=1.0, list_size=8)
    polar_sub_r12 = PolarCode(N_sub_p2, K_sub_r12, design_snr_dB=1.0, list_size=8)
    polar_sub_r34 = PolarCode(N_sub_p2, K_sub_r34, design_snr_dB=1.0, list_size=8)

    N_sub_polar = 256 // n_subchannels
    R_polar = 128 / 256

    gamma_gs, gamma_su = per_subchannel_link_snrs(P_dBW, elev_angles)

    err = {name: 0 for name, _, _, _ in schemes}
    err['nodiv_r14'] = 0
    err['nodiv_r12'] = 0
    err['nodiv_r34'] = 0

    for _ in range(n_mc):
        _, _, gamma_e2e, blocked_mask = generate_dd_channel(
            gamma_su, gamma_gs, n_subchannels, K_rice,
            p_block, shadow_loss_dB, rng)

        erased_mask = np.array([
            conditional_bler(gamma_e2e[j], N_sub_polar, R_polar)
            > epsilon_th
            for j in range(n_subchannels)
        ])
        n_erased = int(np.sum(erased_mask))

        eta_inst = np.zeros(n_subchannels)
        for j in range(n_subchannels):
            if erased_mask[j]:
                eta_inst[j] = 0.0
            else:
                eta_inst[j] = 1.0 - conditional_bler(
                    gamma_e2e[j], N_sub_polar, R_polar)

        for name, code, K, N in schemes:
            info = rng.integers(0, 2, size=K).astype(np.int8)
            if _dt_trial(code, info, gamma_e2e, erased_mask,
                         eta_inst, max_erasures, n_erased, rng, N):
                err[name] += 1

        # No diversity: per-subchannel independent Polar, fail if ANY fails
        if _nodiv_subchannel_trial(polar_sub_r14, gamma_e2e, n_subchannels, rng):
            err['nodiv_r14'] += 1
        if _nodiv_subchannel_trial(polar_sub_r12, gamma_e2e, n_subchannels, rng):
            err['nodiv_r12'] += 1
        if _nodiv_subchannel_trial(polar_sub_r34, gamma_e2e, n_subchannels, rng):
            err['nodiv_r34'] += 1

    return p_idx, {k: v / n_mc for k, v in err.items()}


def simulate(P_range_dBW=None, ref_elevation_deg=50.0,
             p_block=0.18, n_mc=1000, n_workers=None):
    if P_range_dBW is None:
        P_range_dBW = np.arange(5, 26, 3)
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(P_range_dBW))

    n_subchannels = N_S
    max_erasures = DT_CONFIG.max_erasures
    K_rice = 2.0
    epsilon_th = 0.1
    shadow_loss_dB = 20.0

    elev_angles = subchannel_elevation_angles(ref_elevation_deg,
                                              n_subchannels, spread_deg=30.0)

    print(f"FEC comparison (p_block={p_block}, MC={n_mc}, workers={n_workers})")

    base_seed = 2030
    worker_args = []
    for p_idx, P_dBW in enumerate(P_range_dBW):
        worker_args.append((
            p_idx, float(P_dBW), n_mc, base_seed + p_idx * 1000,
            elev_angles, p_block, K_rice, shadow_loss_dB, epsilon_th,
            n_subchannels, max_erasures
        ))

    n_P = len(P_range_dBW)
    scheme_names = ['polar_r14', 'polar_r12', 'polar_r34',
                    'ldpc_r14', 'ldpc_r12', 'ldpc_r34']
    results = {'P_dBW': P_range_dBW}
    for name in scheme_names:
        results[name] = np.zeros(n_P)
    results['nodiv_r14'] = np.zeros(n_P)
    results['nodiv_r12'] = np.zeros(n_P)
    results['nodiv_r34'] = np.zeros(n_P)

    with mp.Pool(n_workers) as pool:
        for p_idx, bler_dict in pool.imap_unordered(_worker, worker_args):
            P_val = P_range_dBW[p_idx]
            for k, v in bler_dict.items():
                results[k][p_idx] = v
            vals = ' | '.join(f"{name}={results[name][p_idx]:.3e}"
                              for name in scheme_names)
            print(f"  P={P_val}: {vals}")
            print(f"         NoDiv_r14={results['nodiv_r14'][p_idx]:.3e} | "
                  f"NoDiv_r12={results['nodiv_r12'][p_idx]:.3e} | "
                  f"NoDiv_r34={results['nodiv_r34'][p_idx]:.3e}")

    return results


def plot(results):
    ieee_setup()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    P = results['P_dBW']

    # No diversity baselines (per-subchannel, matching Fig. 5)
    ax.semilogy(P, np.maximum(results['nodiv_r34'], 1e-8),
                '+--', color='#8c564b', lw=1.0, ms=4,
                label=r'Polar $(R_c\!=\!3/4)$, no div.')
    ax.semilogy(P, np.maximum(results['nodiv_r12'], 1e-8),
                'v--', color='#7f7f7f', lw=1.0, ms=4,
                label=r'Polar $(R_c\!=\!1/2)$, no div.')
    ax.semilogy(P, np.maximum(results['nodiv_r14'], 1e-8),
                'h--', color='#bcbd22', lw=1.0, ms=4,
                label=r'Polar $(R_c\!=\!1/4)$, no div.')

    # R=1/4 + G_DIV
    ax.semilogy(P, np.maximum(results['polar_r14'], 1e-8),
                'o-', color='#2ca02c', lw=1.0, ms=4,
                label=r'Polar $(R_c\!=\!1/4)$ + $\mathbf{G}_{\mathrm{DIV}}$')
    ax.semilogy(P, np.maximum(results['ldpc_r14'], 1e-8),
                's-', color='#17becf', lw=1.0, ms=4,
                label=r'LDPC $(R_c\!\approx\!1/4)$ + $\mathbf{G}_{\mathrm{DIV}}$')

    # R=1/2 + G_DIV
    ax.semilogy(P, np.maximum(results['polar_r12'], 1e-8),
                'D-', color='#ff7f0e', lw=1.2, ms=4,
                label=r'Polar $(R_c\!=\!1/2)$ + $\mathbf{G}_{\mathrm{DIV}}$')
    ax.semilogy(P, np.maximum(results['ldpc_r12'], 1e-8),
                '^-', color='#1f77b4', lw=1.0, ms=4,
                label=r'LDPC $(R_c\!\approx\!1/2)$ + $\mathbf{G}_{\mathrm{DIV}}$')

    # R=3/4 + G_DIV
    ax.semilogy(P, np.maximum(results['polar_r34'], 1e-8),
                'p-', color='#d62728', lw=1.0, ms=4,
                label=r'Polar $(R_c\!=\!3/4)$ + $\mathbf{G}_{\mathrm{DIV}}$')
    ax.semilogy(P, np.maximum(results['ldpc_r34'], 1e-8),
                'x-', color='#9467bd', lw=1.0, ms=4,
                label=r'LDPC $(R_c\!\approx\!3/4)$ + $\mathbf{G}_{\mathrm{DIV}}$')

    ax.set_xlabel(r'Total transmit power $P_{\mathrm{total}}$ (dBW)')
    ax.set_ylabel(r'System BLER $\epsilon_{\mathrm{sys}}$')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlim(4.5, 24)
    ax.set_ylim(5e-3, 1.1)
    ax.legend(fontsize=5.5, loc='lower left', ncol=2, framealpha=1.0,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.5,
              borderpad=0.4, labelspacing=0.3)

    fig.tight_layout()
    save_figure(fig, 'fec_comparison')


def save_csv(results):
    fname = 'fec_comparison.csv'
    keys = [k for k in results if k != 'P_dBW']
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_dBW'] + keys)
        for i in range(len(results['P_dBW'])):
            row = [f"{results['P_dBW'][i]:.1f}"]
            for k in keys:
                row.append(f"{results[k][i]:.6e}")
            w.writerow(row)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    results = simulate(n_mc=5000)
    save_csv(results)
    plot(results)
