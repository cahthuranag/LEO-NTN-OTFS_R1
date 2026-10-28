"""
Transmission latency vs target BLER: DT (single-shot) vs HARQ-CC.

DT:      Polar(256,128) + MDS diversity transform across 6 subchannels.
         Single-shot, latency = T_frame.
HARQ-CC: Polar(256,128) on single subchannel, Chase Combining up to Kmax
         retransmissions. Each retransmission adds T_RTT latency.

Same K=128, same total power, fixed blockage probability.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

from sim_common import (
    ieee_setup, save_figure, DT_CONFIG, N_S, G_FIX, RHO, N_OUT, M_GF,
    subchannel_elevation_angles, per_subchannel_link_snrs,
    generate_dd_channel, conditional_bler,
    diversity_transform_transmit_receive,
    qpsk_modulate, qpsk_soft_demod,
)
from polar_code import PolarCode


def simulate_latency(p_block=0.18, power_levels=None,
                     ref_elevation_deg=50.0, n_mc=1000,
                     Kmax=4, T_RTT_ms=20.0, T_frame_ms=1.0):
    if power_levels is None:
        power_levels = [10, 13, 16, 19]

    rng = np.random.default_rng(2030)
    n_subchannels = N_S
    max_erasures = DT_CONFIG.max_erasures

    N, K = 256, 128
    R = K / N
    polar = PolarCode(N, K, design_snr_dB=1.0, list_size=8)

    elev_angles = subchannel_elevation_angles(ref_elevation_deg, n_subchannels,
                                              spread_deg=30.0)
    N_sub_dt = N // n_subchannels
    j_harq = n_subchannels // 2

    K_rice = 2.0
    shadow_loss_dB = 20.0
    epsilon_th = 0.1

    print(f"Latency vs BLER (p_block={p_block}, MC={n_mc}, "
          f"Kmax={Kmax}, T_RTT={T_RTT_ms} ms)")

    all_results = {}

    for P_dBW in power_levels:
        gamma_gs, gamma_su = per_subchannel_link_snrs(P_dBW, elev_angles)

        err_dt = 0
        err_harq = np.zeros(Kmax, dtype=int)

        for _ in range(n_mc):
            _, _, gamma_e2e, blocked_mask = generate_dd_channel(
                gamma_su, gamma_gs, n_subchannels, K_rice,
                p_block, shadow_loss_dB, rng)

            # === DT: single-shot across 6 subchannels ===
            erased_mask = np.array([
                conditional_bler(gamma_e2e[j], N_sub_dt, R) > epsilon_th
                for j in range(n_subchannels)
            ])
            n_erased = int(np.sum(erased_mask))

            eta_inst = np.zeros(n_subchannels)
            for j in range(n_subchannels):
                if erased_mask[j]:
                    eta_inst[j] = 0.0
                else:
                    eta_inst[j] = 1.0 - conditional_bler(
                        gamma_e2e[j], N_sub_dt, R)

            info_dt = rng.integers(0, 2, size=K).astype(np.int8)
            if n_erased > max_erasures:
                err_dt += 1
            else:
                coded = polar.encode(info_dt)
                llr_dt = diversity_transform_transmit_receive(
                    coded, G_FIX, gamma_e2e, erased_mask, eta_inst, rng)
                if llr_dt is None:
                    err_dt += 1
                else:
                    decoded = polar.decode(llr_dt)
                    if not np.array_equal(decoded, info_dt):
                        err_dt += 1

            # === HARQ-CC: single subchannel, Chase Combining ===
            info_harq = rng.integers(0, 2, size=K).astype(np.int8)
            coded_harq = polar.encode(info_harq)
            qpsk_harq = qpsk_modulate(coded_harq)

            snr_harq = max(gamma_e2e[j_harq], 1e-10)
            if blocked_mask[j_harq]:
                snr_harq = 1e-10

            llr_combined = np.zeros(N, dtype=np.float64)
            first_success = Kmax

            for k in range(Kmax):
                noise_std = np.sqrt(0.5 / snr_harq)
                noise = noise_std * (rng.normal(0, 1, len(qpsk_harq))
                                     + 1j * rng.normal(0, 1, len(qpsk_harq)))
                y = qpsk_harq + noise
                llr_combined += qpsk_soft_demod(y, snr_harq)

                decoded_harq = polar.decode(llr_combined)
                if np.array_equal(decoded_harq, info_harq):
                    first_success = k
                    break

            for k in range(Kmax):
                if first_success > k:
                    err_harq[k] += 1

        bler_dt = err_dt / n_mc
        bler_harq = err_harq / n_mc
        lat_harq = [T_frame_ms + k * T_RTT_ms for k in range(Kmax)]

        all_results[P_dBW] = {
            'bler_dt': bler_dt,
            'bler_harq': bler_harq,
            'lat_dt': T_frame_ms,
            'lat_harq': lat_harq,
        }

        print(f"  P={P_dBW} dBW: DT={bler_dt:.3e}, "
              f"HARQ=[{', '.join(f'{b:.3e}' for b in bler_harq)}], "
              f"lat=[{', '.join(f'{t:.0f}' for t in lat_harq)}] ms")

    return all_results


def plot_latency_vs_bler(results, p_block, T_frame_ms=1.0):
    ieee_setup()
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    power_levels = list(results.keys())

    for i, P_dBW in enumerate(power_levels):
        res = results[P_dBW]

        # DT: single point
        if res['bler_dt'] > 0:
            ax.plot(res['bler_dt'], res['lat_dt'], 'D',
                    color=colors[i], ms=6, zorder=5,
                    label=rf'DT, $P\!=\!{P_dBW}$ dBW')

        # HARQ-CC: staircase
        bler_h = res['bler_harq']
        lat_h = res['lat_harq']
        valid = bler_h > 0
        if np.any(valid):
            ax.plot(bler_h[valid], np.array(lat_h)[valid],
                    'x--', color=colors[i], ms=5, lw=0.8,
                    label=rf'HARQ-CC, $P\!=\!{P_dBW}$ dBW')

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel(
        r'Target system BLER $\epsilon_{\mathrm{sys}}^{\mathrm{target}}$')
    ax.set_ylabel('Transmission latency [ms]')
    ax.set_title(
        rf'$p_{{\mathrm{{block}}}}={p_block}$, '
        rf'$K_{{max}}\!=\!4$, $T_{{\mathrm{{RTT}}}}\!=\!20$ ms',
        fontsize=7)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=5, loc='upper left', framealpha=1.0)

    fig.tight_layout()
    save_figure(fig, 'mrc_sc_comparison')


def save_latency_csv(results, p_block):
    fname = f'latency_vs_bler_pblock{p_block:.2f}.csv'
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_dBW', 'bler_dt', 'lat_dt_ms',
                     'bler_harq_k1', 'bler_harq_k2',
                     'bler_harq_k3', 'bler_harq_k4',
                     'lat_harq_k1', 'lat_harq_k2',
                     'lat_harq_k3', 'lat_harq_k4'])
        for P_dBW, res in results.items():
            row = [P_dBW, f"{res['bler_dt']:.6e}", f"{res['lat_dt']:.1f}"]
            for b in res['bler_harq']:
                row.append(f"{b:.6e}")
            for t in res['lat_harq']:
                row.append(f"{t:.1f}")
            w.writerow(row)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    p_block = 0.18
    power_levels = [10, 13, 16, 19]

    results = simulate_latency(p_block=p_block,
                               power_levels=power_levels,
                               n_mc=1000)
    save_latency_csv(results, p_block)
    plot_latency_vs_bler(results, p_block)
