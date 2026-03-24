"""Regenerate all figures from existing CSV data (no simulation)."""
import csv
import numpy as np


def load_csv(filename):
    """Load CSV into a dict of numpy arrays keyed by column header."""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        cols = {h: [] for h in headers}
        for row in reader:
            for h, v in zip(headers, row):
                cols[h].append(float(v))
    return {h: np.array(v) for h, v in cols.items()}


if __name__ == "__main__":
    # 1. BLER vs Power
    from sim_bler_vs_power import plot as plot_power
    print("Regenerating: bler_comparison")
    plot_power(load_csv('bler_vs_power.csv'))

    # 2. BLER vs Elevation
    from sim_bler_vs_elevation import plot as plot_elev
    print("Regenerating: bler_vs_elevation_comparison")
    plot_elev(load_csv('bler_vs_elevation.csv'))

    # 3. BLER vs Blocklength
    from sim_bler_vs_blocklength import plot as plot_bl
    print("Regenerating: bler_vs_blocklength")
    plot_bl(load_csv('bler_vs_blocklength.csv'))

    # 4. Throughput
    from sim_throughput import plot as plot_tp
    print("Regenerating: throughput_reliability")
    plot_tp(load_csv('throughput_reliability.csv'))

    # 5. OTFS vs OFDM
    from sim_otfs_vs_ofdm import plot as plot_otfs
    print("Regenerating: otfs_vs_ofdm")
    plot_otfs(load_csv('otfs_vs_ofdm.csv'))

    # 6. BLER vs Blockage
    from sim_bler_vs_blockage import plot as plot_blockage
    print("Regenerating: bler_vs_blockage")
    plot_blockage(load_csv('bler_vs_blockage.csv'))

    # 7. BLER vs N subchannels
    from sim_bler_vs_nsubchannels import plot as plot_nsub
    print("Regenerating: bler_vs_n_subchannels")
    plot_nsub(load_csv('bler_vs_n_subchannels.csv'))

    # 8. Prop 1: Closed-form validation
    from sim_prop1_closedform import plot as plot_prop1
    print("Regenerating: prop1_closedform_validation")
    r1 = load_csv('prop1_closedform_validation.csv')
    r1['blocklengths'] = [64, 128, 256, 1024]
    plot_prop1(r1)

    # 9. Prop 2: Diversity order
    from sim_prop2_diversity_order import plot as plot_prop2
    print("Regenerating: prop2_diversity_order")
    r2 = load_csv('prop2_diversity_order.csv')
    r2_mc = load_csv('prop2_diversity_order_mc.csv')
    r2['snr_dB_mc'] = r2_mc['snr_dB']
    for L in [1, 2, 3, 4]:
        mc_key = f'mc_L{L}'
        if mc_key in r2_mc:
            r2[mc_key] = r2_mc[mc_key]
    plot_prop2(r2)

    print("\nAll figures regenerated.")
