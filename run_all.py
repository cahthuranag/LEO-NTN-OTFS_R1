"""
Run all simulations sequentially.

Each simulation can also be run independently:
    python3 sim_bler_vs_power.py
    python3 sim_bler_vs_elevation.py
    python3 sim_throughput.py
    python3 sim_bler_vs_blockage.py
    python3 sim_bler_vs_blocklength.py
    python3 sim_otfs_vs_ofdm.py
    python3 sim_bler_vs_nsubchannels.py
    python3 sim_prop1_closedform.py
    python3 sim_prop2_diversity_order.py
    python3 sim_latency.py
"""

import sim_bler_vs_power
import sim_bler_vs_elevation
import sim_throughput
import sim_bler_vs_blockage
import sim_bler_vs_blocklength
import sim_otfs_vs_ofdm
import sim_bler_vs_nsubchannels
import sim_prop1_closedform
import sim_prop2_diversity_order
import sim_latency


def main():
    print("=" * 70)
    print("Running all simulations")
    print("=" * 70)

    # 1. BLER vs Power
    print("\n" + "-" * 60)
    results_power = sim_bler_vs_power.simulate(n_mc=5000)
    sim_bler_vs_power.plot(results_power)
    sim_bler_vs_power.save_csv(results_power)

    # 2. BLER vs Elevation
    print("\n" + "-" * 60)
    results_elev = sim_bler_vs_elevation.simulate(n_mc=5000)
    sim_bler_vs_elevation.plot(results_elev)
    sim_bler_vs_elevation.save_csv(results_elev)

    # 3. Throughput-Reliability (uses power sweep results)
    print("\n" + "-" * 60)
    results_throughput = sim_throughput.simulate(results_power)
    sim_throughput.plot(results_throughput)
    sim_throughput.save_csv(results_throughput)

    # 4. BLER vs Blockage
    print("\n" + "-" * 60)
    results_blockage = sim_bler_vs_blockage.simulate(n_mc=5000)
    sim_bler_vs_blockage.plot(results_blockage)
    sim_bler_vs_blockage.save_csv(results_blockage)

    # 5. BLER vs Blocklength
    print("\n" + "-" * 60)
    results_bl = sim_bler_vs_blocklength.simulate(n_mc=5000)
    sim_bler_vs_blocklength.plot(results_bl)
    sim_bler_vs_blocklength.save_csv(results_bl)

    # 6. OTFS vs OFDM (high Doppler comparison)
    print("\n" + "-" * 60)
    results_otfs = sim_otfs_vs_ofdm.simulate(n_mc=5000)
    sim_otfs_vs_ofdm.plot(results_otfs)
    sim_otfs_vs_ofdm.save_csv(results_otfs)

    # 7. BLER vs Number of Subchannels
    print("\n" + "-" * 60)
    results_ns = sim_bler_vs_nsubchannels.simulate(n_mc=5000)
    sim_bler_vs_nsubchannels.plot(results_ns)
    sim_bler_vs_nsubchannels.save_csv(results_ns)

    # 8. Proposition 1: Closed-form BLER validation
    print("\n" + "-" * 60)
    results_p1 = sim_prop1_closedform.simulate(n_mc=5000)
    sim_prop1_closedform.plot(results_p1)
    sim_prop1_closedform.save_csv(results_p1)

    # 9. Proposition 2: Diversity order validation
    print("\n" + "-" * 60)
    results_p2 = sim_prop2_diversity_order.simulate()
    sim_prop2_diversity_order.plot(results_p2)
    sim_prop2_diversity_order.save_csv(results_p2)

    # 10. Latency comparison: Diversity Transform vs HARQ
    print("\n" + "-" * 60)
    results_latency = sim_latency.simulate(n_mc=5000)
    sim_latency.plot(results_latency)
    sim_latency.save_csv(results_latency)

    print("\n" + "=" * 70)
    print("All simulations complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
