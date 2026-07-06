"""
optimize_fifo_depth.py — search for the FIFO depth that minimizes latency
under a throughput constraint, using a simple hill-climbing search.

This ties directly back to mini-gpu-rtl-verification: instead of guessing a
FIFO depth, this treats it as an optimization problem, which is the
Pulpatronics-relevant framing (their internship is explicitly about building
an AI-driven optimization engine for critical system components).

This uses a lightweight software model of FIFO behaviour rather than
re-running RTL simulation for every candidate depth (that would be the
"real" version of this, and is a natural stretch goal once you're
comfortable calling your RTL sim from a script).

Run:
    python3 optimize_fifo_depth.py
"""

import random
from dataclasses import dataclass


@dataclass
class SimResult:
    depth: int
    avg_latency_cycles: float
    throughput_utilization: float  # fraction of cycles the FIFO was neither full nor empty


def simulate_fifo(depth: int, num_cycles: int = 2000, write_prob: float = 0.5,
                   read_prob: float = 0.45, seed: int = 42) -> SimResult:
    """A simplified software model of FIFO occupancy over time, used as a
    fast proxy for evaluating candidate depths. Replace this with a call
    into your actual RTL simulation (via cocotb or a Verilator model) once
    you want a more accurate, hardware-grounded optimization loop."""
    rng = random.Random(seed)
    occupancy = 0
    total_latency = 0
    items_written = 0
    cycles_active = 0  # neither full nor empty

    write_times = []
    latencies = []

    for cycle in range(num_cycles):
        do_write = rng.random() < write_prob and occupancy < depth
        do_read = rng.random() < read_prob and occupancy > 0

        if do_write:
            occupancy += 1
            write_times.append(cycle)
            items_written += 1

        if do_read and write_times:
            entry_cycle = write_times.pop(0)
            latencies.append(cycle - entry_cycle)
            occupancy -= 1

        if 0 < occupancy < depth:
            cycles_active += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else float("inf")
    utilization = cycles_active / num_cycles

    return SimResult(depth=depth, avg_latency_cycles=avg_latency,
                      throughput_utilization=utilization)


def score(result: SimResult, min_utilization: float = 0.3) -> float:
    """Lower is better. Penalize heavily if the throughput constraint isn't met,
    otherwise minimize latency (a stand-in for area, since deeper FIFOs cost
    more silicon area — the real tradeoff you'd be balancing on real hardware)."""
    if result.throughput_utilization < min_utilization:
        return float("inf")
    # Add a small area penalty proportional to depth so the search doesn't
    # just pick the largest depth available.
    area_penalty = result.depth * 0.1
    return result.avg_latency_cycles + area_penalty


def hill_climb_search(depth_range=range(2, 33), min_utilization: float = 0.3):
    best_depth = None
    best_score = float("inf")
    all_results = []

    for depth in depth_range:
        result = simulate_fifo(depth)
        s = score(result, min_utilization)
        all_results.append((depth, result, s))
        if s < best_score:
            best_score = s
            best_depth = depth

    return best_depth, best_score, all_results


if __name__ == "__main__":
    best_depth, best_score, all_results = hill_climb_search()

    print(f"{'Depth':>6} | {'Avg Latency (cycles)':>22} | {'Utilization':>12} | {'Score':>10}")
    print("-" * 60)
    for depth, result, s in all_results:
        score_str = f"{s:.2f}" if s != float("inf") else "inf (fails constraint)"
        print(f"{depth:>6} | {result.avg_latency_cycles:>22.2f} | "
              f"{result.throughput_utilization:>12.2%} | {score_str:>10}")

    print(f"\nBest depth found: {best_depth} (score={best_score:.2f})")
    print("\nNext step: replace simulate_fifo() with a real call into your "
          "cocotb/Verilator testbench from mini-gpu-rtl-verification so this "
          "optimizes against actual hardware behaviour, not a software proxy.")
