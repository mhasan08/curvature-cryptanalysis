"""Compute shared-stencil structural + completion query budgets."""

from __future__ import annotations

import argparse


def structural_queries(d: int, probe_locations: int = 1) -> int:
    if d < 1:
        raise ValueError("d must be at least 1")
    if probe_locations < 1:
        raise ValueError("probe_locations must be at least 1")
    return probe_locations * (2 * d * d + 1)


def total_queries(d: int, probe_locations: int, completion_queries: int) -> int:
    if completion_queries < 0:
        raise ValueError("completion_queries must be non-negative")
    return structural_queries(d, probe_locations) + completion_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--probe-locations", "--P", dest="probe_locations", type=int, default=1)
    parser.add_argument("--projections", "--T", dest="projections", type=int, default=16)
    parser.add_argument("--completion-queries", type=int, default=12000)
    args = parser.parse_args()

    if args.projections < args.probe_locations or args.projections % args.probe_locations:
        parser.error("T must be divisible by P & satisfy T >= P")

    q_struct = structural_queries(args.d, args.probe_locations)
    q_total = total_queries(args.d, args.probe_locations, args.completion_queries)
    q_per_probe = structural_queries(args.d, 1)
    q_unshared = args.projections * q_per_probe
    q_per_location = args.projections // args.probe_locations

    print(f"d={args.d}, P={args.probe_locations}, T={args.projections}, Q={q_per_location}")
    print(f"structural_queries={q_struct:,}")
    print(f"completion_queries={args.completion_queries:,}")
    print(f"total_queries={q_total:,}")
    print(f"unshared_structural_baseline={q_unshared:,}")
    print(f"structural_reduction={q_unshared / q_struct:.2f}x")

# main
if __name__ == "__main__":
    main()
