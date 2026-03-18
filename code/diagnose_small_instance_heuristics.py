#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose whether small-instance heuristic gaps come from budget or structure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List

from alns_cwd_vrp import solve_with_alns
from cplex_cwd_vrp import solve_exact
from rl_alns_cwd_vrp import solve_with_rl_alns
from common import BenchmarkInstance, ensure_dir, write_json


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = ROOT / "instances"
RESULT_DIR = ROOT / "results"
TABLE_DIR = RESULT_DIR / "tables"

EXACT_INSTANCES = ["QZ-real-1", "QZ-exact-1"]


def parse_int_list(raw: str) -> List[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def exact_candidate_reachability(
    instance: BenchmarkInstance,
    exact_solution: Dict,
    facility_limit: int | None = None,
) -> Dict:
    blocked = []
    for route in exact_solution["routes"]:
        facility_id = route["facility_id"]
        for customer_id in route["customers"]:
            ranked = instance.nearest_facilities(customer_id, limit=len(instance.facilities))
            rank = ranked.index(facility_id) + 1
            active_limit = facility_limit if facility_limit is not None else len(instance.facilities)
            if facility_id not in ranked[:active_limit]:
                blocked.append(
                    {
                        "customer_id": customer_id,
                        "exact_facility_id": facility_id,
                        "rank": rank,
                        "nearest_facilities": ranked[:active_limit],
                    }
                )
    return {
        "facility_limit": facility_limit,
        "reachable": len(blocked) == 0,
        "blocked_customer_count": len(blocked),
        "blocked_customers": blocked,
    }


def run_multistart(
    solver: Callable[[BenchmarkInstance, int, int], Dict],
    instance: BenchmarkInstance,
    exact_objective: float,
    iterations: int,
    seeds: int,
) -> Dict:
    best_result = None
    best_objective = None
    hit_count = 0

    for seed in range(seeds):
        result = solver(instance, iterations=iterations, seed=seed)
        objective = result["best_metrics"]["objective"]
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_result = result
        if abs(objective - exact_objective) <= 1e-6:
            hit_count += 1

    gap_pct = (best_objective - exact_objective) / exact_objective * 100.0
    return {
        "iterations": iterations,
        "seeds": seeds,
        "best_objective": best_objective,
        "gap_pct": gap_pct,
        "hit_count": hit_count,
        "best_seed": best_result["seed"],
        "best_metrics": best_result["best_metrics"],
        "best_solution": best_result["best_solution"],
    }


def solver_adapter(func: Callable[..., Dict]) -> Callable[[BenchmarkInstance, int, int], Dict]:
    def wrapped(instance: BenchmarkInstance, iterations: int, seed: int) -> Dict:
        return func(instance, iterations=iterations, seed=seed)

    return wrapped


def write_csv_rows(path: Path, rows: List[List[object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Instance",
                "Algorithm",
                "Iterations",
                "Seeds",
                "ExactObjective",
                "BestObjective",
                "GapPct",
                "HitCount",
                "BestSeed",
                "BestOpenFacilities",
                "BestRoutes",
                "ExactReachableUnderDefaultCandidates",
                "BlockedCustomerCount",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alns-seeds", type=int, default=30)
    parser.add_argument("--rl-seeds", type=int, default=12)
    parser.add_argument("--alns-iterations", default="280,1200")
    parser.add_argument("--rl-iterations", default="350,1200")
    parser.add_argument(
        "--output-json",
        default=str(RESULT_DIR / "small_instance_diagnostics.json"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(TABLE_DIR / "small_instance_diagnostics.csv"),
    )
    args = parser.parse_args()

    alns_grid = parse_int_list(args.alns_iterations)
    rl_grid = parse_int_list(args.rl_iterations)

    payload: Dict[str, Dict] = {}
    csv_rows: List[List[object]] = []
    alns_solver = solver_adapter(solve_with_alns)
    rl_solver = solver_adapter(solve_with_rl_alns)

    for name in EXACT_INSTANCES:
        instance = BenchmarkInstance.load(INSTANCE_DIR / f"{name}.json")
        exact = solve_exact(instance, time_limit=180)
        exact_objective = exact["best_metrics"]["objective"]
        current_reachability = exact_candidate_reachability(instance, exact["best_solution"])
        legacy_reachability = exact_candidate_reachability(instance, exact["best_solution"], facility_limit=4)

        payload[name] = {
            "exact": exact,
            "current_candidate_reachability": current_reachability,
            "legacy_limit4_reachability": legacy_reachability,
            "ALNS": {},
            "RL-ALNS": {},
        }

        for iterations in alns_grid:
            result = run_multistart(alns_solver, instance, exact_objective, iterations, args.alns_seeds)
            payload[name]["ALNS"][str(iterations)] = result
            csv_rows.append(
                [
                    name,
                    "ALNS",
                    iterations,
                    args.alns_seeds,
                    round(exact_objective, 6),
                    round(result["best_objective"], 6),
                    round(result["gap_pct"], 4),
                    result["hit_count"],
                    result["best_seed"],
                    result["best_metrics"]["open_facilities"],
                    result["best_metrics"]["routes"],
                    current_reachability["reachable"],
                    legacy_reachability["blocked_customer_count"],
                ]
            )

        for iterations in rl_grid:
            result = run_multistart(rl_solver, instance, exact_objective, iterations, args.rl_seeds)
            payload[name]["RL-ALNS"][str(iterations)] = result
            csv_rows.append(
                [
                    name,
                    "RL-ALNS",
                    iterations,
                    args.rl_seeds,
                    round(exact_objective, 6),
                    round(result["best_objective"], 6),
                    round(result["gap_pct"], 4),
                    result["hit_count"],
                    result["best_seed"],
                    result["best_metrics"]["open_facilities"],
                    result["best_metrics"]["routes"],
                    current_reachability["reachable"],
                    legacy_reachability["blocked_customer_count"],
                ]
            )

    write_json(Path(args.output_json), payload)
    write_csv_rows(Path(args.output_csv), csv_rows)

    for name in EXACT_INSTANCES:
        current_reachability = payload[name]["current_candidate_reachability"]
        legacy_reachability = payload[name]["legacy_limit4_reachability"]
        print(
            f"{name}: exact reachable under current candidates -> "
            f"{current_reachability['reachable']} | "
            f"legacy limit=4 blocked customers: {legacy_reachability['blocked_customer_count']}"
        )
        for algorithm in ("ALNS", "RL-ALNS"):
            for iteration, result in payload[name][algorithm].items():
                print(
                    f"  {algorithm:7s} iter={int(iteration):4d} seeds={result['seeds']:2d} "
                    f"best={result['best_objective']:.6f} gap={result['gap_pct']:.2f}% "
                    f"hit={result['hit_count']} best_seed={result['best_seed']}"
                )

    print(f"Saved JSON diagnostics to {args.output_json}")
    print(f"Saved CSV diagnostics to {args.output_csv}")


if __name__ == "__main__":
    main()
