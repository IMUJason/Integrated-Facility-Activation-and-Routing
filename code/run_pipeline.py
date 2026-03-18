#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the public computational pipeline from the released benchmark instances."""

from __future__ import annotations

import csv
import statistics as stats
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import BenchmarkInstance, ROOT, evaluate_solution, read_json, write_json
from alns_cwd_vrp import solve_with_alns
from benders_cwd_vrp import solve_with_lbbd
from cplex_compact_mip import solve_time_limited_mip
from cplex_cwd_vrp import solve_exact
from rl_alns_cwd_vrp import solve_with_rl_alns


RESULT_DIR = ROOT / "results"
TABLE_DIR = RESULT_DIR / "tables"

EXACT_INSTANCES = ["QZ-real-1", "QZ-exact-1"]
HEURISTIC_SETTINGS = {
    "QZ-small-1": {"alns_iterations": 250, "rl_iterations": 300, "seeds": [3, 7]},
    "QZ-medium-1": {"alns_iterations": 250, "rl_iterations": 300, "seeds": [3, 7]},
    "QZ-medium-2": {"alns_iterations": 220, "rl_iterations": 260, "seeds": [3, 7]},
    "QZ-large-1": {"alns_iterations": 180, "rl_iterations": 220, "seeds": [3, 7]},
}
BENDERS_INSTANCES = {
    "QZ-real-1": {"max_iterations": 8, "seed": 3},
    "QZ-medium-1": {"max_iterations": 8, "seed": 3},
    "QZ-medium-2": {"max_iterations": 6, "seed": 3},
}
CPLEX_BOUND_INSTANCES = {
    "QZ-medium-1": {"time_limit": 600},
    "QZ-medium-2": {"time_limit": 600},
}
def load_instance(name: str) -> BenchmarkInstance:
    return BenchmarkInstance.load(ROOT / "instances" / f"{name}.json")


def summarise_runs(results: List[Dict], label: str) -> Dict:
    objectives = [result["best_metrics"]["objective"] for result in results]
    distances = [result["best_metrics"]["travel_distance"] for result in results]
    routes = [result["best_metrics"]["routes"] for result in results]
    feasible = [result["best_metrics"]["feasible"] for result in results]
    elapsed = [result["elapsed_seconds"] for result in results]
    return {
        "label": label,
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_distance": float(np.mean(distances)),
        "mean_routes": float(np.mean(routes)),
        "mean_elapsed": float(np.mean(elapsed)),
        "all_feasible": all(feasible),
        "seeds": [result["seed"] for result in results],
        "runs": results,
    }


def write_csv(path: Path, headers: List[str], rows: List[List]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def run_exact_validation() -> Dict:
    exact_results = {}
    table_rows = []

    for name in EXACT_INSTANCES:
        instance = load_instance(name)
        exact = solve_exact(instance, time_limit=180)
        alns = solve_with_alns(instance, iterations=280, seed=7)
        exact_results[name] = {"exact": exact, "alns": alns}
        opt_obj = exact["best_metrics"]["objective"]
        alns_obj = alns["best_metrics"]["objective"]
        gap_pct = (alns_obj - opt_obj) / opt_obj * 100.0
        table_rows.append(
            [
                name,
                round(opt_obj, 2),
                round(alns_obj, 2),
                round(gap_pct, 2),
                exact["best_metrics"]["routes"],
                alns["best_metrics"]["routes"],
                round(exact["elapsed_seconds"], 2),
                round(alns["elapsed_seconds"], 2),
            ]
        )

    write_csv(
        TABLE_DIR / "exact_validation.csv",
        ["Instance", "ExactObjective", "ALNSObjective", "GapPct", "ExactRoutes", "ALNSRoutes", "ExactTime", "ALNSTime"],
        table_rows,
    )
    return exact_results


def run_heuristic_benchmark() -> Dict:
    payload = {}
    summary_rows = []

    for name, settings in HEURISTIC_SETTINGS.items():
        instance = load_instance(name)
        alns_runs = [
            solve_with_alns(instance, iterations=settings["alns_iterations"], seed=seed)
            for seed in settings["seeds"]
        ]
        rl_runs = [
            solve_with_rl_alns(instance, iterations=settings["rl_iterations"], seed=seed)
            for seed in settings["seeds"]
        ]
        payload[name] = {
            "ALNS": summarise_runs(alns_runs, "ALNS"),
            "RL-ALNS": summarise_runs(rl_runs, "RL-ALNS"),
        }
        alns_summary = payload[name]["ALNS"]
        rl_summary = payload[name]["RL-ALNS"]
        improvement_pct = (
            (alns_summary["mean_objective"] - rl_summary["mean_objective"])
            / alns_summary["mean_objective"]
            * 100.0
        )
        summary_rows.append(
            [
                name,
                round(alns_summary["mean_objective"], 2),
                round(rl_summary["mean_objective"], 2),
                round(improvement_pct, 2),
                round(alns_summary["mean_distance"], 2),
                round(rl_summary["mean_distance"], 2),
                round(alns_summary["mean_elapsed"], 2),
                round(rl_summary["mean_elapsed"], 2),
            ]
        )

    write_csv(
        TABLE_DIR / "heuristic_benchmark.csv",
        ["Instance", "ALNSObjective", "RLObjective", "ImprovementPct", "ALNSDistance", "RLDistance", "ALNSTime", "RLTime"],
        summary_rows,
    )
    return payload


def run_benders_benchmark() -> Dict:
    payload = {}
    table_rows = []

    for name, settings in BENDERS_INSTANCES.items():
        instance = load_instance(name)
        standard = solve_with_lbbd(
            instance,
            seed=settings["seed"],
            accelerated=False,
            max_iterations=settings["max_iterations"],
        )
        accelerated = solve_with_lbbd(
            instance,
            seed=settings["seed"],
            accelerated=True,
            max_iterations=settings["max_iterations"],
        )
        payload[name] = {"LBBD": standard, "Accelerated-LBBD": accelerated}
        table_rows.append(
            [
                name,
                round(standard["best_metrics"]["objective"], 2),
                round(accelerated["best_metrics"]["objective"], 2),
                round(standard["elapsed_seconds"], 2),
                round(accelerated["elapsed_seconds"], 2),
                len(standard["history"]),
                len(accelerated["history"]),
            ]
        )

    write_csv(
        TABLE_DIR / "benders_benchmark.csv",
        ["Instance", "LBBDObjective", "AcceleratedObjective", "LBBDTime", "AcceleratedTime", "LBBDIters", "AcceleratedIters"],
        table_rows,
    )
    return payload


def run_cplex_time_limited_benchmark(heuristic_payload: Dict) -> Dict:
    payload = {}
    table_rows = []

    for name, settings in CPLEX_BOUND_INSTANCES.items():
        instance = load_instance(name)
        alns_best = min(
            heuristic_payload[name]["ALNS"]["runs"],
            key=lambda item: item["best_metrics"]["objective"],
        )
        rl_best = min(
            heuristic_payload[name]["RL-ALNS"]["runs"],
            key=lambda item: item["best_metrics"]["objective"],
        )
        warm_start_run = min([alns_best, rl_best], key=lambda item: item["best_metrics"]["objective"])
        mip_result = solve_time_limited_mip(
            instance,
            time_limit=settings["time_limit"],
            warm_start=warm_start_run["best_solution"],
        )

        incumbent = mip_result.get("best_metrics", {}).get("objective")
        best_bound = mip_result.get("best_bound")
        alns_obj = alns_best["best_metrics"]["objective"]
        rl_obj = rl_best["best_metrics"]["objective"]

        alns_gap_to_ub = ((alns_obj - incumbent) / incumbent * 100.0) if incumbent else None
        rl_gap_to_ub = ((rl_obj - incumbent) / incumbent * 100.0) if incumbent else None
        alns_gap_to_lb = ((alns_obj - best_bound) / best_bound * 100.0) if best_bound else None
        rl_gap_to_lb = ((rl_obj - best_bound) / best_bound * 100.0) if best_bound else None

        payload[name] = {
            "warm_start_algorithm": warm_start_run["algorithm"],
            "warm_start_objective": warm_start_run["best_metrics"]["objective"],
            "ALNS_best": alns_best,
            "RL-ALNS_best": rl_best,
            "CPLEX-Compact-600s": mip_result,
        }

        table_rows.append(
            [
                name,
                warm_start_run["algorithm"],
                round(incumbent, 2) if incumbent else "",
                round(best_bound, 2) if best_bound else "",
                round(mip_result["mip_gap_pct"], 2) if mip_result.get("mip_gap_pct") is not None else "",
                round(alns_obj, 2),
                round(alns_gap_to_ub, 2) if alns_gap_to_ub is not None else "",
                round(alns_gap_to_lb, 2) if alns_gap_to_lb is not None else "",
                round(rl_obj, 2),
                round(rl_gap_to_ub, 2) if rl_gap_to_ub is not None else "",
                round(rl_gap_to_lb, 2) if rl_gap_to_lb is not None else "",
                mip_result["status"],
            ]
        )

    write_csv(
        TABLE_DIR / "cplex_600s_benchmark.csv",
        [
            "Instance",
            "WarmStart",
            "CPLEXIncumbent",
            "CPLEXBestBound",
            "CPLEXGapPct",
            "ALNSBestObjective",
            "ALNSGapToUBPct",
            "ALNSGapToLBPct",
            "RLBestObjective",
            "RLGapToUBPct",
            "RLGapToLBPct",
            "Status",
        ],
        table_rows,
    )
    return payload


def scenario_evaluate(instance: BenchmarkInstance, solution: Dict, multipliers: Dict[int, float]) -> Dict:
    perturbed = BenchmarkInstance.from_dict(instance.to_dict())
    for customer in perturbed.customers:
        scaled = int(round(customer.demand * multipliers.get(customer.id, 1.0)))
        customer.demand = max(1, scaled)
    perturbed.__post_init__()
    metrics = evaluate_solution(perturbed, solution)
    return metrics


def run_sensitivity_and_robustness(heuristic_payload: Dict) -> Dict:
    instance = load_instance("QZ-medium-2")
    sensitivity_rows = []

    for fixed_multiplier in [0.8, 1.0, 1.2]:
        for restriction_penalty in [1.0, 1.18, 1.35]:
            test_instance = BenchmarkInstance.from_dict(instance.to_dict())
            for facility in test_instance.facilities:
                facility.fixed_cost = round(facility.fixed_cost * fixed_multiplier, 2)
            test_instance.restricted_penalty_factor = restriction_penalty
            test_instance.__post_init__()
            result = solve_with_rl_alns(test_instance, iterations=180, seed=5)
            metrics = result["best_metrics"]
            sensitivity_rows.append(
                {
                    "facility_cost_multiplier": fixed_multiplier,
                    "restriction_penalty": restriction_penalty,
                    "objective": metrics["objective"],
                    "open_facilities": metrics["open_facilities"],
                    "routes": metrics["routes"],
                    "distance": metrics["travel_distance"],
                }
            )

    write_csv(
        TABLE_DIR / "sensitivity.csv",
        ["FacilityCostMultiplier", "RestrictionPenalty", "Objective", "OpenFacilities", "Routes", "Distance"],
        [
            [
                row["facility_cost_multiplier"],
                row["restriction_penalty"],
                round(row["objective"], 2),
                row["open_facilities"],
                row["routes"],
                round(row["distance"], 2),
            ]
            for row in sensitivity_rows
        ],
    )

    alns_solution = min(
        heuristic_payload["QZ-medium-2"]["ALNS"]["runs"],
        key=lambda item: item["best_metrics"]["objective"],
    )["best_solution"]
    rl_solution = min(
        heuristic_payload["QZ-medium-2"]["RL-ALNS"]["runs"],
        key=lambda item: item["best_metrics"]["objective"],
    )["best_solution"]
    rng = np.random.default_rng(20260311)
    scenarios = []
    for scenario_id in range(20):
        multipliers = {
            customer.id: float(np.clip(rng.normal(1.0, 0.10), 0.8, 1.2))
            for customer in instance.customers
        }
        alns_metrics = scenario_evaluate(instance, alns_solution, multipliers)
        rl_metrics = scenario_evaluate(instance, rl_solution, multipliers)
        scenarios.append(
            {
                "scenario": scenario_id,
                "alns_objective": alns_metrics["objective"],
                "rl_objective": rl_metrics["objective"],
                "alns_penalty": alns_metrics["penalty"],
                "rl_penalty": rl_metrics["penalty"],
            }
        )

    write_csv(
        TABLE_DIR / "robustness.csv",
        ["Scenario", "ALNSObjective", "RLObjective", "ALNSPenalty", "RLPenalty"],
        [
            [
                row["scenario"],
                round(row["alns_objective"], 2),
                round(row["rl_objective"], 2),
                round(row["alns_penalty"], 2),
                round(row["rl_penalty"], 2),
            ]
            for row in scenarios
        ],
    )

    return {"sensitivity": sensitivity_rows, "robustness": scenarios}


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    exact_payload = run_exact_validation()
    heuristic_payload = run_heuristic_benchmark()
    cplex_bound_payload = run_cplex_time_limited_benchmark(heuristic_payload)
    benders_payload = run_benders_benchmark()
    scenario_payload = run_sensitivity_and_robustness(heuristic_payload)

    summary = {
        "exact_validation": exact_payload,
        "heuristic_benchmark": heuristic_payload,
        "cplex_time_limited_benchmark": cplex_bound_payload,
        "benders_benchmark": benders_payload,
        "scenario_analysis": scenario_payload,
    }
    write_json(RESULT_DIR / "experiments_summary.json", summary)
    print("Pipeline finished. Summary written to:")
    print(RESULT_DIR / "experiments_summary.json")


if __name__ == "__main__":
    main()
