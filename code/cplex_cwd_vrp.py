#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact CPLEX validation via route enumeration and set partitioning."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Dict, List

from docplex.mp.model import Model

from common import BenchmarkInstance, evaluate_solution, write_json


def enumerate_feasible_routes(
    instance: BenchmarkInstance,
    facility_id: int,
    max_subset_size: int = 5,
) -> List[Dict]:
    customer_ids = [customer.id for customer in instance.customers]
    routes = []
    route_id = 0
    for subset_size in range(1, min(max_subset_size, len(customer_ids)) + 1):
        for subset in itertools.combinations(customer_ids, subset_size):
            total_demand = sum(instance.customer_map[cid].demand for cid in subset)
            if total_demand > instance.vehicle_capacity:
                continue

            best_sequence = None
            best_distance = None
            for ordering in itertools.permutations(subset):
                metrics = instance.route_metrics(facility_id, ordering)
                feasible = (
                    metrics["load"] <= instance.vehicle_capacity
                    and metrics["duration"] <= instance.max_route_minutes
                    and metrics["lateness"] <= 1e-9
                )
                if not feasible:
                    continue
                if best_distance is None or metrics["distance"] < best_distance:
                    best_distance = metrics["distance"]
                    best_sequence = list(ordering)

            if best_sequence is None:
                continue

            routes.append(
                {
                    "route_id": f"r_{facility_id}_{route_id}",
                    "facility_id": facility_id,
                    "customers": best_sequence,
                    "load": total_demand,
                    "distance": best_distance,
                    "route_cost": instance.vehicle_fixed_cost + best_distance * instance.transport_cost_per_km,
                }
            )
            route_id += 1
    return routes


def solve_exact(instance: BenchmarkInstance, time_limit: int = 180) -> Dict:
    start_time = time.time()
    route_pool = []
    for facility in instance.facilities:
        route_pool.extend(enumerate_feasible_routes(instance, facility.id))

    customer_cover = {customer.id: [] for customer in instance.customers}
    facility_routes = {facility.id: [] for facility in instance.facilities}
    for route in route_pool:
        facility_routes[route["facility_id"]].append(route)
        for customer_id in route["customers"]:
            customer_cover[customer_id].append(route)

    model = Model(name=f"cwd_exact_{instance.name}")
    model.parameters.timelimit = time_limit
    model.parameters.mip.tolerances.mipgap = 0.0
    model.context.cplex_parameters.threads = 1
    model.context.solver.log_output = False

    y = {facility.id: model.binary_var(name=f"y_{facility.id}") for facility in instance.facilities}
    x = {route["route_id"]: model.binary_var(name=route["route_id"]) for route in route_pool}

    for customer_id, routes in customer_cover.items():
        if not routes:
            raise RuntimeError(f"No feasible exact route covers customer {customer_id} in {instance.name}.")
        model.add_constraint(model.sum(x[route["route_id"]] for route in routes) == 1)

    for facility in instance.facilities:
        routes = facility_routes[facility.id]
        if not routes:
            continue
        for route in routes:
            model.add_constraint(x[route["route_id"]] <= y[facility.id])
        model.add_constraint(
            model.sum(route["load"] * x[route["route_id"]] for route in routes)
            <= facility.capacity * y[facility.id]
        )

    objective = (
        model.sum(facility.fixed_cost * y[facility.id] for facility in instance.facilities)
        + model.sum(route["route_cost"] * x[route["route_id"]] for route in route_pool)
    )
    model.minimize(objective)
    solution = model.solve(log_output=False)
    elapsed = time.time() - start_time

    if solution is None:
        return {
            "instance": instance.name,
            "algorithm": "CPLEX-Exact",
            "status": "no_solution",
            "elapsed_seconds": elapsed,
        }

    chosen_routes = [route for route in route_pool if x[route["route_id"]].solution_value > 0.5]
    heuristic_solution = {
        "routes": [
            {"facility_id": route["facility_id"], "customers": list(route["customers"])}
            for route in chosen_routes
        ]
    }
    metrics = evaluate_solution(instance, heuristic_solution)
    return {
        "instance": instance.name,
        "algorithm": "CPLEX-Exact",
        "status": solution.solve_status.name,
        "elapsed_seconds": elapsed,
        "best_metrics": metrics,
        "best_solution": heuristic_solution,
        "route_pool_size": len(route_pool),
        "selected_route_ids": [route["route_id"] for route in chosen_routes],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    instance_path = root / "instances" / "QZ-exact-1.json"
    instance = BenchmarkInstance.load(instance_path)
    result = solve_exact(instance, time_limit=180)
    output_path = root / "results" / "small_instance_validation.json"
    write_json(output_path, result)
    if "best_metrics" in result:
        metrics = result["best_metrics"]
        print(f"Exact solve finished for {instance.name}")
        print(f"  Status: {result['status']}")
        print(f"  Objective: {metrics['objective']:.2f}")
        print(f"  Open facilities: {metrics['open_facilities']}")
        print(f"  Routes: {metrics['routes']}")
        print(f"  Distance: {metrics['travel_distance']:.2f}")
        print(f"  Saved to: {output_path}")
    else:
        print(f"Exact solve failed for {instance.name}: {result['status']}")


if __name__ == "__main__":
    main()
