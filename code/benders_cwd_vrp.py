#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logic-based Benders decomposition for the Quzhou CWD benchmark."""

from __future__ import annotations

import time
from typing import Dict, List

from docplex.mp.model import Model

from common import BenchmarkInstance, evaluate_solution
from alns_cwd_vrp import fixed_facility_subproblem
from rl_alns_cwd_vrp import solve_with_rl_alns


def build_master(instance: BenchmarkInstance, accelerated: bool = False):
    model = Model(name=f"cwd_lbbd_{instance.name}")
    model.context.solver.log_output = False
    model.context.cplex_parameters.threads = 1
    model.parameters.timelimit = 120

    y = {facility.id: model.binary_var(name=f"y_{facility.id}") for facility in instance.facilities}
    a = {
        (customer.id, facility.id): model.binary_var(name=f"a_{customer.id}_{facility.id}")
        for customer in instance.customers
        for facility in instance.facilities
    }
    m = {
        facility.id: model.integer_var(lb=0, ub=len(instance.customers), name=f"m_{facility.id}")
        for facility in instance.facilities
    }
    theta = {facility.id: model.continuous_var(lb=0, name=f"theta_{facility.id}") for facility in instance.facilities}

    for customer in instance.customers:
        model.add_constraint(
            model.sum(a[(customer.id, facility.id)] for facility in instance.facilities) == 1
        )

    for facility in instance.facilities:
        for customer in instance.customers:
            model.add_constraint(a[(customer.id, facility.id)] <= y[facility.id])

        model.add_constraint(
            model.sum(customer.demand * a[(customer.id, facility.id)] for customer in instance.customers)
            <= facility.capacity * y[facility.id]
        )
        model.add_constraint(
            model.sum(customer.demand * a[(customer.id, facility.id)] for customer in instance.customers)
            <= instance.vehicle_capacity * m[facility.id]
        )
        model.add_constraint(m[facility.id] <= len(instance.customers) * y[facility.id])

        if accelerated:
            workload_terms = []
            for customer in instance.customers:
                travel_minutes = instance.travel_time("f", facility.id, "c", customer.id, instance.day_start)
                workload = customer.service_time + 2.0 * travel_minutes
                workload_terms.append(workload * a[(customer.id, facility.id)])
            model.add_constraint(model.sum(workload_terms) <= instance.max_route_minutes * m[facility.id])

    surrogate = model.sum(
        0.45
        * instance.transport_cost_per_km
        * instance.distance("c", customer.id, "f", facility.id)
        * a[(customer.id, facility.id)]
        for customer in instance.customers
        for facility in instance.facilities
    )
    objective = (
        model.sum(facility.fixed_cost * y[facility.id] for facility in instance.facilities)
        + model.sum(instance.vehicle_fixed_cost * m[facility.id] for facility in instance.facilities)
        + model.sum(theta[facility.id] for facility in instance.facilities)
        + surrogate
    )
    model.minimize(objective)
    return model, y, a, m, theta


def warm_start_master(model: Model, instance: BenchmarkInstance, y, a, m, theta, warm_solution: Dict) -> None:
    start = model.new_solution()
    metrics = warm_solution["best_metrics"]
    assignments = {}
    for route in warm_solution["best_solution"]["routes"]:
        for customer_id in route["customers"]:
            assignments[customer_id] = route["facility_id"]

    facility_routes = {facility.id: 0 for facility in instance.facilities}
    facility_travel = {facility.id: 0.0 for facility in instance.facilities}
    for route in warm_solution["best_solution"]["routes"]:
        facility_routes[route["facility_id"]] += 1
        route_metrics = instance.route_metrics(route["facility_id"], route["customers"])
        facility_travel[route["facility_id"]] += route_metrics["distance"] * instance.transport_cost_per_km

    for facility in instance.facilities:
        start.add_var_value(y[facility.id], 1 if metrics["facility_usage"][facility.id] > 0 else 0)
        start.add_var_value(m[facility.id], facility_routes[facility.id])
        start.add_var_value(theta[facility.id], facility_travel[facility.id])
    for customer in instance.customers:
        for facility in instance.facilities:
            start.add_var_value(a[(customer.id, facility.id)], int(assignments.get(customer.id) == facility.id))
    model.add_mip_start(start)


def solve_with_lbbd(
    instance: BenchmarkInstance,
    seed: int = 0,
    accelerated: bool = False,
    max_iterations: int = 14,
) -> Dict:
    start_time = time.time()
    model, y, a, m, theta = build_master(instance, accelerated=accelerated)

    warm_result = None
    if accelerated:
        warm_result = solve_with_rl_alns(instance, iterations=220, seed=seed)
        warm_start_master(model, instance, y, a, m, theta, warm_result)

    history: List[Dict] = []
    best_feasible = None
    best_metrics = None
    algorithm_name = "Accelerated-LBBD" if accelerated else "LBBD"

    for iteration in range(max_iterations):
        solution = model.solve(log_output=False)
        if solution is None:
            break

        lower_bound = float(model.objective_value)
        assignments = {
            facility.id: []
            for facility in instance.facilities
        }
        mismatches = []
        for customer in instance.customers:
            chosen_facility = max(
                instance.facilities,
                key=lambda facility: a[(customer.id, facility.id)].solution_value,
            ).id
            assignments[chosen_facility].append(customer.id)
            mismatches.append((customer.id, chosen_facility))

        stitched_solution = {"routes": []}
        total_route_cost = 0.0
        total_route_count = 0
        cut_added = False

        for facility in instance.facilities:
            customer_ids = assignments[facility.id]
            if not customer_ids:
                continue
            subproblem = fixed_facility_subproblem(
                instance,
                facility.id,
                customer_ids,
                seed=seed + 17 * (iteration + 1) + facility.id,
                iterations=120 if accelerated else 70,
            )
            stitched_solution["routes"].extend(subproblem["routes"])
            total_route_cost += subproblem["travel_cost"]
            total_route_count += subproblem["vehicle_count"]

            if accelerated:
                mismatch_expr = (
                    model.sum(1 - a[(customer_id, facility.id)] for customer_id in customer_ids)
                    + model.sum(
                        a[(customer.id, facility.id)]
                        for customer in instance.customers
                        if customer.id not in customer_ids
                    )
                )
                model.add_constraint(
                    theta[facility.id]
                    >= subproblem["travel_cost"] - max(subproblem["travel_cost"], 1.0) * mismatch_expr
                )
                model.add_constraint(
                    m[facility.id]
                    >= subproblem["vehicle_count"] - max(subproblem["vehicle_count"], 1) * mismatch_expr
                )
                cut_added = True

        feasible_metrics = evaluate_solution(instance, stitched_solution)
        if best_metrics is None or feasible_metrics["objective"] < best_metrics["objective"]:
            best_metrics = feasible_metrics
            best_feasible = stitched_solution

        if not accelerated:
            mismatch_expr = model.sum(
                (1 - a[(customer_id, facility_id)])
                if facility_id == assigned_facility
                else a[(customer_id, facility_id)]
                for customer_id, assigned_facility in mismatches
                for facility_id in [assigned_facility]
            )
            total_theta = model.sum(theta[facility.id] for facility in instance.facilities)
            total_m = model.sum(m[facility.id] for facility in instance.facilities)
            model.add_constraint(total_theta >= total_route_cost - max(total_route_cost, 1.0) * mismatch_expr)
            model.add_constraint(total_m >= total_route_count - max(total_route_count, 1) * mismatch_expr)
            cut_added = True

        history.append(
            {
                "iteration": iteration,
                "lower_bound": lower_bound,
                "upper_bound": feasible_metrics["objective"],
                "gap": feasible_metrics["objective"] - lower_bound,
                "open_facilities": feasible_metrics["open_facilities"],
                "routes": feasible_metrics["routes"],
                "distance": feasible_metrics["travel_distance"],
            }
        )

        if not cut_added or feasible_metrics["objective"] - lower_bound <= 1e-4:
            break

    elapsed = time.time() - start_time
    return {
        "instance": instance.name,
        "algorithm": algorithm_name,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "best_solution": best_feasible,
        "best_metrics": best_metrics,
        "history": history,
        "warm_start": warm_result["best_metrics"] if warm_result else None,
    }


def main() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    instance = BenchmarkInstance.load(root / "instances" / "QZ-medium-1.json")
    standard = solve_with_lbbd(instance, seed=20260311, accelerated=False)
    accelerated = solve_with_lbbd(instance, seed=20260311, accelerated=True)
    print(f"LBBD objective: {standard['best_metrics']['objective']:.2f}")
    print(f"Accelerated objective: {accelerated['best_metrics']['objective']:.2f}")


if __name__ == "__main__":
    main()
