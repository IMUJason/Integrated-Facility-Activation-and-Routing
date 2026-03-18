#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time-limited compact MIP benchmark for medium/large instances."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

from docplex.mp.model import Model

from common import BenchmarkInstance, evaluate_solution, read_json, write_json


def route_service_starts(instance: BenchmarkInstance, facility_id: int, customers: List[int]) -> List[float]:
    starts: List[float] = []
    current_time = float(instance.day_start)
    previous_kind = "f"
    previous_id = facility_id
    for customer_id in customers:
        travel_minutes = instance.travel_time(previous_kind, previous_id, "c", customer_id, current_time)
        current_time += travel_minutes
        customer = instance.customer_map[customer_id]
        if current_time < customer.earliest:
            current_time = float(customer.earliest)
        starts.append(current_time)
        current_time += customer.service_time
        previous_kind = "c"
        previous_id = customer_id
    return starts


def vehicle_limit(instance: BenchmarkInstance, warm_start_routes: int) -> int:
    total_demand = sum(customer.demand for customer in instance.customers)
    minimum_routes = math.ceil(total_demand / instance.vehicle_capacity)
    slack = max(3, math.ceil(0.10 * minimum_routes))
    upper = max(minimum_routes + slack, warm_start_routes + 2)
    return min(len(instance.customers), upper)


def build_arc_sets(instance: BenchmarkInstance) -> Dict:
    customers = instance.customers
    facilities = instance.facilities
    start_arcs = {}
    end_arcs = {}
    customer_arcs = {}

    for facility in facilities:
        for customer in customers:
            start_time = instance.travel_time("f", facility.id, "c", customer.id, instance.day_start)
            end_time = instance.travel_time("c", customer.id, "f", facility.id, instance.day_start)
            direct_finish = max(instance.day_start + start_time, customer.earliest) + customer.service_time + end_time
            if instance.day_start + start_time <= customer.latest and direct_finish <= instance.day_start + instance.max_route_minutes:
                start_arcs[(facility.id, customer.id)] = {
                    "time": start_time,
                    "distance": instance.distance("f", facility.id, "c", customer.id),
                }
                end_arcs[(customer.id, facility.id)] = {
                    "time": end_time,
                    "distance": instance.distance("c", customer.id, "f", facility.id),
                }

    for left in customers:
        min_return = min(
            end_arcs[(left.id, facility.id)]["time"]
            for facility in facilities
            if (left.id, facility.id) in end_arcs
        )
        if left.earliest + left.service_time + min_return > instance.day_start + instance.max_route_minutes:
            continue
        for right in customers:
            if left.id == right.id:
                continue
            travel_time = instance.travel_time("c", left.id, "c", right.id, instance.day_start)
            earliest_arrival = left.earliest + left.service_time + travel_time
            if earliest_arrival > right.latest:
                continue
            customer_arcs[(left.id, right.id)] = {
                "time": travel_time,
                "distance": instance.distance("c", left.id, "c", right.id),
            }

    return {
        "start_arcs": start_arcs,
        "end_arcs": end_arcs,
        "customer_arcs": customer_arcs,
    }


def build_mip_start(
    model: Model,
    instance: BenchmarkInstance,
    warm_start: Dict,
    used_vars: Dict[int, object],
    open_vars: Dict[int, object],
    assign_vars: Dict[Tuple[int, int], object],
    visit_vars: Dict[Tuple[int, int], object],
    start_vars: Dict[Tuple[int, int, int], object],
    customer_vars: Dict[Tuple[int, int, int], object],
    end_vars: Dict[Tuple[int, int, int], object],
    load_vars: Dict[Tuple[int, int], object],
    time_vars: Dict[Tuple[int, int], object],
) -> None:
    mip_start = model.new_solution()
    for route_index, route in enumerate(warm_start.get("routes", [])):
        if route_index not in used_vars or not route["customers"]:
            continue
        facility_id = route["facility_id"]
        customers = list(route["customers"])
        route_load = sum(instance.customer_map[customer_id].demand for customer_id in customers)
        mip_start.add_var_value(used_vars[route_index], 1)
        mip_start.add_var_value(open_vars[facility_id], 1)
        mip_start.add_var_value(assign_vars[(facility_id, route_index)], 1)
        mip_start.add_var_value(load_vars[(facility_id, route_index)], route_load)

        service_starts = route_service_starts(instance, facility_id, customers)
        first_customer = customers[0]
        last_customer = customers[-1]
        if (facility_id, first_customer, route_index) in start_vars:
            mip_start.add_var_value(start_vars[(facility_id, first_customer, route_index)], 1)
        if (last_customer, facility_id, route_index) in end_vars:
            mip_start.add_var_value(end_vars[(last_customer, facility_id, route_index)], 1)

        for position, customer_id in enumerate(customers):
            mip_start.add_var_value(visit_vars[(customer_id, route_index)], 1)
            mip_start.add_var_value(time_vars[(customer_id, route_index)], service_starts[position])
            if position + 1 < len(customers):
                next_customer = customers[position + 1]
                if (customer_id, next_customer, route_index) in customer_vars:
                    mip_start.add_var_value(customer_vars[(customer_id, next_customer, route_index)], 1)

    model.add_mip_start(mip_start)


def reconstruct_solution(
    instance: BenchmarkInstance,
    vehicles: List[int],
    used_vars: Dict[int, object],
    assign_vars: Dict[Tuple[int, int], object],
    start_vars: Dict[Tuple[int, int, int], object],
    customer_vars: Dict[Tuple[int, int, int], object],
) -> Dict:
    routes = []
    customer_ids = [customer.id for customer in instance.customers]
    facility_ids = [facility.id for facility in instance.facilities]

    for vehicle in vehicles:
        if used_vars[vehicle].solution_value < 0.5:
            continue
        facility_id = next(
            facility_id
            for facility_id in facility_ids
            if assign_vars[(facility_id, vehicle)].solution_value > 0.5
        )
        first_customer = next(
            customer_id
            for customer_id in customer_ids
            if (facility_id, customer_id, vehicle) in start_vars
            and start_vars[(facility_id, customer_id, vehicle)].solution_value > 0.5
        )
        route_customers = [first_customer]
        current_customer = first_customer
        while True:
            next_customer = None
            for candidate in customer_ids:
                key = (current_customer, candidate, vehicle)
                if key in customer_vars and customer_vars[key].solution_value > 0.5:
                    next_customer = candidate
                    break
            if next_customer is None:
                break
            route_customers.append(next_customer)
            current_customer = next_customer
        routes.append({"facility_id": facility_id, "customers": route_customers})

    return {"routes": routes}


def solve_time_limited_mip(
    instance: BenchmarkInstance,
    time_limit: int,
    warm_start: Dict | None = None,
) -> Dict:
    start_clock = time.time()
    warm_routes = len(warm_start.get("routes", [])) if warm_start else 0
    vehicles = list(range(vehicle_limit(instance, warm_routes)))
    arc_sets = build_arc_sets(instance)
    start_arcs = arc_sets["start_arcs"]
    end_arcs = arc_sets["end_arcs"]
    customer_arcs = arc_sets["customer_arcs"]

    model = Model(name=f"cwd_compact_{instance.name}")
    model.parameters.timelimit = time_limit
    model.parameters.mip.tolerances.mipgap = 0.0
    model.parameters.emphasis.mip = 1
    model.context.solver.log_output = False

    open_vars = {
        facility.id: model.binary_var(name=f"open_{facility.id}")
        for facility in instance.facilities
    }
    used_vars = {vehicle: model.binary_var(name=f"used_{vehicle}") for vehicle in vehicles}
    assign_vars = {
        (facility.id, vehicle): model.binary_var(name=f"z_{facility.id}_{vehicle}")
        for facility in instance.facilities
        for vehicle in vehicles
    }
    visit_vars = {
        (customer.id, vehicle): model.binary_var(name=f"v_{customer.id}_{vehicle}")
        for customer in instance.customers
        for vehicle in vehicles
    }
    start_vars = {
        (facility_id, customer_id, vehicle): model.binary_var(name=f"xs_{facility_id}_{customer_id}_{vehicle}")
        for (facility_id, customer_id) in start_arcs
        for vehicle in vehicles
    }
    customer_vars = {
        (left_id, right_id, vehicle): model.binary_var(name=f"xc_{left_id}_{right_id}_{vehicle}")
        for (left_id, right_id) in customer_arcs
        for vehicle in vehicles
    }
    end_vars = {
        (customer_id, facility_id, vehicle): model.binary_var(name=f"xe_{customer_id}_{facility_id}_{vehicle}")
        for (customer_id, facility_id) in end_arcs
        for vehicle in vehicles
    }
    load_vars = {
        (facility.id, vehicle): model.continuous_var(lb=0, ub=instance.vehicle_capacity, name=f"load_{facility.id}_{vehicle}")
        for facility in instance.facilities
        for vehicle in vehicles
    }
    time_vars = {
        (customer.id, vehicle): model.continuous_var(lb=0, ub=customer.latest, name=f"t_{customer.id}_{vehicle}")
        for customer in instance.customers
        for vehicle in vehicles
    }

    incoming_by_customer = {customer.id: [] for customer in instance.customers}
    outgoing_by_customer = {customer.id: [] for customer in instance.customers}
    starts_by_facility = {facility.id: [] for facility in instance.facilities}
    ends_by_facility = {facility.id: [] for facility in instance.facilities}

    for (facility_id, customer_id, vehicle), var in start_vars.items():
        incoming_by_customer[customer_id].append(var)
        starts_by_facility[facility_id].append((vehicle, var))
    for (left_id, right_id, vehicle), var in customer_vars.items():
        outgoing_by_customer[left_id].append(var)
        incoming_by_customer[right_id].append(var)
    for (customer_id, facility_id, vehicle), var in end_vars.items():
        outgoing_by_customer[customer_id].append(var)
        ends_by_facility[facility_id].append((vehicle, var))

    for customer in instance.customers:
        model.add_constraint(
            model.sum(visit_vars[(customer.id, vehicle)] for vehicle in vehicles) == 1,
            ctname=f"cover_{customer.id}",
        )
        for vehicle in vehicles:
            model.add_constraint(
                model.sum(
                    start_vars[(facility.id, customer.id, vehicle)]
                    for facility in instance.facilities
                    if (facility.id, customer.id, vehicle) in start_vars
                )
                + model.sum(
                    customer_vars[(other.id, customer.id, vehicle)]
                    for other in instance.customers
                    if other.id != customer.id and (other.id, customer.id, vehicle) in customer_vars
                )
                == visit_vars[(customer.id, vehicle)],
                ctname=f"in_{customer.id}_{vehicle}",
            )
            model.add_constraint(
                model.sum(
                    end_vars[(customer.id, facility.id, vehicle)]
                    for facility in instance.facilities
                    if (customer.id, facility.id, vehicle) in end_vars
                )
                + model.sum(
                    customer_vars[(customer.id, other.id, vehicle)]
                    for other in instance.customers
                    if other.id != customer.id and (customer.id, other.id, vehicle) in customer_vars
                )
                == visit_vars[(customer.id, vehicle)],
                ctname=f"out_{customer.id}_{vehicle}",
            )

    for vehicle in vehicles:
        route_load = model.sum(
            customer.demand * visit_vars[(customer.id, vehicle)]
            for customer in instance.customers
        )
        model.add_constraint(
            model.sum(assign_vars[(facility.id, vehicle)] for facility in instance.facilities) == used_vars[vehicle],
            ctname=f"assign_once_{vehicle}",
        )
        model.add_constraint(
            route_load <= instance.vehicle_capacity * used_vars[vehicle],
            ctname=f"veh_cap_{vehicle}",
        )
        model.add_constraint(
            model.sum(load_vars[(facility.id, vehicle)] for facility in instance.facilities) == route_load,
            ctname=f"load_balance_{vehicle}",
        )
        if vehicle + 1 in used_vars:
            model.add_constraint(used_vars[vehicle] >= used_vars[vehicle + 1], ctname=f"sym_{vehicle}")

        for facility in instance.facilities:
            model.add_constraint(
                model.sum(
                    start_vars[(facility.id, customer.id, vehicle)]
                    for customer in instance.customers
                    if (facility.id, customer.id, vehicle) in start_vars
                )
                == assign_vars[(facility.id, vehicle)],
                ctname=f"start_{facility.id}_{vehicle}",
            )
            model.add_constraint(
                model.sum(
                    end_vars[(customer.id, facility.id, vehicle)]
                    for customer in instance.customers
                    if (customer.id, facility.id, vehicle) in end_vars
                )
                == assign_vars[(facility.id, vehicle)],
                ctname=f"end_{facility.id}_{vehicle}",
            )
            model.add_constraint(
                load_vars[(facility.id, vehicle)] <= instance.vehicle_capacity * assign_vars[(facility.id, vehicle)],
                ctname=f"load_up_{facility.id}_{vehicle}",
            )
            model.add_constraint(
                load_vars[(facility.id, vehicle)] <= route_load,
                ctname=f"load_match_up_{facility.id}_{vehicle}",
            )
            model.add_constraint(
                load_vars[(facility.id, vehicle)] >= route_load - instance.vehicle_capacity * (1 - assign_vars[(facility.id, vehicle)]),
                ctname=f"load_match_low_{facility.id}_{vehicle}",
            )
            model.add_constraint(
                assign_vars[(facility.id, vehicle)] <= open_vars[facility.id],
                ctname=f"open_link_{facility.id}_{vehicle}",
            )

    for facility in instance.facilities:
        model.add_constraint(
            model.sum(load_vars[(facility.id, vehicle)] for vehicle in vehicles) <= facility.capacity * open_vars[facility.id],
            ctname=f"fac_cap_{facility.id}",
        )

    for customer in instance.customers:
        for vehicle in vehicles:
            visit_var = visit_vars[(customer.id, vehicle)]
            model.add_constraint(
                time_vars[(customer.id, vehicle)] >= customer.earliest * visit_var,
                ctname=f"earliest_{customer.id}_{vehicle}",
            )
            model.add_constraint(
                time_vars[(customer.id, vehicle)] <= customer.latest * visit_var + customer.latest * (1 - visit_var),
                ctname=f"latest_{customer.id}_{vehicle}",
            )

    for (facility_id, customer_id), arc in start_arcs.items():
        customer = instance.customer_map[customer_id]
        big_m = max(0.0, instance.day_start + arc["time"] - customer.earliest)
        for vehicle in vehicles:
            var = start_vars[(facility_id, customer_id, vehicle)]
            model.add_constraint(
                time_vars[(customer_id, vehicle)] >= instance.day_start + arc["time"] - big_m * (1 - var),
                ctname=f"time_start_{facility_id}_{customer_id}_{vehicle}",
            )

    for (left_id, right_id), arc in customer_arcs.items():
        left_customer = instance.customer_map[left_id]
        right_customer = instance.customer_map[right_id]
        big_m = max(0.0, left_customer.latest + left_customer.service_time + arc["time"] - right_customer.earliest)
        for vehicle in vehicles:
            var = customer_vars[(left_id, right_id, vehicle)]
            model.add_constraint(
                time_vars[(right_id, vehicle)]
                >= time_vars[(left_id, vehicle)] + left_customer.service_time + arc["time"] - big_m * (1 - var),
                ctname=f"time_arc_{left_id}_{right_id}_{vehicle}",
            )

    for (customer_id, facility_id), arc in end_arcs.items():
        customer = instance.customer_map[customer_id]
        big_m = max(
            0.0,
            customer.latest + customer.service_time + arc["time"] - (instance.day_start + instance.max_route_minutes),
        )
        for vehicle in vehicles:
            var = end_vars[(customer_id, facility_id, vehicle)]
            model.add_constraint(
                time_vars[(customer_id, vehicle)] + customer.service_time + arc["time"]
                <= instance.day_start + instance.max_route_minutes + big_m * (1 - var),
                ctname=f"time_end_{customer_id}_{facility_id}_{vehicle}",
            )

    objective = (
        model.sum(facility.fixed_cost * open_vars[facility.id] for facility in instance.facilities)
        + instance.vehicle_fixed_cost * model.sum(used_vars[vehicle] for vehicle in vehicles)
        + instance.transport_cost_per_km
        * (
            model.sum(
                arc["distance"] * start_vars[(facility_id, customer_id, vehicle)]
                for (facility_id, customer_id), arc in start_arcs.items()
                for vehicle in vehicles
            )
            + model.sum(
                arc["distance"] * customer_vars[(left_id, right_id, vehicle)]
                for (left_id, right_id), arc in customer_arcs.items()
                for vehicle in vehicles
            )
            + model.sum(
                arc["distance"] * end_vars[(customer_id, facility_id, vehicle)]
                for (customer_id, facility_id), arc in end_arcs.items()
                for vehicle in vehicles
            )
        )
    )
    model.minimize(objective)

    if warm_start:
        build_mip_start(
            model,
            instance,
            warm_start,
            used_vars,
            open_vars,
            assign_vars,
            visit_vars,
            start_vars,
            customer_vars,
            end_vars,
            load_vars,
            time_vars,
        )

    solution = model.solve(log_output=False)
    elapsed = time.time() - start_clock
    solve_details = model.solve_details
    incumbent_objective = float(solution.objective_value) if solution is not None else None
    best_bound = float(solve_details.best_bound) if solve_details.best_bound is not None else None
    if incumbent_objective is not None and incumbent_objective > 1e-9 and best_bound is not None and best_bound <= 1e-6:
        best_bound = None
    mip_gap_pct = None
    if incumbent_objective is not None and best_bound is not None and incumbent_objective > 1e-9:
        mip_gap_pct = (incumbent_objective - best_bound) / incumbent_objective * 100.0

    if solution is None:
        return {
            "instance": instance.name,
            "algorithm": "CPLEX-Compact-600s",
            "status": str(solve_details.status),
            "elapsed_seconds": elapsed,
            "vehicle_limit": len(vehicles),
            "best_bound": best_bound,
            "mip_gap_pct": mip_gap_pct,
        }

    rebuilt = reconstruct_solution(
        instance,
        vehicles,
        used_vars,
        assign_vars,
        start_vars,
        customer_vars,
    )
    metrics = evaluate_solution(instance, rebuilt)
    return {
        "instance": instance.name,
        "algorithm": "CPLEX-Compact-600s",
        "status": str(solve_details.status),
        "elapsed_seconds": elapsed,
        "vehicle_limit": len(vehicles),
        "best_bound": best_bound,
        "mip_gap_pct": mip_gap_pct,
        "incumbent_objective": incumbent_objective,
        "best_metrics": metrics,
        "best_solution": rebuilt,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = read_json(root / "results" / "experiments_summary.json")
    instance = BenchmarkInstance.load(root / "instances" / "QZ-medium-1.json")
    alns_runs = summary["heuristic_benchmark"]["QZ-medium-1"]["ALNS"]["runs"]
    rl_runs = summary["heuristic_benchmark"]["QZ-medium-1"]["RL-ALNS"]["runs"]
    warm_start = min(alns_runs + rl_runs, key=lambda item: item["best_metrics"]["objective"])["best_solution"]
    result = solve_time_limited_mip(instance, time_limit=600, warm_start=warm_start)
    output_path = root / "results" / "compact_mip_debug.json"
    write_json(output_path, result)
    print("Saved:", output_path)
    print("Status:", result["status"])
    if "best_metrics" in result:
        print("Objective:", round(result["best_metrics"]["objective"], 2))
        print("Best bound:", round(result["best_bound"], 2))


if __name__ == "__main__":
    main()
