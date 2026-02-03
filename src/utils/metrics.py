import numpy as np
from typing import Dict, List
from core.solution_representation import VRPSolution


def calculate_gap_from_optimal(solution: VRPSolution, instance: Dict) -> float:
    if 'optimal_cost' not in instance or instance['optimal_cost'] is None:
        return 0.0

    optimal_cost = instance['optimal_cost']
    solution_cost = solution.cost

    if optimal_cost == 0:
        return 0.0

    gap = ((solution_cost - optimal_cost) / optimal_cost) * 100
    return gap


def calculate_dispatch_rounds(solution: VRPSolution) -> int:
    num_routes = len(solution.routes)
    num_vehicles = solution.num_vehicles

    dispatch_rounds = int(np.ceil(num_routes / num_vehicles))
    return dispatch_rounds


def calculate_dispatch_rounds_increase(solution: VRPSolution, instance: Dict) -> float:
    if 'optimal_routes' not in instance or instance['optimal_routes'] is None:
        return 0.0

    optimal_routes = instance['optimal_routes']
    num_vehicles = instance.get('num_vehicles', len(optimal_routes))

    optimal_dispatch_rounds = int(np.ceil(len(optimal_routes) / num_vehicles))
    solution_dispatch_rounds = calculate_dispatch_rounds(solution)

    if optimal_dispatch_rounds == 0:
        return 0.0

    increase = ((solution_dispatch_rounds - optimal_dispatch_rounds) / optimal_dispatch_rounds) * 100
    return increase


def calculate_route_utilization(solution: VRPSolution) -> float:
    if not solution.routes:
        return 0.0

    utilizations = []
    for route in solution.routes:
        if route:
            demand = solution.calculate_route_demand(route)
            utilization = (demand / solution.capacity) * 100
            utilizations.append(utilization)

    return np.mean(utilizations) if utilizations else 0.0


def calculate_capacity_violations(solution: VRPSolution) -> int:
    violations = 0
    for route in solution.routes:
        demand = solution.calculate_route_demand(route)
        if demand > solution.capacity:
            violations += 1
    return violations


def calculate_all_metrics(solution: VRPSolution, instance: Dict) -> Dict[str, float]:
    metrics = {
        'cost': solution.cost,
        'gap_from_optimal': calculate_gap_from_optimal(solution, instance),
        'dispatch_rounds': calculate_dispatch_rounds(solution),
        'dispatch_rounds_increase': calculate_dispatch_rounds_increase(solution, instance),
        'avg_route_utilization': calculate_route_utilization(solution),
        'is_feasible': solution.is_feasible,
        'capacity_violations': calculate_capacity_violations(solution),
        'num_routes': len(solution.routes)
    }

    return metrics


def calculate_fitness(solution: VRPSolution,
                      instance: Dict,
                      w_cost: float = 0.7,
                      w_dispatch: float = 0.3,
                      infeasibility_penalty: float = 200.0) -> float:

    fitness = 0

    if not solution.is_feasible:
        fitness += calculate_capacity_violations(solution)*infeasibility_penalty

    cost_increase = calculate_gap_from_optimal(solution, instance)
    dispatch_increase = calculate_dispatch_rounds_increase(solution, instance)

    fitness += w_cost * cost_increase + w_dispatch * dispatch_increase

    return 100-fitness


def calculate_population_diversity(population: List[VRPSolution]) -> float:
    if len(population) <= 1:
        return 0.0

    costs = [sol.cost for sol in population]
    diversity = np.std(costs)

    return diversity


def calculate_population_stats(population: List[VRPSolution], instance: Dict) -> Dict[str, float]:
    if not population:
        return {}

    costs = [sol.cost for sol in population]
    fitnesses = [calculate_fitness(sol, instance) for sol in population]
    feasible_count = sum(1 for sol in population if sol.is_feasible)

    stats = {
        'population_size': len(population),
        'avg_cost': np.mean(costs),
        'min_cost': np.min(costs),
        'max_cost': np.max(costs),
        'std_cost': np.std(costs),
        'diversity': calculate_population_diversity(population),
        'avg_fitness': np.mean(fitnesses),
        'min_fitness': np.min(fitnesses),
        'max_fitness': np.max(fitnesses),
        'feasible_solutions': feasible_count,
        'feasibility_rate': (feasible_count / len(population)) * 100
    }

    return stats