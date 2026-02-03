import csv
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np
import yaml

from core.instance_loader import load_cached_instance
from ea.evolutionary_algorithm import EvolutionaryAlgorithm
from roots import get_project_root
from utils.logging import EALogger
from utils.metrics import calculate_all_metrics


def load_config() -> Dict:
    root = get_project_root()
    config_path = root / "src" / "frameworks" / "vanilla_ea" / "config_rl.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_all_cached_instances() -> List[str]:
    root = get_project_root()
    cache_dir = root / "data" / "processed" / "cached_instances"
    instance_files = sorted(cache_dir.glob("*.pkl"))
    return [f.stem for f in instance_files]


def save_best_solution(instance_name: str, run_id: int, genotype: List[int],
                       phenotype, metrics: Dict) -> None:
    root = get_project_root()
    solution_dir = root / "experiments" / "exp_01_vanilla" / instance_name / f"run_{run_id}"
    solution_dir.mkdir(parents=True, exist_ok=True)

    solution_data = {
        'instance': instance_name,
        'run_id': run_id,
        'seed': metrics['seed'],
        'genotype': [int(x) for x in genotype],
        'routes': [[int(customer) for customer in route] for route in phenotype.routes],
        'metrics': {
            'cost': float(metrics['cost']),
            'gap_from_optimal': float(metrics['gap_from_optimal']),
            'dispatch_rounds': int(metrics['dispatch_rounds']),
            'dispatch_rounds_increase': float(metrics['dispatch_rounds_increase']),
            'avg_route_utilization': float(metrics['avg_route_utilization']),
            'is_feasible': bool(metrics['is_feasible']),
            'capacity_violations': int(metrics['capacity_violations']),
            'num_routes': int(metrics['num_routes'])
        }
    }

    solution_path = solution_dir / "best_solution.yaml"
    with open(solution_path, 'w') as f:
        yaml.dump(solution_data, f, default_flow_style=False, sort_keys=False)


def run_single_experiment(instance_name: str, config: Dict, seed: int, run_id: int) -> Dict:
    instance_data = load_cached_instance(instance_name)

    if instance_data is None:
        raise ValueError(f"Failed to load instance: {instance_name}")

    logger = EALogger(
        exp_id=f"exp_01_vanilla/{instance_name}/run_{run_id}",
        n_period=config['experiment']['n_log_period']
    )

    ea = EvolutionaryAlgorithm(
        instance_data=instance_data,
        mu=config['mu'],
        lambd=config['lambd'],
        mutation_rate=config['mutation_rate'],
        plus_selection=config['plus_selection'],
        early_stopping_patience=config['early_stopping_patience'],
        seed=seed
    )

    best_solution = ea.evolve(
        generations=config['generations'],
        logger=logger
    )

    metrics = calculate_all_metrics(best_solution, instance_data)
    metrics['instance'] = instance_name
    metrics['seed'] = seed
    metrics['run_id'] = run_id

    save_best_solution(instance_name, run_id, ea.best_individual, best_solution, metrics)

    return metrics


def process_instance(instance_name: str, config: Dict) -> List[Dict]:
    print(f"\n[Worker] Starting instance: {instance_name}")

    results = []
    for run_id in range(config['experiment']['num_runs']):
        seed = config['experiment']['seed_start'] + run_id
        print(f"[Worker] {instance_name} - Run {run_id + 1}/{config['experiment']['num_runs']} (seed={seed})")

        result = run_single_experiment(instance_name, config, seed, run_id)
        results.append(result)

        print(
            f"[Worker] {instance_name} - Run {run_id + 1} complete: Cost={result['cost']:.2f}, Gap={result['gap_from_optimal']:.2f}%")

    print(f"[Worker] Completed instance: {instance_name}")
    return results


def aggregate_results(all_results: List[Dict]) -> List[Dict]:
    instances = sorted(set(r['instance'] for r in all_results))

    aggregated = []
    for instance in instances:
        instance_results = [r for r in all_results if r['instance'] == instance]

        costs = [r['cost'] for r in instance_results]
        gaps = [r['gap_from_optimal'] for r in instance_results]
        dispatch_rounds = [r['dispatch_rounds'] for r in instance_results]
        feasible = [r['is_feasible'] for r in instance_results]

        optimal_cost = instance_results[0].get('optimal_cost', None)

        agg = {
            'instance': instance,
            'num_runs': len(instance_results),
            'best_cost': np.min(costs),
            'avg_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'worst_cost': np.max(costs),
            'optimal_cost': optimal_cost,
            'best_gap': np.min(gaps),
            'avg_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'avg_dispatch_rounds': np.mean(dispatch_rounds),
            'feasibility_rate': (sum(feasible) / len(feasible)) * 100
        }

        aggregated.append(agg)

    return aggregated


def save_summary(all_results: List[Dict], aggregated: List[Dict], config: Dict):
    root = get_project_root()
    results_dir = root / "results" / "summaries"
    results_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = results_dir / "vanilla_ea_detailed.csv"
    with open(detailed_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    summary_path = results_dir / "vanilla_ea_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        if aggregated:
            writer = csv.DictWriter(f, fieldnames=aggregated[0].keys())
            writer.writeheader()
            writer.writerows(aggregated)

    print(f"\nResults saved:")
    print(f"  Detailed: {detailed_path}")
    print(f"  Summary: {summary_path}")


def main():
    print("Starting Vanilla EA Experiment (exp_01)")
    print("=" * 60)

    config = load_config()
    print(f"\nConfiguration:")
    print(f"  mu: {config['mu']}")
    print(f"  lambda: {config['lambd']}")
    print(f"  mutation_rate: {config['mutation_rate']}")
    print(f"  generations: {config['generations']}")
    print(f"  plus_selection: {config['plus_selection']}")
    print(f"  early_stopping_patience: {config['early_stopping_patience']}")
    print(f"  runs per instance: {config['experiment']['num_runs']}")

    instances = get_all_cached_instances()
    print(f"\nFound {len(instances)} cached instances")

    num_cores = min(cpu_count(), 9)
    print(f"Using {num_cores} cores for parallel processing")
    print(f"Total runs: {len(instances) * config['experiment']['num_runs']}")

    print(f"\n{'=' * 60}")
    print("Starting parallel execution...")
    print(f"{'=' * 60}")

    process_func = partial(process_instance, config=config)

    with Pool(processes=num_cores) as pool:
        results_nested = pool.map(process_func, instances)

    all_results = [result for instance_results in results_nested for result in instance_results]

    print("\n" + "=" * 60)
    print("Aggregating results...")
    aggregated = aggregate_results(all_results)

    print("\nSummary by instance:")
    print(f"{'Instance':<20} {'Best Cost':>12} {'Avg Gap %':>12} {'Feasible %':>12}")
    print("-" * 60)
    for agg in aggregated:
        print(
            f"{agg['instance']:<20} {agg['best_cost']:>12.2f} {agg['avg_gap']:>12.2f} {agg['feasibility_rate']:>12.1f}")

    save_summary(all_results, aggregated, config)

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()