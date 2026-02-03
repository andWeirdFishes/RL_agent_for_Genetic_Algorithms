import csv
import yaml
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Any

from core.instance_loader import load_cached_instance
from ea.evolutionary_algorithm import EvolutionaryAlgorithm
from frameworks.rl_param_optimizer.agent import RLAgent
from roots import get_project_root
from utils.logging import EALogger
from utils.metrics import calculate_all_metrics


def load_config() -> Dict[str, Any]:
    root = get_project_root()
    config_path = root / "src" / "frameworks" / "rl_param_optimizer" / "config_rl.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_all_cached_instances() -> List[str]:
    root = get_project_root()
    cache_dir = root / "data" / "processed" / "cached_instances"
    return [f.stem for f in sorted(cache_dir.glob("*.pkl"))]


def save_best_solution(instance_name: str, run_id: int, genotype: List[int],
                       phenotype: Any, metrics: Dict[str, Any]) -> None:
    root = get_project_root()
    solution_dir = root / "experiments" / "exp_02_rl_tuned" / instance_name / f"run_{run_id}"
    solution_dir.mkdir(parents=True, exist_ok=True)

    solution_data = {
        'instance': instance_name,
        'run_id': run_id,
        'agent_controlled': True,
        'genotype': [int(x) for x in genotype],
        'routes': [[int(c) for c in r] for r in phenotype.routes],
        'metrics': metrics
    }

    with open(solution_dir / "best_solution.yaml", 'w') as f:
        yaml.dump(solution_data, f, default_flow_style=False, sort_keys=False)


def run_single_experiment(instance_name: str, config: Dict[str, Any], seed: int, run_id: int, model_path: Path) -> Dict[
    str, Any]:
    instance_data = load_cached_instance(instance_name)

    ea = EvolutionaryAlgorithm(
        instance_data=instance_data,
        mu=config['mu'],
        lambd=config['lambd'],
        mutation_rate=config['mutation_rate'],
        plus_selection=config['plus_selection'],
        seed=seed
    )

    agent = RLAgent(env=None, model_path=model_path)
    logger = EALogger(exp_id=f"exp_02_rl/{instance_name}/run_{run_id}", n_period=config['experiment']['n_log_period'])

    total_gens = config['generations']
    obs_interval = config['n_check_gen']

    for _ in range(0, total_gens, obs_interval):
        state = ea.get_state()
        action, _ = agent.model.predict(state, deterministic=True)

        ea.mutation_rate = np.clip(action[0], 0.01, 0.5)
        ea.lambd = int(np.clip(action[1], 10, 100))

        ea.evolve(generations=obs_interval, logger=logger, continue_training=True)

    best_sol = ea.get_best_solution()
    metrics = calculate_all_metrics(best_sol, instance_data)
    metrics.update({'instance': instance_name, 'seed': seed, 'run_id': run_id})

    save_best_solution(instance_name, run_id, ea.best_individual, best_sol, metrics)
    return metrics


def process_instance(instance_name: str, config: Dict[str, Any], model_path: Path) -> List[Dict[str, Any]]:
    print(f"[RL-Worker] Processing: {instance_name}")
    results = []
    for run_id in range(config['experiment']['num_runs']):
        seed = config['experiment']['seed_start'] + run_id
        results.append(run_single_experiment(instance_name, config, seed, run_id, model_path))
    return results


def main() -> None:
    root = get_project_root()
    config = load_config()
    model_path = root / "src" / "frameworks" / "rl_param_optimizer" / "trained_models" / "best_model.zip"

    if not model_path.exists():
        model_path = root / "src" / "frameworks" / "rl_param_optimizer" / "trained_models" / "rl_model_10000_steps.zip"

    instances = get_all_cached_instances()
    process_func = partial(process_instance, config=config, model_path=model_path)

    with Pool(processes=4) as pool:
        results_nested = pool.map(process_func, instances)

    all_results = [r for inst in results_nested for r in inst]
    print(f"\nExperiment 02 (RL-Tuned) Complete. Results saved to experiments/exp_02_rl_tuned/")


if __name__ == "__main__":
    main()