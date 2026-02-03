import os
import yaml
from typing import Dict, Tuple
from core.solution_representation import VRPSolution
from utils.metrics import calculate_fitness
from utils.metrics import calculate_all_metrics
from core.instance_loader import load_cached_instance
from roots import get_project_root
import pandas as pd
suggested_path = os.path.join(get_project_root(), "experiments", "exp_01_vanilla")

def get_best_vrp_solutions(root_dir: str = None) -> Dict[str, VRPSolution]:
    if root_dir is None:
        root_dir = suggested_path
    ultimate_solutions: Dict[str, VRPSolution] = {}
    for inst_folder in os.listdir(root_dir):
        inst_path = os.path.join(root_dir, inst_folder)
        if not os.path.isdir(inst_path):
            continue
        best_fitness: float = -float('inf')
        best_sol_obj: VRPSolution = None
        for run_folder in os.listdir(inst_path):
            yaml_file = os.path.join(inst_path, run_folder, "best_solution.yaml")
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                current_sol = VRPSolution(routes=data['routes'], instance_data=load_cached_instance(data['instance']))
                current_fitness = calculate_fitness(current_sol, current_sol.instance_data)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_sol_obj = current_sol
        if best_sol_obj:
            ultimate_solutions[inst_folder] = best_sol_obj
    return ultimate_solutions

def get_dataframe(root_dir: str = None) -> pd.DataFrame:
    if root_dir is None:
        root_dir = suggested_path
    best_found_sols = get_best_vrp_solutions()
    optimal_sols = dict()
    df = pd.DataFrame()
    for k, v in best_found_sols.items():
        instance_data = load_cached_instance(k)
        optimal_routes = instance_data['optimal_routes']
        optimal_sols[k] = VRPSolution(routes=optimal_routes, instance_data=instance_data)
        optimal_metrics = calculate_all_metrics(optimal_sols[k], instance_data)
        optimal_metrics['fitness'] = calculate_fitness(optimal_sols[k], instance_data)
        found_metrics = calculate_all_metrics(best_found_sols[k], instance_data)
        found_metrics['fitness'] = calculate_fitness(best_found_sols[k], instance_data)
        new_row = dict()
        new_row['instance'] = k
        for kk, vv in optimal_metrics.items():
            new_row[f'optimal_{kk}'] = vv
        for kk, vv in found_metrics.items():
            new_row[f'best_found_{kk}'] = vv
        new_row = pd.DataFrame([new_row])
        df = pd.concat([df, new_row])
    return df

def get_optimal_and_best_found(root_dir: str = None) -> Tuple[Dict[str, Tuple[VRPSolution]],Dict[str, Tuple[VRPSolution]]]:
    if root_dir is None:
        root_dir = suggested_path
    best_found_sols = get_best_vrp_solutions()
    optimal_sols = dict()
    for k, v in best_found_sols.items():
        instance_data = load_cached_instance(k)
        optimal_routes = instance_data['optimal_routes']
        optimal_sols[k] = VRPSolution(routes=optimal_routes, instance_data=instance_data)
    return optimal_sols, best_found_sols