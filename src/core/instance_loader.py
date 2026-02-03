import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from roots import get_project_root


def parse_vrp_file(vrp_path: str) -> Dict:
    with open(vrp_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = {}
    section = None

    for line in lines:
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
            data['name'] = name
            if '-k' in name:
                k_part = name.split('-k')[1]
                num_vehicles = int(''.join(filter(str.isdigit, k_part.split('-')[0])))
                data['num_vehicles'] = num_vehicles
        elif line.startswith('CAPACITY'):
            data['capacity'] = int(line.split(':')[1].strip())
        elif line.startswith('DIMENSION'):
            data['dimension'] = int(line.split(':')[1].strip())
        elif 'NODE_COORD_SECTION' in line:
            section = 'coords'
            data['coordinates'] = []
        elif 'DEMAND_SECTION' in line:
            section = 'demands'
            data['demands'] = []
        elif 'DEPOT_SECTION' in line:
            section = 'depot'
        elif 'EOF' in line or line == '-1':
            break
        elif section == 'coords' and line:
            parts = line.split()
            if len(parts) >= 3:
                idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                data['coordinates'].append((x, y))
        elif section == 'demands' and line:
            parts = line.split()
            if len(parts) >= 2:
                idx, demand = int(parts[0]), int(parts[1])
                data['demands'].append(demand)

    coords_array = np.array(data['coordinates'])
    demands_array = np.array(data['demands'])

    n = len(coords_array)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(coords_array[i] - coords_array[j])

    data['coordinates'] = coords_array
    data['demands'] = demands_array
    data['distance_matrix'] = distance_matrix

    return data


def parse_sol_file(sol_path: str) -> Dict:
    with open(sol_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    routes = []
    cost = None

    for line in lines:
        if line.startswith('Route'):
            route_str = line.split(':')[1].strip()
            route = [int(x) for x in route_str.split()]
            routes.append(route)
        elif line.startswith('Cost'):
            cost = float(line.split()[1])

    return {
        'optimal_routes': routes,
        'optimal_cost': cost
    }


def load_instance(vrp_path: str, sol_path: Optional[str] = None) -> Dict:
    if not os.path.exists(vrp_path):
        raise FileNotFoundError(f"VRP file not found: {vrp_path}")

    instance_data = parse_vrp_file(vrp_path)

    if sol_path is None:
        auto_sol_path = vrp_path.replace('.vrp', '.sol')
        if os.path.exists(auto_sol_path):
            sol_path = auto_sol_path

    if sol_path:
        if not os.path.exists(sol_path):
            raise FileNotFoundError(f"SOL file not found: {sol_path}")
        sol_data = parse_sol_file(sol_path)
        instance_data.update(sol_data)

    return instance_data


def get_cache_path(instance_name: str) -> str:
    root = get_project_root()
    cache_dir = os.path.join(root, 'data', 'processed', 'cached_instances')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{instance_name}.pkl")


def cache_instance(instance_name: str, instance_data: Dict) -> None:
    cache_path = get_cache_path(instance_name)
    with open(cache_path, 'wb') as f:
        pickle.dump(instance_data, f)


def load_cached_instance(instance_name: str) -> Optional[Dict]:
    cache_path = get_cache_path(instance_name)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_and_cache_instance(vrp_path: str, sol_path: Optional[str] = None, force_reload: bool = False) -> Dict:
    instance_name = Path(vrp_path).stem

    if not force_reload:
        cached = load_cached_instance(instance_name)
        if cached is not None:
            return cached

    instance_data = load_instance(vrp_path, sol_path)
    cache_instance(instance_name, instance_data)

    return instance_data