import os
from pathlib import Path
from typing import List
from src.core.instance_loader import load_and_cache_instance
from src.roots import get_project_root


def cache_all_instances(directory_name: str = "A") -> None:
    """
    Identifies all .vrp files in the specified raw data directory and caches them.
    """
    root = get_project_root()
    raw_data_dir = root / "data" / "raw" / directory_name

    if not raw_data_dir.exists():
        print(f"Error: Directory {raw_data_dir} does not exist.")
        return

    vrp_files: List[Path] = list(raw_data_dir.glob("*.vrp"))

    print(f"Found {len(vrp_files)} instances in {raw_data_dir}. Starting caching...")

    for vrp_path in vrp_files:
        sol_path = vrp_path.with_suffix(".sol")

        actual_sol = str(sol_path) if sol_path.exists() else None

        try:
            load_and_cache_instance(str(vrp_path), actual_sol, force_reload=True)
            print(f"Successfully cached: {vrp_path.name}")
        except Exception as e:
            print(f"Failed to cache {vrp_path.name}: {e}")


if __name__ == "__main__":
    cache_all_instances()