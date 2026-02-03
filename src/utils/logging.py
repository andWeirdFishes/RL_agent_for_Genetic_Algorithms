import csv
from datetime import datetime
from typing import List, Dict

from core.solution_representation import VRPSolution
from roots import get_project_root
from utils.metrics import calculate_population_stats


class EALogger:
    def __init__(self, exp_id: str, n_period: int = 1):
        """

        :param exp_id:
        :param n_period:
        """
        self.n_period: int = n_period
        self.root = get_project_root()
        self.log_dir = self.root / "experiments" / exp_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.log_dir / f"log_{timestamp}.csv"
        self.headers_written: bool = False

    def log_generation(self, generation: int, phenotypes: List[VRPSolution], instance: Dict) -> None:
        """

        :param generation:
        :param phenotypes:
        :param instance:
        :return:
        """
        if generation % self.n_period != 0:
            return

        stats = calculate_population_stats(phenotypes, instance)
        stats['generation'] = generation

        with open(self.file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not self.headers_written:
                writer.writeheader()
                self.headers_written = True
            writer.writerow(stats)