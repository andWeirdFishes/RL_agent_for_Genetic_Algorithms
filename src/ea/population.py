import random
from typing import List, Dict, Optional
from src.core.solution_representation import VRPSolution
from src.utils.metrics import calculate_fitness
from src.utils.seeding import get_rng

class Population:
    def __init__(self,
                 size: int,
                 instance_data: Dict,
                 use_fixed_k: bool = True,
                 seed: Optional[int] = None):
        """
        Manages a population of giant tour genotypes and handles fitness evaluation.
        """
        self.size: int = size
        self.instance_data: Dict = instance_data
        self.use_fixed_k: bool = use_fixed_k
        self.rng: random.Random = get_rng(seed)
        self.num_customers: int = instance_data['dimension'] - 1
        self.num_vehicles: int = instance_data.get('num_vehicles', 1)
        self.individuals: List[List[int]] = self._initialize_population()

    def _initialize_population(self) -> List[List[int]]:
        population = []
        customer_ids = list(range(1, self.num_customers + 1))
        for _ in range(self.size):
            individual = list(customer_ids)
            self.rng.shuffle(individual)
            population.append(individual)
        return population

    def get_phenotype(self, individual: List[int]) -> VRPSolution:
        """
        Maps the giant tour genotype to a VRPSolution phenotype.
        """
        k = self.num_vehicles if self.use_fixed_k else None
        return VRPSolution.from_giant_tour(individual, self.instance_data, num_routes=k)

    def get_all_phenotypes(self) -> List[VRPSolution]:
        """
        Converts the Population object into a list of VRPSolution objects.
        """
        return [self.get_phenotype(ind) for ind in self.individuals]

    def evaluate_fitness(self, individual: List[int]) -> float:
        """
        Calculates fitness for a genotype, applying penalties for infeasibility.
        """
        solution = self.get_phenotype(individual)
        return calculate_fitness(solution, self.instance_data)