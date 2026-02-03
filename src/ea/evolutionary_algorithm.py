from typing import List, Dict, Optional
from core.solution_representation import VRPSolution
from ea.population import Population
from ea.selection import tournament_selection
from ea.crossover import ordered_crossover
from ea.mutation import apply_mutation
from utils.logging import EALogger


class EvolutionaryAlgorithm:
    def __init__(self,
                 instance_data: Dict,
                 mu: int = 50,
                 lambd: int = 100,
                 mutation_rate: float = 0.1,
                 plus_selection: Optional[bool] = None,
                 early_stopping_patience: Optional[int] = None,
                 seed: Optional[int] = None):
        self.instance_data = instance_data
        self.mu: int = mu
        self.lambd: int = lambd
        self.mutation_rate: float = mutation_rate
        self.plus_selection: bool = plus_selection if plus_selection is not None else (mu <= lambd)
        self.patience: Optional[int] = early_stopping_patience

        self.population: Population = Population(mu, instance_data, use_fixed_k=True, seed=seed)
        self.best_individual: List[int] = []
        self.best_fitness: float = float('-inf')
        self.no_improvement_count: int = 0

    def evolve(self, generations: int, logger: Optional['EALogger'] = None) -> VRPSolution:
        for gen in range(generations):
            offspring = self._generate_offspring()
            candidates = offspring + (self.population.individuals if self.plus_selection else [])
            self.population.individuals = self._select_next_gen(candidates)

            improved = self._update_best()

            if logger:
                logger.log_generation(gen, self.population.get_all_phenotypes(), self.instance_data)

            if self.patience is not None:
                self.no_improvement_count = 0 if improved else self.no_improvement_count + 1
                if self.no_improvement_count >= self.patience:
                    break

        return self.population.get_phenotype(self.best_individual)

    def _generate_offspring(self) -> List[List[int]]:
        offspring: List[List[int]] = []
        while len(offspring) < self.lambd:
            p1 = tournament_selection(self.population)
            p2 = tournament_selection(self.population)
            c1, c2 = ordered_crossover(p1, p2, self.population.rng)

            if self.population.rng.random() < self.mutation_rate:
                c1 = apply_mutation(c1, self.population.rng)
            if self.population.rng.random() < self.mutation_rate:
                c2 = apply_mutation(c2, self.population.rng)

            offspring.extend([c1, c2])
        return offspring[:self.lambd]

    def _select_next_gen(self, candidates: List[List[int]]) -> List[List[int]]:
        scored_candidates = [
            (self.population.evaluate_fitness(ind), ind)
            for ind in candidates
        ]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [ind for score, ind in scored_candidates[:self.mu]]

    def get_best_solution(self) -> VRPSolution:
        return self.population.get_phenotype(self.best_individual)

    def _update_best(self) -> bool:
        current_best = max(self.population.individuals, key=lambda ind: self.population.evaluate_fitness(ind))
        current_fitness = self.population.evaluate_fitness(current_best)

        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_individual = list(current_best)
            return True
        return False