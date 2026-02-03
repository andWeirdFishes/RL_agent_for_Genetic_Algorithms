import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy


class AdaptiveEAEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self,
                 ea_class,
                 instance_pool: List[Dict],
                 base_mu: int = 800,
                 base_generations: int = 1000,
                 n_check_gen: int = 20,
                 plus_selection: bool = True,
                 early_stopping_patience: Optional[int] = 50,
                 seed: Optional[int] = None):

        super().__init__()

        self.ea_class = ea_class
        self.instance_pool = instance_pool
        self.base_mu = base_mu
        self.base_generations = base_generations
        self.n_check_gen = n_check_gen
        self.plus_selection = plus_selection
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        self.mutation_rate_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.lambd_options = [200, 300, 400, 500, 600]

        self.action_space = spaces.Discrete(len(self.mutation_rate_options) * len(self.lambd_options))

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.current_instance = None
        self.ea = None
        self.current_gen = 0
        self.last_best_fitness = 0.0
        self.last_avg_fitness = 0.0
        self.last_diversity = 0.0
        self.fitness_history = []
        self.current_mutation_rate = 0.3
        self.current_lambd = 400
        self.episode_step = 0
        self.rng = np.random.RandomState(seed)

    def _action_to_params(self, action: int) -> Tuple[float, int]:
        mutation_idx = action // len(self.lambd_options)
        lambd_idx = action % len(self.lambd_options)
        return self.mutation_rate_options[mutation_idx], self.lambd_options[lambd_idx]

    def _get_state(self) -> np.ndarray:
        norm_best = np.clip(self.last_best_fitness / 100.0, 0.0, 1.0)
        norm_avg = np.clip(self.last_avg_fitness / 100.0, 0.0, 1.0)

        if len(self.fitness_history) >= 2:
            improvement = (self.fitness_history[-1] - self.fitness_history[-2]) / 100.0
            recent_improvement = np.clip(improvement, -1.0, 1.0)
        else:
            recent_improvement = 0.0

        state = np.array([
            norm_best,
            norm_avg,
            recent_improvement,
            self.last_diversity,
            self.current_gen / self.base_generations,
            self.current_mutation_rate,
            self.current_lambd / 600.0,
            (self.current_gen % self.n_check_gen) / self.n_check_gen
        ], dtype=np.float32)
        return state

    def _calculate_diversity(self, population_individuals: List[Any]) -> float:
        if len(population_individuals) <= 1:
            return 0.0
        fitnesses = [self.ea.population.evaluate_fitness(ind) for ind in population_individuals]
        diversity = np.std(fitnesses)
        return float(min(diversity / 50.0, 1.0))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        instance_idx = self.rng.randint(0, len(self.instance_pool))
        self.current_instance = self.instance_pool[instance_idx]
        episode_seed = self.rng.randint(0, 1000000)

        self.current_mutation_rate = 0.3
        self.current_lambd = 400

        self.ea = self.ea_class(
            instance_data=self.current_instance,
            mu=self.base_mu,
            lambd=self.current_lambd,
            mutation_rate=self.current_mutation_rate,
            plus_selection=self.plus_selection,
            early_stopping_patience=self.early_stopping_patience,
            seed=episode_seed
        )

        self.current_gen = 0
        self.last_best_fitness = 0.0
        self.last_avg_fitness = 0.0
        self.last_diversity = 0.0
        self.fitness_history = []
        self.episode_step = 0

        return self._get_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        new_mut, new_lambd = self._action_to_params(action)
        self.ea.mutation_rate = new_mut
        self.ea.lambd = new_lambd
        self.current_mutation_rate = new_mut
        self.current_lambd = new_lambd

        gens_to_run = min(self.n_check_gen, self.base_generations - self.current_gen)

        for _ in range(gens_to_run):
            offspring = self.ea._generate_offspring()
            candidates = offspring + (self.ea.population.individuals if self.ea.plus_selection else [])
            self.ea.population.individuals = self.ea._select_next_gen(candidates)
            self.ea._update_best()
            self.current_gen += 1
            if self.ea.patience is not None and self.ea.no_improvement_count >= self.ea.patience:
                break

        pop = self.ea.population.individuals
        phenotypes = self.ea.population.get_all_phenotypes()
        fitnesses = [self.ea.population.evaluate_fitness(ind) for ind in pop]

        current_best = max(fitnesses)
        current_avg = np.mean(fitnesses)
        current_diversity = self._calculate_diversity(pop)
        feasible_ratio = sum(1 for p in phenotypes if p.is_feasible) / len(phenotypes)

        remaining_gap = 100.0 - self.last_best_fitness
        improvement = current_best - self.last_best_fitness

        reward = 0.0
        if remaining_gap > 0:
            reward += (improvement / remaining_gap) * 5.0

        reward += (current_best / 100.0)

        if current_best >= 100.0:
            reward += 10.0
            done = True
        else:
            done = False

        reward -= 0.05
        if feasible_ratio < 0.5: reward -= 1.0
        if current_diversity < 0.05: reward -= 0.5

        self.fitness_history.append(current_best)
        self.last_best_fitness = current_best
        self.last_avg_fitness = current_avg
        self.last_diversity = current_diversity

        terminated = (self.current_gen >= self.base_generations) or done
        if self.ea.patience is not None and self.ea.no_improvement_count >= self.ea.patience:
            terminated = True

        info = {
            'gen': self.current_gen,
            'fit': current_best,
            'div': current_diversity,
            'mut': new_mut,
            'inst': self.current_instance.get('name', 'none')
        }

        self.episode_step += 1
        return self._get_state(), float(reward), terminated, False, info

    def render(self):
        pass

    def close(self):
        pass