from typing import List

from ea.population import Population


def tournament_selection(population: Population,
                         tournament_size: int = 3) -> List[int]:
    """
    Selects a winner from a random subset of the population based on fitness.
    """
    participants = population.rng.sample(population.individuals, tournament_size)
    return max(participants, key=lambda ind: population.evaluate_fitness(ind))