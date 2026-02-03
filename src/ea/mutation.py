import random
from typing import List, Tuple


def inversion_mutation(individual: List[int], rng: random.Random) -> List[int]:
    if len(individual) < 2:
        return individual
    a, b = sorted(rng.sample(range(len(individual)), 2))
    individual[a:b] = reversed(individual[a:b])
    return individual


def swap_mutation(individual: List[int], rng: random.Random) -> List[int]:
    if len(individual) < 2:
        return individual
    i, j = rng.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual


def scramble_mutation(individual: List[int], rng: random.Random) -> List[int]:
    if len(individual) < 2:
        return individual
    a, b = sorted(rng.sample(range(len(individual)), 2))
    segment = individual[a:b]
    rng.shuffle(segment)
    individual[a:b] = segment
    return individual


def displacement_mutation(individual: List[int], rng: random.Random) -> List[int]:
    if len(individual) < 3:
        return individual

    a, b = sorted(rng.sample(range(len(individual)), 2))
    if b - a < 1:
        return individual

    segment = individual[a:b]
    remaining = individual[:a] + individual[b:]

    insert_pos = rng.randint(0, len(remaining))
    individual[:] = remaining[:insert_pos] + segment + remaining[insert_pos:]

    return individual


def apply_mutation(individual: List[int],
                   rng: random.Random,
                   mutation_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)) -> List[int]:
    mutations = [
        inversion_mutation,
        swap_mutation,
        scramble_mutation,
        displacement_mutation
    ]

    chosen_mutation = rng.choices(mutations, weights=mutation_weights, k=1)[0]
    return chosen_mutation(individual, rng)