import random
from typing import List, Tuple


def ordered_crossover(p1: List[int], p2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    size: int = len(p1)
    a, b = sorted(rng.sample(range(size), 2))

    def fill_offspring(parent_a: List[int], parent_b: List[int]) -> List[int]:
        child: List[Optional[int]] = [None] * size
        child[a:b] = parent_a[a:b]

        inherited = set(child[a:b])
        remaining = [item for item in parent_b if item not in inherited]

        idx: int = b % size
        for item in remaining:
            child[idx] = item
            idx = (idx + 1) % size
        return child

    return fill_offspring(p1, p2), fill_offspring(p2, p1)