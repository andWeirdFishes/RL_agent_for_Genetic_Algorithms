import random
import numpy as np
from typing import Optional

def set_seed(seed: Optional[int] = None) -> None:
    """
    Sets the global seed for random and numpy to ensure reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def get_rng(seed: Optional[int] = None) -> random.Random:
    """
    Returns a local seeded random.Random instance for isolated stochastic behavior.
    """
    return random.Random(seed)