# The idea of the algorithm is to score each available bin

import numpy as np

def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
    max_bin_cap = max(remaining_capacity)
    score = (remaining_capacity - max_bin_cap)**2 / item_size + remaining_capacity**2 / (item_size**2)
    score += remaining_capacity**2 / item_size**3
    score[remaining_capacity > item_size] = -score[remaining_capacity > item_size]
    score[1:] -= score[:-1]
    return score