# The idea of the algorithm is to return the ratio of the prize and the inter-node distance

import numpy as np

def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return prize[np.newaxis, :] / distance