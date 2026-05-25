import numpy as np

def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    ratio = prize[np.newaxis, :] / distance
    distance_penalty = 1 / distance ** 1.5  # Moderately decreasing, favoring shorter distances
    proximity_penalty = 1 - (distance / maxlen) ** 1.4  # Moderately decreasing, favoring closer proximity to the end of the path
    non_linear_adjustment = (prize / np.max(prize)) ** 1.75  # Non-linear adjustment to be more sensitive to lower rewards while considering higher ones
    decay_factor = 0.96  # Base exponential decay factor
    avg_reward = np.mean(prize)  # Adaptive decay factor based on average reward
    decay_factor = max(0.8, decay_factor * (avg_reward / np.max(prize)))  # Ensure decay factor is not too low
    decayed_distance = distance * (decay_factor ** (maxlen - distance))
    return ratio * distance_penalty * proximity_penalty * non_linear_adjustment * prize[np.newaxis, :] / decayed_distance