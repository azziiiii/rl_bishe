import numpy as np

def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    adjusted_distance = distance + 1e-10  # Avoid division by zero
    reward_importance = prize / (adjusted_distance * np.log(1 + adjusted_distance))

    momentum_factor = np.zeros_like(prize)
    for i in range(len(prize)):
        visited_indices = np.where(distance[i] <= maxlen)[0]
        momentum_factor[i] = np.sum(prize[visited_indices]) if visited_indices.size > 0 else 0

    distance_scaling = (maxlen - adjusted_distance) ** 2
    distance_scaling[distance_scaling < 0] = 0  # Only apply scaling for reachable distances

    # Credit differentiation: modify normalization based on individual rewards
    credit_differentiation = prize / (adjusted_distance + 1e-10)  # Scale by distance to enhance credit

    clustering_effect = np.zeros_like(prize)
    for i in range(len(prize)):
        nearby_indices = np.where(distance[i] <= 2 * maxlen)[0]  # Consider a larger neighborhood
        if nearby_indices.size > 0:
            clustering_effect[i] = np.sum(prize[nearby_indices]) / (2 * maxlen)

    final_heuristic = reward_importance * momentum_factor * distance_scaling * clustering_effect * credit_differentiation
    return final_heuristic