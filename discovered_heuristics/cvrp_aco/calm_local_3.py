import numpy as np

def advanced_heuristics_v7(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    capacity_prob = demands / capacity
    distance_reciprocal = 1 / distance_matrix
    proximity_factor = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=2)
    proximity_factor /= np.max(proximity_factor)  # Normalize between 0 and 1
    proximity_factor = 1 - proximity_factor  # Invert for higher penalty as proximity increases
    
    remaining_demands = capacity - demands
    future_savings = (remaining_demands[:, np.newaxis] * remaining_demands) / (distance_matrix * (remaining_demands[:, np.newaxis] + remaining_demands))
    capacity_ratio = remaining_demands / capacity
    proximity_savings = proximity_factor * capacity_ratio
    
    # Cluster-based proximity adaptive savings potential
    cluster_savings = np.zeros_like(distance_matrix)
    cluster_distance = np.sum(distance_matrix, axis=1) / np.linalg.norm(capacity_prob - 1, ord=1)
    cluster_adj_factor = (remaining_demands[:, np.newaxis] * remaining_demands * cluster_distance ** 3.5) / (distance_matrix * (remaining_demands[:, np.newaxis] + remaining_demands))
    
    # Adaptive balance factor adjusted based on remaining capacity and cluster adjustment
    balance_factor = np.min([1, 0.975 + 0.05 * capacity_prob.mean() + 0.03 * cluster_adj_factor.mean() + 0.005 * np.var(capacity_prob)])
    
    # Penalty factor that heavily penalizes nodes that are too close to each other, focusing on the proximity to the next node
    penalty_factor = proximity_factor ** 3
    
    # Combine all components
    probability = distance_reciprocal * capacity_prob * proximity_factor * future_savings * proximity_savings * cluster_adj_factor * (1 - balance_factor + proximity_savings * balance_factor) * (1 - penalty_factor) * (1 + cluster_adj_factor * 0.6)
    return probability