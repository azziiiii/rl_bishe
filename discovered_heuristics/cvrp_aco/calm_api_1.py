import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    base_heuristic = 1 / (distance_matrix + 1e-5)
    demand_efficiency = demands / (distance_matrix + 1e-5)

    proximity_penalty = np.zeros_like(demands)

    # Derive a neighborhood radius based on a fixed proportion of the maximum demand in the instance
    max_demand = np.max(demands)
    neighborhood_radius = max_demand / 5.0

    for i in range(len(coordinates)):
        neighbors = np.linalg.norm(coordinates - coordinates[i], axis=1) < neighborhood_radius
        if np.sum(neighbors) > 1:
            neighbor_demands = demands[neighbors]
            average_neighbor_demand = np.mean(neighbor_demands)
            proximity_penalty[i] = 1 - (average_neighbor_demand / (demands[i] + 1e-5))

    # Apply deterministic rule for dynamic distance adjustment considering demand
    adjusted_distance_matrix = distance_matrix + demands / (np.max(demands) + 1e-5)

    # Enhanced differentiated credits based on direct demand comparison
    total_demand = np.sum(demands) + 1e-5
    differentiated_credits = demands / total_demand
    
    # Incorporating a distance influence factor that scales credits for distances
    centroid = np.mean(coordinates, axis=0)
    distances_from_centroid = np.linalg.norm(coordinates - centroid, axis=1)
    min_distance = np.min(distances_from_centroid) + 1e-5
    scaled_credits = differentiated_credits * (distances_from_centroid / min_distance)

    # Introducing service time estimation based on demand and distance
    service_time = demands / (np.max(demands) + 1e-5) + (distance_matrix / np.max(distance_matrix + 1e-5))
    adjusted_service_time = service_time / np.sum(service_time) if np.sum(service_time) > 0 else service_time

    # New component: Adaptive penalty based on travel time estimates
    estimated_travel_times = distance_matrix / (demands[:, None] + 1e-5)  # simple estimate for travel time
    adaptive_penalty = np.mean(estimated_travel_times, axis=1)
    adjusted_heuristic = base_heuristic * demand_efficiency * (1 - proximity_penalty) * scaled_credits * (1 - adjusted_service_time) * (1 - adaptive_penalty / np.max(adaptive_penalty + 1e-5))

    adjusted_heuristic /= adjusted_distance_matrix  # Incorporate dynamic adjustment

    return adjusted_heuristic / np.sum(adjusted_heuristic) if np.sum(adjusted_heuristic) > 0 else adjusted_heuristic