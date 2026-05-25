import numpy as np

def hybrid_heuristic(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    # Calculate the distance-based heuristic
    dist_heuristic = 1 / distance_matrix
    
    # Calculate the capacity-based heuristic
    capacity_heuristic = demands / capacity
    
    # Calculate remaining capacity heuristic
    remaining_capacity_heuristic = (capacity - demands) / capacity  # Higher remaining capacity is better
    
    # Calculate congestion factor (to account for high traffic areas with a penalty for travel distance)
    num_customers = len(demands)
    congestion_factor = np.ones((num_customers, num_customers))
    
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                # Calculate distance-based penalty factor
                travel_distance_penalty_factor = (capacity - demands[i]) * (distance_matrix[i, j] ** 3)
                
                # Calculate dynamic travel distance penalty adjustment based on remaining capacity and current solution state
                travel_distance_density_adjustment = np.clip((capacity - demands[i]) * (distance_matrix[i, j] ** 2.0) * np.log(1 + demands[j]) / np.log(capacity * demands[j]) * np.log(i + 1), 0.1, 100)
                
                # Calculate congestion factor as the inverse of the remaining capacity plus a penalty factor times the travel distance
                remaining_capacity_j = capacity - demands[j]
                travel_distance_ji = distance_matrix[i, j]
                
                # New congestion-based penalty factor with dynamic travel distance adjustment
                traffic_density_penalty = (travel_distance_penalty_factor + travel_distance_ji * (capacity - remaining_capacity_j) ** 2) / (remaining_capacity_j * travel_distance_ji + 1)
                
                remaining_capacity_penalty = traffic_density_penalty + travel_distance_penalty_factor * travel_distance_density_adjustment
                
                # Ensure the penalty adjustment does not affect values less than the distance penalty factor
                remaining_capacity_penalty = remaining_capacity_penalty + (travel_distance_penalty_factor * travel_distance_density_adjustment > 0.1) * 0.1
                
                congestion_factor[i, j] = remaining_capacity_j / (distance_matrix[i, j] + travel_distance_ji * remaining_capacity_penalty)
    
    # Combine the heuristics with remaining capacity and congestion
    combined_heuristic = (dist_heuristic * capacity_heuristic * remaining_capacity_heuristic) * congestion_factor
    
    # Ensure non-zero values (to avoid division by zero or invalid heuristics)
    combined_heuristic[combined_heuristic == 0] = np.finfo(float).eps
    
    return combined_heuristic