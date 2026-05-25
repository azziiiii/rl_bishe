import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    num_customers = demands.shape[0]
    heuristic_values = np.zeros((num_customers, num_customers))

    total_demand = demands.sum()
    cumulative_demand = np.zeros(num_customers)

    for i in range(num_customers):
        for j in range(num_customers):
            if demands[j] <= capacity - cumulative_demand[i]:  # Ensure the customer can be served based on vehicle's cumulative capacity
                demand_proportion = demands[j] / (total_demand + 1e-8)
                distance_penalty = 1 / (distance_matrix[i, j] + 1e-8) ** 2  # Strong penalty for distance
                proximity_adjustment = 1 / (np.linalg.norm(coordinates[i] - coordinates[j]) + 1e-8)  # Influence from spatial proximity

                # Cumulative demand adjustment to reflect previously served customers
                cumulative_demand_factor = 1 - (cumulative_demand[i] / (capacity + 1e-8))  # Adjust based on previously served demands

                # Calculate the heuristic value focusing on demand, spatial proximity, and cumulative adjustments
                heuristic_values[i, j] = demand_proportion * distance_penalty * proximity_adjustment * cumulative_demand_factor

    return heuristic_values