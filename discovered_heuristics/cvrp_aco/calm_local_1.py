import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    normalized_distance_matrix = 1 / distance_matrix
    
    demand_ratio = demands / capacity
    demand_ratio_normalized = (demand_ratio - demand_ratio.min()) / (demand_ratio.max() - demand_ratio.min())
    
    travel_cost_ratio = np.sum(distance_matrix, axis=1) / distance_matrix.sum()
    travel_cost_ratio_normalized = (travel_cost_ratio - travel_cost_ratio.min()) / (travel_cost_ratio.max() - travel_cost_ratio.min())
    
    demand_density = np.sum(demands.reshape((-1, 1)) > coordinates, axis=1) / coordinates.shape[0]
    demand_density_normalized = (demand_density - demand_density.min()) / (demand_density.max() - demand_density.min())
    
    demand_trend = np.array([demands[i] - np.mean(demands) for i in range(len(demands))])
    demand_trend_normalized = (demand_trend - demand_trend.min()) / (demand_trend.max() - demand_trend.min())
    
    demand_frequency = demands / np.sum(demands)
    demand_frequency_normalized = (demand_frequency - demand_frequency.min()) / (demand_frequency.max() - demand_frequency.min())
    
    demand_density_factor = demand_density_normalized ** 3.9

    demand_frequency_factor = demand_frequency_normalized
    
    demand_trend_factor = demand_trend_normalized ** 3.7

    demand_trend_factor = demand_trend_factor ** (demand_density_normalized + 3.0)
    
    demand_remaining_capacity = (capacity - demands) / capacity
    demand_remaining_capacity_normalized = (demand_remaining_capacity - demand_remaining_capacity.min()) / (demand_remaining_capacity.max() - demand_remaining_capacity.min())
    
    demand_density_factor = demand_density_normalized ** 3.9
    demand_frequency_factor = demand_frequency_normalized
    demand_trend_factor = demand_trend_normalized ** 3.7
    demand_trend_factor = demand_trend_factor ** (demand_density_normalized + 3.0)

    demand_remaining_capacity_factor = demand_remaining_capacity_normalized ** 3.8

    travel_cost_factor = travel_cost_ratio_normalized * 0.8

    demand_remaining_capacity_factor = demand_remaining_capacity_normalized ** 3.7

    distance_sensitive_factor = demand_density_normalized ** 3.0 * travel_cost_ratio_normalized * 0.8

    temperature = 1 / ((demand_ratio.max() + travel_cost_ratio.max()) * (demand_density_normalized.max() + demand_trend_normalized.max()))
    
    distance_sensitive_factor = demand_density_normalized ** 3.1 * distance_sensitive_factor * 0.8
    
    adjusted_distances = np.power(distance_matrix, -temperature)  # Applying temperature factor to distances

    return (normalized_distance_matrix * (1 + 
                                         demand_ratio_normalized * 0.4 + 
                                         travel_cost_ratio_normalized * 0.2 + 
                                         demand_density_factor * 0.4 + 
                                         demand_frequency_factor * 0.1 + 
                                         demand_trend_factor * 0.3)) * demand_frequency_factor * (1 + 
                                                                                                        distance_sensitive_factor) ** 3.9 * travel_cost_factor * adjusted_distances * (1 + 
                                                                                                                                                                                                         demand_remaining_capacity_factor) ** 3.8