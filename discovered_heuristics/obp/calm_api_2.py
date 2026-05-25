import numpy as np

def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
    score = np.zeros_like(remaining_capacity)

    # Immediate fit score
    immediate_fit = remaining_capacity - item_size
    immediate_fit_score = np.where(immediate_fit >= 0, 1 / (1 + immediate_fit), 0)

    # Historical utilization factor
    historical_utilization = (remaining_capacity - item_size) / (remaining_capacity + 1e-5)
    historical_utilization_score = np.clip(historical_utilization, 0, 1)

    # Future capacity potential score
    future_capacity_potential = 1 - (remaining_capacity / np.max(remaining_capacity + 1e-5))
    future_capacity_score = np.clip(future_capacity_potential, 0, 1)

    # Differentiation factor based on item size
    size_factor = item_size / np.max(remaining_capacity + 1e-5)
    
    # Composite score calculation with differentiated credit assignment
    score = (immediate_fit_score * (0.4 - size_factor * 0.1) + 
             historical_utilization_score * (0.4 + size_factor * 0.1) + 
             future_capacity_score * 0.2)

    return score