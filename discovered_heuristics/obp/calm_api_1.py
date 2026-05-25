import numpy as np

def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
    avg_item_size = np.mean(item_size) if item_size > 0 else 1.0
    adaptive_factor = avg_item_size / np.maximum(remaining_capacity, 1e-10)

    fit_score = np.maximum(remaining_capacity - item_size, 0) / (remaining_capacity + 1e-10)
    fit_score[remaining_capacity < item_size] = -np.inf

    sustainability_score = (remaining_capacity - avg_item_size) ** 2
    sustainability_score[remaining_capacity < item_size] = np.inf

    historical_fit_scores = np.cumsum(fit_score)
    normalized_historical_fit_scores = historical_fit_scores / (np.max(historical_fit_scores) + 1e-10)
    
    combined_scores = (
        (0.5 * fit_score * adaptive_factor) + 
        (0.3 / (sustainability_score + 1e-10)) - 
        (0.2 * normalized_historical_fit_scores)
    )

    differentiation_factor = 1 / (1 + np.arange(len(remaining_capacity)) * 0.1)
    combined_scores *= differentiation_factor

    cumulative_fit_impact = np.cumsum(fit_score) / (np.arange(1, len(remaining_capacity) + 1) + 1)
    cumulative_fit_adjustment = np.maximum(fit_score - cumulative_fit_impact, 0)

    combined_scores += 0.4 * cumulative_fit_adjustment

    temporal_utilization_metric = np.arange(len(remaining_capacity)) / (np.maximum(remaining_capacity, 1e-10) + 1e-10)
    combined_scores *= (1 + temporal_utilization_metric)

    sequential_elasticity = np.exp(-np.arange(len(remaining_capacity)) / (np.mean(np.maximum(remaining_capacity, 1e-10)) + 1e-10))
    combined_scores *= sequential_elasticity

    distinct_scores = combined_scores * (1 + (remaining_capacity / (np.sum(remaining_capacity) + 1e-10)))

    return distinct_scores