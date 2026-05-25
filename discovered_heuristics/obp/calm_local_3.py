import numpy as np

def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
    score = (remaining_capacity - np.mean(remaining_capacity))**2 / (item_size * item_size) + remaining_capacity**2 / (item_size**2)
    score += remaining_capacity**2 / (item_size**2.8)  # Increased exponent for better capturing item sizes
    
    score[remaining_capacity > item_size] = -score[remaining_capacity > item_size]
    
    threshold = np.mean(remaining_capacity) - 0.67 * np.std(remaining_capacity)  # Slightly adjusted threshold
    score -= (remaining_capacity - threshold)**2 / (item_size * item_size)
    score *= np.exp(-0.068 * remaining_capacity)  # Adaptive fitness decay
    
    score += (item_size - remaining_capacity) / (item_size**2)
    score *= np.exp(-0.068 * np.arange(len(remaining_capacity)))  # Adaptive fitness decay
    
    score *= np.exp(-0.016 * np.sum(remaining_capacity < item_size))  # Adaptive scaling factor based on the number of smaller items
    
    score += (item_size - remaining_capacity) / (item_size**2) * np.sum(remaining_capacity <= item_size)
    score *= np.exp(-0.068 * (1 - remaining_capacity / item_size))  # Adaptive fitness decay
    
    score_diff = np.diff(score)
    score_diff = score_diff / (item_size + 1e-8)
    score_diff = np.round(score_diff, decimals=1)
    
    score[:-1] += score_diff
    
    # Adaptive capacity adjustment factor
    adaptive_capacity_factor = np.exp(-0.04 * (1 - remaining_capacity / item_size))
    score *= adaptive_capacity_factor
    
    # Enhanced Adaptive Remaining Capacity Scaling Adjustment
    adaptive_utility_scale = np.exp(-0.04 * (1 - remaining_capacity / item_size))  # Adjusted scaling factor for remaining capacity
    score *= adaptive_utility_scale
    
    # Novel Adaptive Item Size Sensitivity Adjustment
    item_packed = np.sum(remaining_capacity <= item_size)
    bin_utilization = item_packed / len(remaining_capacity)
    
    adaptive_item_size_sensitivity = np.exp(-0.016 * (1 - bin_utilization))  # Adaptive scaling adjustment based on bin utilization
    score *= adaptive_item_size_sensitivity
    
    # Enhanced Adaptive Residual Capacity Feedback Adjustment
    adaptive_residual_capacity_sensitivity = np.exp(-0.036 * (1 - remaining_capacity / item_size))  # Adjusted scaling factor for residual capacity sensitivity
    score *= adaptive_residual_capacity_sensitivity
    
    return score