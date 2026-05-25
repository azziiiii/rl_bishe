import numpy as np

def step(item_size: np.ndarray, remaining_capacity: np.ndarray) -> np.ndarray:
    avg_item_size = np.mean(item_size)
    max_bin_capacity = max(remaining_capacity)
    avg_packing_efficiency = np.mean(remaining_capacity / item_size)
    freq_of_overfills = np.sum(remaining_capacity > item_size) / len(remaining_capacity)
    
    # Contextual adjustment of scaling factors for each term
    quadratic_factor = (1 + (freq_of_overfills - avg_item_size / remaining_capacity))**(-2) * (avg_item_size / remaining_capacity)**2
    cubic_factor = (1 + (avg_item_size / remaining_capacity - freq_of_overfills))**(-2) * (avg_item_size / remaining_capacity)**3
    log_scaling = (1 + (1 + freq_of_overfills - avg_item_size / remaining_capacity))**(-2) * (avg_item_size / remaining_capacity)**3
    exponential_factor = (1 + (avg_item_size - remaining_capacity))**(-2) * (avg_item_size / remaining_capacity)**2
    
    # Contextual elasticity adjustment for logarithmic term
    elasticity_scale = (1 + (avg_item_size / remaining_capacity - 1) * 0.8) * (1 + (avg_item_size / remaining_capacity - freq_of_overfills) * 0.8)
    log_scaling *= elasticity_scale
    
    # Contextual adjustment of weights based on the current packing efficiency and item size distribution
    weight_quadratic = (avg_packing_efficiency - freq_of_overfills) * quadratic_factor
    weight_cubic = (freq_of_overfills - avg_packing_efficiency) * cubic_factor
    weight_log = (avg_item_size - remaining_capacity) * log_scaling
    weight_exp = (remaining_capacity - max_bin_capacity) * exponential_factor
    
    # DCASR: Diversified Contextual Adaptive Scoring Refinement
    historical_efficiency_adjustment = (1 + (avg_packing_efficiency - 1))**2
    unique_diversity_factor = 1 + (avg_item_size / remaining_capacity - 1) * 0.8
    diversity_scaling = (avg_item_size / remaining_capacity)**3 * unique_diversity_factor * historical_efficiency_adjustment
    
    score = (remaining_capacity - max_bin_capacity)**2 / (item_size * avg_packing_efficiency) + \
            (remaining_capacity - avg_item_size)**3 / (item_size**3) + \
            (remaining_capacity / item_size)**log_scaling + \
            weight_quadratic * (remaining_capacity / avg_item_size)**2 / (remaining_capacity + 1) + \
            weight_cubic * (remaining_capacity / avg_item_size)**3 / (remaining_capacity + 1) - \
            weight_log * (remaining_capacity / avg_item_size)**2 / (remaining_capacity + 1) + \
            weight_exp * (remaining_capacity / avg_item_size)**2 / (remaining_capacity + 1)
    
    score[remaining_capacity > item_size] = -score[remaining_capacity > item_size]
    
    # DCASR: Contextual assignment of credits to bins
    unique_contribution = (remaining_capacity - avg_item_size) / avg_item_size
    unique_contribution *= (avg_item_size / remaining_capacity)**3 * diversity_scaling
    
    score -= unique_contribution * historical_efficiency_adjustment
    
    score = np.array([score[0]] + list(score[1:]))
    score[1:] -= score[:-1]
    
    return score