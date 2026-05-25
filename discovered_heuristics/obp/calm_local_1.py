import numpy as np

def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
    max_bin_cap = max(remaining_capacity)
    bin_density = np.sum(remaining_capacity) / (item_size * len(remaining_capacity))
    log_adj = np.log(remaining_capacity + 1) / np.log(max_bin_cap + 1)
    score = (remaining_capacity - max_bin_cap)**2 / item_size + remaining_capacity**2 / (item_size**2) + remaining_capacity**2 / (item_size**3) + bin_density * remaining_capacity
    
    score[remaining_capacity > item_size] = -score[remaining_capacity > item_size]
    score[1:] -= score[:-1]
    
    score *= log_adj
    score += log_adj * remaining_capacity
    
    score *= log_adj
    new_component = remaining_capacity / (item_size - remaining_capacity + 1)
    score += new_component
    
    new_component = remaining_capacity * np.log(remaining_capacity + 1) / (item_size * np.log(max_bin_cap + 1)) * (1 - remaining_capacity / item_size)
    score += new_component
    
    new_adjustment = (remaining_capacity / item_size) * log_adj
    score += new_adjustment
    
    remaining_capacity_adjusted = remaining_capacity / item_size
    score += np.log(remaining_capacity_adjusted + 1) / np.log(max_bin_cap + 1)
    
    new_component = (remaining_capacity - 1) / (item_size - remaining_capacity + 1) * log_adj / np.log(max_bin_cap + 1)
    score += new_component
    
    new_component = log_adj * remaining_capacity / (item_size - remaining_capacity)
    score += new_component
    
    new_component = remaining_capacity * np.log(remaining_capacity + 1) / (item_size**2) * (1 - remaining_capacity / item_size)
    score += new_component
    
    return score