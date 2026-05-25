import numpy as np

def enhanced_heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    # Exponential decay for immediate high Subscription nodes
    exp_ratio = np.exp(prize[np.newaxis, :] / distance - maxlen)
    
    # Logarithmic scaling for exploration
    log_ratio = np.log(prize[np.newaxis, :] + 1) / distance
    
    # Sinusoidal decay for recent nodes with a sinusoidal smoothness adjustment
    sinusoidal_penalty = 0.5 * (1 + np.sin(np.pi * distance / (maxlen + 1))) * (distance / maxlen) * maxlen
    
    # Combined ratio
    combined_ratio = exp_ratio * log_ratio * (1 - sinusoidal_penalty)
    
    # Ensure the ratio is non-negative
    combined_ratio[combined_ratio < 0] = 0
    
    return combined_ratio