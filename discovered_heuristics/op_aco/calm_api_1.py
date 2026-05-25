import numpy as np

def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    distance_scaled = np.clip(distance, 1e-5, None)  # Avoid division by zero
    reward_cost_ratio = prize / distance_scaled  # Reward-to-cost ratio

    # Score with logarithmic adjustment and normalization
    effective_score = reward_cost_ratio / np.log1p(distance_scaled)
    local_future_rewards = np.sum(prize * np.exp(-distance), axis=0)  # Future rewards
    temporal_decay_factor = np.exp(-distance_scaled)  # Adjusts future rewards based on distance
    adjusted_future_rewards = local_future_rewards * temporal_decay_factor

    # Assign unique weights based on distance
    distance_weights = 1 / (1 + distance_scaled)  # Inversely proportional to distance
    weighted_future_rewards = adjusted_future_rewards * distance_weights

    predictive_factor = weighted_future_rewards / np.max(weighted_future_rewards) if np.max(weighted_future_rewards) > 0 else 0

    # Final combined score
    final_score = effective_score * predictive_factor
    normalized_score = (final_score - final_score.min()) / (final_score.max() - final_score.min()) if final_score.max() > final_score.min() else final_score
    return normalized_score