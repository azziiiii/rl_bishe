import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    
    scores = {}
    
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        
        score = (
            0.55 * distance_matrix[current_node][node] 
            - 0.28 * average_distance_to_unvisited 
            + 0.18 * std_dev_distance_to_unvisited 
            - 0.11 * distance_matrix[destination_node][node] 
            + 0.05 * std_dev_distance_to_unvisited**3 
            - 0.05 * (average_distance_to_unvisited + std_dev_distance_to_unvisited**3) 
            + 0.012 * std_dev_distance_to_unvisited**2 
            + 0.006 * std_dev_distance_to_unvisited**3 
            + 0.0019 * std_dev_distance_to_unvisited**3 
            + 0.00025 * std_dev_distance_to_unvisited**2 
            + 0.15 * std_dev_distance_to_unvisited**2 
            + 0.08 * std_dev_distance_to_unvisited**4
        )

        # Introduce a dynamic risk-aversion term to account for the variability and risk associated with the chosen node.
        risk_aversion_factor = 1 + std_dev_distance_to_unvisited**2
        score -= 0.02 * std_dev_distance_to_unvisited * risk_aversion_factor

        # Incorporate a risk-sensitivity term to ensure that if a node has high risk, it is less likely to be chosen.
        risk_sensitivity_factor = np.sqrt(std_dev_distance_to_unvisited**2 + 1)
        score -= 0.01 * risk_sensitivity_factor

        scores[node] = score
    
    next_node = min(scores, key=scores.get)
    return next_node