import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    scores = {}

    for node in unvisited_nodes:
        distances_to_unvisited = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance = np.mean(distances_to_unvisited)
        std_dev_distance = np.std(distances_to_unvisited)

        remaining_distance = np.sum([distance_matrix[node][i] for i in unvisited_nodes])
        urgency_factor = 1 / (1 + np.exp(remaining_distance - np.mean([distance_matrix[current_node][i] for i in unvisited_nodes])))

        score = (
            distance_matrix[current_node][node]
            - 0.5 * average_distance
            + 0.3 * std_dev_distance
            - 0.2 * distance_matrix[destination_node][node]
            + 0.4 * urgency_factor
        )
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node