import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    c1, c2, c3, c4, c5, c6 = 0.4, 0.25, 0.15, 0.1, 0.1, 0.2
    scores = {}

    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)

        return_penalty = c5 * (distance_matrix[current_node][node] if node in unvisited_nodes else 0)

        # Calculate the contextual proximity adjustment
        proximity_adjustment = c6 * (1 / (1 + np.sum([distance_matrix[prev_node][node] for prev_node in unvisited_nodes if prev_node != current_node])))

        score = (
            c1 * distance_matrix[current_node][node]
            - c2 * average_distance_to_unvisited
            + c3 * std_dev_distance_to_unvisited
            - c4 * distance_matrix[destination_node][node]
            + return_penalty
            + proximity_adjustment
        )
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node