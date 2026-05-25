import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    scores = {}

    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance = np.mean(all_distances)
        standard_deviation = np.std(all_distances)
        variance_of_distances = np.var([distance_matrix[current_node][i] for i in unvisited_nodes if i != node])
        entropy_of_distances = -np.sum(np.log2([distance_matrix[destination_node][i] for i in unvisited_nodes if i != node]) / len(unvisited_nodes))
        average_distance_to_destination = np.mean([distance_matrix[destination_node][i] for i in unvisited_nodes if i != node])

        score = (
            0.6 * distance_matrix[current_node][node]
            - 0.4 * average_distance
            + 0.3 * standard_deviation
            - 0.2 * entropy_of_distances
            - 0.1 * distance_matrix[destination_node][node]
            - 0.08 * variance_of_distances
            - 0.05 * average_distance_to_destination
            - 0.01 * (np.mean([distance_matrix[current_node][i] for i in unvisited_nodes]) - average_distance)
            - 0.005 * entropy_of_distances
            - 0.008 * distance_matrix[current_node][node] * distance_matrix[node][destination_node]
            - 0.006 * standard_deviation * distance_matrix[node][destination_node]
        )
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node