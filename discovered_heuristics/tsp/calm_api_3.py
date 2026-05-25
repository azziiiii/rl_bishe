import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    threshold = 0.7
    c1, c2, c3, c4, c5 = 0.4, 0.3, 0.2, 0.1, 0.1
    scores = {}

    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)

        # New component: consider dynamic path optimization feedback
        feedback_paths = [distance_matrix[i][node] for i in range(len(distance_matrix)) if i not in unvisited_nodes and distance_matrix[current_node][i] < threshold]
        average_feedback_distance = np.mean(feedback_paths) if feedback_paths else 0
        
        score = (
            c1 * distance_matrix[current_node][node]
            - c2 * average_distance_to_unvisited
            + c3 * std_dev_distance_to_unvisited
            - c4 * distance_matrix[destination_node][node]
            + c5 * average_feedback_distance
        )
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node