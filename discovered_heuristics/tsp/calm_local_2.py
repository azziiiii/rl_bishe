import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 = 0.45, 0.25, 0.15, 0.09, 0.1, 0.025, 0.03, 0.01, 0.001, 0.01, 0.005, 0.003, 0.01, 0.002
    scores = {}

    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        average_connectivity_current = np.mean([distance_matrix[current_node][i] for i in unvisited_nodes if i != node])
        average_connectivity_destination = np.mean([distance_matrix[destination_node][i] for i in unvisited_nodes if i != node])

        # Adaptive clustering factor based on connectivity and context
        clustering_factor = c9 * (average_connectivity_current + average_connectivity_destination)

        # Adaptive clustering score that considers the connectivity of the current and destination nodes
        clustering_score = c8 * (average_connectivity_current + average_connectivity_destination + clustering_factor)

        # Adaptive node reconnection score
        reconnection_score = c7 * (average_connectivity_current + average_connectivity_destination)

        # Context-sensitive adaptive adjustment factor
        adaptive_factor = c6 * (average_connectivity_current - average_connectivity_destination)

        # Connectivity score adjustment
        connectivity_score = c5 * (average_connectivity_current + average_connectivity_destination)

        # Adaptive weights adjustment based on the context
        c1_adjusted = c1 * (1 + c6 * adaptive_factor)
        c2_adjusted = c2 * (1 - c6 * average_distance_to_unvisited)
        c3_adjusted = c3 * (1 + c6 * std_dev_distance_to_unvisited)
        c4_adjusted = c4 * (1 + c6 * adaptive_factor)

        # Path relaxation score
        path_relaxation = 0.1 * (average_connectivity_current + 0.5 * adaptive_factor)

        # Geometric spatial clustering factor based on the convex combination of the current and destination nodes' positions
        geometric_factor = c9 * (average_connectivity_current + average_connectivity_destination)

        # Geometric spatial clustering score that considers the spatial positions of the nodes in the tour
        clustering_score += c8 * (average_connectivity_current + average_connectivity_destination + clustering_factor)

        # Novel node relevance (NNR) score
        nnr_score = c11 * (distance_matrix[current_node][destination_node] - np.min(distance_matrix[node]))

        # Adaptive node reconnection factor
        reconnection_factor = c12 * (average_connectivity_current + 0.5 * reconnection_score)

        # Adaptive historical reconnection adjustment
        historical_contribution = c13 * (average_connectivity_current + 0.01)  # Adjust this based on historical data

        score = (
            c1_adjusted * distance_matrix[current_node][node]
            - c2_adjusted * average_distance_to_unvisited
            + c3_adjusted * std_dev_distance_to_unvisited
            - c4_adjusted * distance_matrix[destination_node][node]
            + connectivity_score
            + reconnection_score
            + clustering_score
            + path_relaxation
            + reconnection_factor
            + nnr_score
            + historical_contribution
        )
        scores[node] = score

    next_node = min(scores, key=scores.get)
    return next_node