# The algorithm used for TSPLib

import numpy as np

def select_next_node(current_node: int, start_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    def anticipate_future_cost(node, nodes_remaining):
        cost_sum = distance_matrix[current_node][node]
        remaining = nodes_remaining - {node}
        focus_node = node

        while remaining:
            if len(remaining) == 1:
                last_node = remaining.pop()
                cost_sum += distance_matrix[focus_node][last_node] + distance_matrix[last_node][start_node]
                break

            potential_nodes = sorted(remaining, key=lambda n: distance_matrix[focus_node][n])[:4]
            forecasted_costs = []

            for poss_node in potential_nodes:
                direct_cost = distance_matrix[focus_node][poss_node]
                cumulative_cost = direct_cost
                future_path = remaining - {poss_node}

                scout_node = poss_node
                while future_path:
                    optimal_choice = min(future_path, key=lambda n: distance_matrix[scout_node][n])
                    cumulative_cost += distance_matrix[scout_node][optimal_choice]
                    scout_node = optimal_choice
                    future_path.remove(optimal_choice)

                cumulative_cost += distance_matrix[scout_node][start_node]
                forecasted_costs.append(0.6 * direct_cost + 0.4 * cumulative_cost)
                
            best_projection = potential_nodes[np.argmin(forecasted_costs)]
            cost_sum += distance_matrix[focus_node][best_projection]
            remaining.remove(best_projection)
            focus_node = best_projection

        return cost_sum

    primary_candidates = sorted(unvisited_nodes, key=lambda n: distance_matrix[current_node][n])[:8]
    
    # Assign differentiated credits using a refined weight function
    def calculate_weighted_score(node):
        immediate_cost = distance_matrix[current_node][node]
        future_cost = anticipate_future_cost(node, unvisited_nodes)
        
        # Calculate clustering coefficient
        connections = [distance_matrix[node][n] < np.median(distance_matrix[node]) for n in unvisited_nodes]
        clustering_coefficient = sum(connections) / max(len(unvisited_nodes), 1)

        # Calculating weighted score for differentiation
        node_score = 0.3 * immediate_cost + 0.5 * future_cost + 0.2 * clustering_coefficient
        return node_score
    
    selected_node = min(primary_candidates, key=calculate_weighted_score)
    return selected_node