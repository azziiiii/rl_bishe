import numpy as np
from os import path
from scipy.spatial.distance import cdist

name = 'traveling salesman'
description = 'the algorithm must find a tour that visits each node exactly once and returns to the start node. The objective is to minimize the length of the tour.'
unit = 'length units of the tour'

cfp = path.abspath(path.dirname(__file__))

def compute_distance_matrices(coords):
    # coords: shape (batch_size, num_nodes, 2)
    batch_size = coords.shape[0]
    dist_matrices = np.empty((batch_size, coords.shape[1], coords.shape[1]))
    for i in range(batch_size):
        dist_matrices[i] = cdist(coords[i], coords[i], metric='euclidean')
    return dist_matrices  # shape: (batch_size, num_nodes, num_nodes)
    
class Environment:
    def __init__(self, seed=19970508):
        self.seed = seed
        self.dataset_dir = path.join(path.dirname(cfp), 'dataset', 'tsp')

    def training_dataset(self, source_type=None):
        points = np.load(path.join(self.dataset_dir, 'train50_dataset.npy'))
        return compute_distance_matrices(points)
        
    def run_async(self, policy, instances, show_progress=False):
        perfs = []
        for dist_mat in instances:
            n_nodes = dist_mat.shape[0]
            visited_nodes = [0]
            tour_len = 0
            while len(visited_nodes) < n_nodes:
                unvisited_nodes = [n for n in range(n_nodes) if n not in visited_nodes]
                selected_node = policy(visited_nodes[-1], 0, set(unvisited_nodes), dist_mat.copy())
                selected_node = int(selected_node)
                if selected_node == -1:
                    selected_node = unvisited_nodes[-1]
                assert selected_node in unvisited_nodes, 'Node must be selected from unvisited ones'

                visited_nodes.append(selected_node)
                tour_len += dist_mat[visited_nodes[-1], visited_nodes[-2]]
            tour_len += dist_mat[visited_nodes[-1], visited_nodes[0]]
            perfs.append(-tour_len)
        return {
            'performance': perfs
        }
    
    def testing_dataset(self):
        res = {}
        
        for N in [50, 100, 200]:
            d_name = f"test{N}"
            fp = path.join(self.dataset_dir, f"{d_name}_dataset.npy")
            res[d_name] = {
                'instances': compute_distance_matrices(np.load(fp))
            }
            print('Dataset loaded')

        return res
