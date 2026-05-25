import torch
import numpy as np
import inspect
from os import path

cfp = path.abspath(path.dirname(__file__))

name = 'capacitated vehicle routing'
description = 'a fleet of vehicles with limited carrying capacity must deliver goods to a set of geographically distributed customers with known demands, minimizing the total travel distance while ensuring that no vehicle exceeds its capacity. The goal is to design a heuristic function that estimates the desirability of moving between customers, to be used within an Ant Colony Optimization (ACO) algorithm.'
unit = 'units of travel distance'

N_ITERATIONS = 100  # set to 500 in testing
N_ANTS = 30
CAPACITY = 50

class ACO():

    def __init__(self,  # 0: depot
                 distances, # (n, n)
                 demand,   # (n, )
                 heuristic, # (n, n)
                 capacity,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 ):
        
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.demand = torch.tensor(demand, device=device) if not isinstance(demand, torch.Tensor) else demand
        self.capacity = capacity
        self.gen = torch.Generator()
        self.gen.manual_seed(42)
                
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device
        

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
       
            self.update_pheronome(paths, costs)

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants
        
        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths_list.append(actions.clone())
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)
            
        return torch.stack(paths_list)
        
    def pick_move(self, prev, visit_mask, capacity_mask):
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        weights = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask) # shape: (n_ants, p_size)
        actions = torch.multinomial(weights, 1, generator=self.gen).squeeze(1)
        return actions
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()    

def compute_distance_matrix(coords):
    # coords: shape (num_nodes, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # shape: (num_nodes, num_nodes, 2)
    dist_matrix = np.linalg.norm(diff, axis=-1)  # Euclidean distance
    return dist_matrix  # shape: (num_nodes, num_nodes)
    
class Environment:
    def __init__(self, seed=19970508):
        self.seed = seed

    def training_dataset(self):
        return np.load(path.join(cfp, '../', 'dataset/cvrp_aco/train50_dataset.npy'))
    
    
    def testing_dataset(self):
        res = {}
        dataset_dir_path = path.join(cfp, '../', 'dataset/cvrp_aco/')
        for n_nodes in [50, 100, 200]:
            file_name = f'test{n_nodes}'
            fp = path.join(dataset_dir_path, file_name + '_dataset.npy')
            res[file_name] = {
                'instances': np.load(fp)
            }
        return res


    def run_async(self, policy, instances):
        res = []
        for instance in instances:
            demand = instance[:, 0]
            node_positions = instance[:, 1:]
            dist_mat = compute_distance_matrix(node_positions)
            dist_mat[np.diag_indices_from(dist_mat)] = 1
            heu = None
            if len(inspect.getfullargspec(policy).args) == 4:
                heu = policy(dist_mat.copy(), node_positions.copy(), demand.copy(), CAPACITY) + 1e-9
            elif len(inspect.getfullargspec(policy).args) == 2:
                heu = policy(dist_mat.copy(), demand / CAPACITY) + 1e-9
            heu[heu < 1e-9] = 1e-9
            aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
            obj = aco.run(N_ITERATIONS)
            res.append(- obj)
        return {
            'performance': res,
        }