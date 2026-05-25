import numpy as np
import itertools
import pickle
from os import path

cfp = path.abspath(path.dirname(__file__))

name = 'online bin packing'
description = 'items arrive sequentially and must be placed immediately into bins only if they fit within the remaining capacity. The objective is to minimize the number of bins used.'
unit = 'percent of the gap to the lower bound'

    
class Environment:
    def __init__(self, seed=19970508):
        self.seed = seed

    def training_dataset(self):
        res = []
        dataset_dir_path = path.join(path.abspath(path.join(cfp, '../')), 'dataset', 'obp')
        for file_name in ['weibull_5k_train1', 'weibull_5k_train2', 'weibull_1k_train1', 'weibull_1k_train2']:
            fp = path.join(dataset_dir_path, file_name + '.pickle')
            for key, ins_dict in pickle.load(open(fp, 'rb')).items():
                if key == 'l1_bound':
                    continue
                items = ins_dict['items']
                capacity = ins_dict['capacity']
                res.append({
                    'weights': items,
                    'bin_capacity': capacity,
                    'lb': np.ceil(np.sum(items) / capacity)
                })
        return res
    
    
    def testing_dataset(self):
        res = {}
        dataset_dir_path = path.join(path.abspath(path.join(cfp, '../')), 'dataset', 'obp')
        # for i, j in itertools.product((1, 5, 10), (100, 500)):
            # file_name = f'weibull_{i}k_test_{j}'
        for file_name in ['weibull_5k_train1', 'weibull_5k_train2', 'weibull_1k_train1', 'weibull_1k_train2']:
            fp = path.join(dataset_dir_path, file_name + '.pickle')
            cur_d = []
            for key, ins_dict in pickle.load(open(fp, 'rb')).items():
                if key == 'l1_bound':
                    continue
                items = ins_dict['items']
                capacity = ins_dict['capacity']
                cur_d.append({
                    'weights': items.tolist(),
                    'bin_capacity': capacity,
                    'lb': np.ceil(np.sum(items) / capacity)
                })
            res[file_name] = {
                'instances': cur_d
            }
        return res

    def run_async(self, policy, instances):
        res = []
        rewards = []
        lbs = []
        for instance in instances:
            weights = instance['weights']
            bin_capacity = instance['bin_capacity']
            lb = instance['lb']
            lbs.append(-lb)
            bins = np.zeros(len(weights)) + bin_capacity
            for weight in weights:
                valid_indices = np.nonzero(bins >= weight)[0]
                priorities = policy(float(weight), bins[valid_indices])
                selected_bin = valid_indices[np.argmax(priorities)]
                bins[selected_bin] -= weight
            n_used_bins = np.sum(bins != bin_capacity)
            res.append(-n_used_bins)
            rew_indicator = max((1.1 * lb - n_used_bins) / (1.1 * lb - lb), .0)
            rewards.append(1.01 + np.cos(np.pi + np.pi / 2 * rew_indicator))
        return {
            'performance': res,
            'opt': lbs,
            'reward': rewards
        }