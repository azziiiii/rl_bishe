import os
import re
from os import path
import ray
import json
import importlib
import numpy as np
import time
import argparse

from utils import extract_function_from_string


def extract_step_from_filename(filepath):
    """
    Extracts the Step number from a filename using patterns like 'Step=123' or 'S123_'.
    Returns 0 if no step found.
    """
    fname = path.basename(filepath)
    m = re.search(r"Step=(\d+)", fname)
    if m:
        return int(m.group(1))
    m2 = re.search(r"S(\d+)_", fname)
    if m2:
        return int(m2.group(1))
    return 0


def ray_get_with_timeout(object_refs, timeout=60):
    start_time = time.time()
    remaining_refs = list(object_refs)
    ready_refs = []

    while remaining_refs:
        elapsed = time.time() - start_time
        time_left = timeout - elapsed
        if time_left <= 0:
            break
        ready, not_ready = ray.wait(remaining_refs, timeout=time_left, num_returns=len(remaining_refs))
        ready_refs.extend(ready)
        remaining_refs = not_ready

    if remaining_refs:
        for ref in remaining_refs:
            try:
                ray.cancel(ref, force=True)
            except Exception as e:
                print(f"Warning: Failed to cancel task {ref}: {e}")
        raise TimeoutError(f"Ray tasks timed out after {timeout} seconds.")

    return ray.get(ready_refs)


class HeuristicPolicy:
    def __init__(self, step_func, name, problem_name):
        if isinstance(step_func, str):
            if path.exists(step_func):
                if step_func.endswith('.py'):
                    step_func_code = open(step_func, 'r').read()
                elif step_func.endswith('.json'):
                    step_func_code = json.loads(open(step_func, 'r').read())['code']
                self.code = open(step_func, 'r').read()
            else:
                step_func_code = step_func
            self.code = step_func_code
            self.step_func = extract_function_from_string(step_func_code)
        else:
            self.step_func = step_func
        self.name = name
        self.perf = None
        self.perfs = None
        self.idea = None
        self.parent_prompt_type = 'seed'
        self.last_used_epoch = 0
        self.birth = 0
        self.problem_name = problem_name
        self.response = ''
        self.env = importlib.import_module(f'problems.{problem_name}').Environment()

    def dumps(self):
        return {
            'name': self.name,
            'perf': self.perf,
            'idea': self.idea,
            'birth': self.birth,
            'code': self.code,
            'problem_name': self.problem_name
        }
    
    @property
    def perf_str(self):
        return str(np.floor(1000*abs(self.perf))/1000)
    
    @property
    def sid(self):
        assert self.perf is not None
        return f"""{self.parent_prompt_type}(Perf={str(np.floor(1000*abs(self.perf))/1000)}, Step={self.birth}))"""

    def run_one_episode_sync(self, instances=None, timeout=60):
        if instances is None:
            instances = self.env.validation_dataset
        results = ray_get_with_timeout(
            [env_runner.remote(self.env, self.step_func, instance) for instance in instances], timeout=timeout
        )
        perfs, opts, rewards = [], [], []
        for result in results:
            perfs.append(result['performance'])
            if 'opt' in result: opts.append(result['opt'])
            if 'reward' in result: rewards.append(result['reward'])
        perfs = np.ravel(perfs)
        ret = {'performance_raw': perfs.copy()}
        if opts:
            opts = np.ravel(opts)
            ret['opt'] = opts
            perfs = -100 * (opts - perfs) / np.abs(opts)
        ret['performance'] = perfs
        if rewards:
            ret['reward'] = np.ravel(rewards)
        return ret

    def test(self, datasets=None):
        if datasets is None:
            datasets = self.env.testing_dataset()
        perfs = []
        for d_name, d in datasets.items():
            inst = d['instances']
            res = self.run_one_episode_sync(inst, timeout=90)
            raw = res['performance_raw']
            if 'opt' in res or 'opt' in d:
                opts = res.get('opt', d.get('opt'))
                perf = 100 * (opts.sum() - raw.sum()) / abs(opts.sum())
            else:
                perf = np.mean(raw)
            print(d_name, perf)
            perfs.append(perf)
        print('mean:', np.mean(perfs))
        return perfs
        
    def __eq__(self, other):
        return isinstance(other, HeuristicPolicy) and abs(self.perf - other.perf) < 1e-7
    
    def __hash__(self):
        return hash(self.code)

@ray.remote

def env_runner(env, step_func, instance):
    return env.run_async(step_func, [instance])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run HeuristicPolicy with specified parameters.")
    parser.add_argument('algo', type=str, nargs='+', help='Path(s) to algorithm file(s) or directory(ies)')
    parser.add_argument('--problem', type=str, default=None, help='Name of the problem domain (inferred if not provided)')
    parser.add_argument('--name', type=str, default='our_algo', help='Name of the algorithm (default: our_algo)')
    parser.add_argument('--last-n', type=int, default=None, help='Only run the last N files per directory by step')
    args = parser.parse_args()

    # Infer problem if not given
    problem = args.problem
    if problem is None:
        first = args.algo[0]
        parts = first.split(os.sep)
        if 'calm_saved' in parts:
            idx = parts.index('calm_saved')
            if idx + 1 < len(parts):
                problem = parts[idx+1]
    if problem is None:
        raise ValueError("Could not infer problem domain. Please provide --problem.")

    # Collect files and atomic dirs
    file_algos = []
    atomic_dirs_map = {}
    for p in args.algo:
        if path.isfile(p) and (p.endswith('.py') or p.endswith('.json')):
            file_algos.append(p)
        elif path.isdir(p):
            for root, dirs, files in os.walk(p):
                py = [path.join(root, f) for f in files if (f.endswith('.py') or f.endswith('.json'))]
                if py and not dirs:
                    atomic_dirs_map[root] = py
                    file_algos.extend(py)
        else:
            raise ValueError(f"Provided path '{p}' is not valid .py file or directory.")

    # Filter to last N per directory if requested
    if args.last_n is not None:
        grouped = {}
        for f in file_algos:
            d = path.dirname(f)
            grouped.setdefault(d, []).append(f)
        selected = []
        for d, flist in grouped.items():
            sorted_files = sorted(flist, key=lambda x: extract_step_from_filename(x), reverse=True)
            selected.extend(sorted_files[:args.last_n])
        file_algos = selected

    # Load environment and datasets
    env = importlib.import_module(f'problems.{problem}').Environment()
    datasets = env.testing_dataset()
    print(f"Dataset loaded for problem '{problem}'")

    file_perf_map = {}

    # Evaluate each file
    for algo in file_algos:
        name = path.basename(algo)
        dir_ = path.dirname(algo)
        policy = HeuristicPolicy(algo, args.name, problem)
        try:
            perfs = policy.test(datasets=datasets)
            mperf = float(np.mean(perfs))
            file_perf_map[algo] = mperf
            print(f"<<< {name} >>>")
            for k, v in zip(datasets.keys(), perfs): print(f"  {k}: {v}")
            print(f"  mean: {mperf:.4f}")
            print(f"  [from {dir_}]\n")
        except Exception as e:
            print(f"algo {algo} raised error during test, skip ({e})")

    # Report atomic-directory summaries
    if atomic_dirs_map:
        print("Atomic-directory summaries:")
        for atomic_dir, files in atomic_dirs_map.items():
            best = max(files, key=lambda f: file_perf_map.get(f, -np.inf))
            best_val = file_perf_map.get(best)
            if best_val is not None:
                print(f"  Directory '{atomic_dir}': best file '{path.basename(best)}' with mean perf {best_val:.4f}")
            else:
                print(f"  Directory '{atomic_dir}': no successful evaluations.")
