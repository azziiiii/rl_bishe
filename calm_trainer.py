import re
from datasets import Dataset
import openai
import traceback
import numpy as np
import torch
from args import Args as CALM_ARGS

import os
from os import path
import importlib
from heuristic import HeuristicPolicy
from utils import extract_function_from_string, dedent, extract_first_double_braced, extract_idea_description, get_code, idea_distance
import ray

cfp = path.abspath(path.dirname(__file__))

class Prompt:
    def __init__(self, prompt):
        self.prompt = prompt
        self.base_codes = get_code(prompt)
        if not isinstance(self.base_codes, list):
            self.base_codes = [self.base_codes]
        self.n_calls = 0
        self.last_used_epoch = 0
        self.trials = {}
        
        self.feasible_algo_generated = False

    @property
    def op(self):
        if self.is_injection:
            return "injection"
        if self.is_crossover:
            return "crossover"
        if self.is_simplification:
            return "simplify"
        if self.is_creation:
            return "create"
        basic_mod = 'For the following algorithm, identify '
        if (basic_mod + 'a fixed, instance-independent decision rule') in self.prompt:
            return 'replacement_ins'
        if (basic_mod + 'a key hyper-parameter expressed as either a constant literal or a stationary variable') in self.prompt:
            return 'replacement_hyp'
        if (basic_mod + 'a fragment that assigns equal or near-equal credits to multiple elements') in self.prompt:
            return 'replacement_crd'
        return 'initialization'

    @property
    def is_creation(self):
        return 'Be very creative and inventive. Generate an efficient algorithm following the template below' in self.prompt
    @property
    def is_injection(self):
        return "Inject a novel, meaningful component into the following algorithm. The component may be self-devised or inspired by ideas from other domains or problems." in self.prompt
    
    @property
    def is_crossover(self):
        return "Please generate a new algorithm that is motivated by the following algorithms but performs better on any same instance" in self.prompt

    @property
    def is_replacement(self):
        return "For the following algorithm, identify" in self.prompt
    
    @property
    def is_simplification(self):
        return "Please create a simplified and more elegant version of an algorithm by distilling and refining the core ideas" in self.prompt

    def __hash__(self):
        return hash(self.prompt)
    def __eq__(self, other):
        if isinstance(other, Prompt):
            return self.prompt == other.prompt  # Compare based on `prompt`
        elif isinstance(other, str):
            return self.prompt == other  # Compare with string directly
        return False
    
@ray.remote
def evaluate_code(algo, instances):
    try:
        return algo.run_one_episode_sync(instances)
    except:
        print(traceback.format_exc())
        return 'code_bug'


class Trainer:
    def __init__(self,
            calm_args: CALM_ARGS,
            model,
        ):
        self.problem_name = calm_args.problem_name
        self.calm_args = calm_args
        algorithm_template_fp = self.calm_args.template_fp if path.exists(self.calm_args.template_fp) else path.join(cfp, 'configs', self.problem_name, 'template.py')
        if path.exists(algorithm_template_fp):
            self.algorithm_template = open(algorithm_template_fp, 'r').read()
        self.problem = importlib.import_module(f'problems.{self.problem_name}')
        self.env = self.problem.Environment()
        self.instances = self.env.training_dataset()
        self.model = model
        # 直接从 model 获取 tokenizer
        if hasattr(self.model, 'tokenizer'):
            self.tokenizer = self.model.tokenizer
        elif hasattr(self.model, 'processing_class'):
            self.tokenizer = self.model.processing_class
        else:
            self.tokenizer = None

        # online mode: determined by model name prefix
        self.online = calm_args.model_name.lower().startswith('online')

        self.algos = None
        self.seed_algos = []
        self.age_stuck = 0
        self.global_best_perf = -float('inf')  # Track true global best across all steps

        self.n_prompts = calm_args.n_prompts
        self.population_size = calm_args.population_size

        self.save_dir = path.join(cfp, 'calm_saved', self.problem_name, self.calm_args.log_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_dir = path.join(self.save_dir, 'output.log')

        self.used_prompts = []
        self.injected_components = []
        self.messages = []
        self.log_step = 0

        self.train_epoch = 0

        # 追踪最佳算法
        self.use_dpo = calm_args.use_dpo
        if self.use_dpo and not self.online:
            import torch
            # 对于 unsloth + LoRA + vLLM，deepcopy 会复制 vLLM 状态导致 OOM
            # 解决方案：单独加载一个 4bit 的 ref model（不共享 vLLM 状态）
            from unsloth import FastLanguageModel
            try:
                self.reference_model, _ = FastLanguageModel.from_pretrained(
                    model_name=calm_args.model_name,
                    max_seq_length=4096,
                    load_in_4bit=True,  # 4bit 量化，显存占用小
                )
                for param in self.reference_model.parameters():
                    param.requires_grad = False
                self.reference_model.eval()
                # 确保 ref model 在同一设备上
                self.ref_device = next(self.model.parameters()).device
                self.reference_model = self.reference_model.to(self.ref_device)
            except Exception as e:
                print(f"Warning: Failed to load separate ref model: {e}")
                print("Fallback to using base model as reference")
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'base_model'):
                    self.reference_model = self.model.model.base_model
                elif hasattr(self.model, 'base_model'):
                    self.reference_model = self.model.base_model
                else:
                    self.reference_model = self.model
                self.reference_model.eval()
                self.ref_device = next(self.model.parameters()).device
        self.preference_pairs = []  # List of (prompt, chosen_algo, rejected_algo)

        # DPO hyper-parameters for preference pair selection
        self.n_preference_pairs = calm_args.n_preference_pairs
        self.diversity_threshold = calm_args.diversity_threshold
        self.pref_gap_min = calm_args.pref_gap_min
        self.pref_gap_max = calm_args.pref_gap_max

        self.prepare_dataset()

        if self.online:
            assert self.calm_args.api_key != '' and self.calm_args.base_url != '', 'To use API-based CALM, please provide your OpenAI API key and base URL in the config file.'
            self.client = openai.OpenAI(
                api_key=self.calm_args.api_key,
                base_url=self.calm_args.base_url,
            )
            assert self.client is not None
        else:
            # DPO 模式：不初始化 GRPOTrainer，避免不必要的设置
            if self.use_dpo:
                # DPO 模式下，设置 optimizer 用于 DPO 更新
                # 注意：不覆盖 train_dataset，它在 prepare_dataset() 中已经设置好了
                import torch.optim as optim
                # 获取可训练参数（LoRA 参数）
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = optim.AdamW(trainable_params, lr=calm_args.lr, betas=(0.9, 0.99))
    
    @property
    def best_perf(self):
        if len(self.algos) > 0:
            return np.max([a.perf for a in self.algos])
        return -float('inf')

    def log_info(self, log_msg):
        with open(self.log_dir, 'a+') as fp:
            fp.write(f"<Epoch {self.train_epoch} / Step {self.log_step}> " + str(log_msg) + '\n')
        
    def min_performance_distance(self, algo, population):
        d = np.min([abs(algo.perf - a.perf) for a in population])
        scale = abs(population[0].perf - population[-1].perf)
        return d / scale
    
    # =========================================================================
    def prepare_dataset(self):

        dataset_dict = {"prompt": []}

        if self.algos is None:
            self.algos = self.get_algos_and_performance()
            for init_algo in self.algos:
                self.seed_algos.append(init_algo)
                if init_algo.perf is not None:
                    self.global_best_perf = max(self.global_best_perf, init_algo.perf)

        # 按性能排序，保持最好的在前面
        sorted_indices = np.argsort([a.perf for a in self.algos])[::-1]
        self.algos = [self.algos[i] for i in sorted_indices]

        algos_head = self.algos[:self.population_size]

        curr_used_prompts = []

        # ---------------------------------------------------------------------------- #
        #                                Prepare dataset                               #
        # ---------------------------------------------------------------------------- #
        ub_simplification = self.calm_args.ub_simplification
        ub_injection = self.calm_args.ub_injection
        ub_replacement = self.calm_args.ub_replacement
        ub_crossover = self.calm_args.ub_crossover

        n_ops = np.zeros(4, dtype=int)
        p_ops = np.array([ub_simplification, ub_injection, ub_replacement, ub_crossover]).astype(float)
        if len(self.algos) < 2:
            p_ops[-1] = 0
        if len(self.algos) < self.population_size and ub_injection > 0:
            p_ops[1] = np.max(p_ops)
        p_ops /= np.sum(p_ops)
        sample_res = np.random.choice(4, size=self.n_prompts, p=p_ops, replace=True)
        for i_op in sample_res:
            n_ops[i_op] += 1
        ub_simplification, ub_injection, ub_replacement, ub_crossover = n_ops
        self.log_info(f'UB of OPs: simplification - {ub_simplification}, injection - {ub_injection}, replacement - {ub_replacement}, crossover - {ub_crossover}')

        rank = 1 + np.arange(len(algos_head))
        p = 1 / rank
        p /= np.sum(p)
        # ------------------ Simplification, Injection, Replacement ----------------- #
        if len(algos_head) > 0:
            for upper_bound, prompt_template in zip([ub_simplification, ub_injection, ub_replacement], [self.prompt_simplification, self.prompt_injection, self.prompt_replacement]):
                n_new_prompts = 0
                n_trial = 0
                while n_trial <= 1000 and n_new_prompts < upper_bound:
                    n_trial += 1
                    indices = [np.random.choice(len(algos_head), p=p)]
                    algos_for_prompt = [algos_head[i] for i in indices]
                    prompt = Prompt(prompt_template(algos_for_prompt))
                    if prompt not in self.used_prompts:
                        self.used_prompts.append(prompt)
                    if prompt not in curr_used_prompts:
                        curr_used_prompts.append(prompt)
                    else:
                        continue
                    dataset_dict['prompt'].append([
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': prompt.prompt}
                    ])
                    n_new_prompts += 1

            # --------------------------------- Crossover -------------------------------- #
            n_trial = 0
            n_new_crossover = 0
            while len(self.algos) >= 2 and n_trial < 1000 and n_new_crossover < ub_crossover:
                n_trial += 1
                algo_0_idx = np.random.choice(len(algos_head), p=p)
                algo_0 = algos_head[algo_0_idx]
                algo_1_idx = None
                log_msg = None
                if np.random.random() <= .5:
                    # Performance-based
                    algo_1_idx = np.random.choice(len(algos_head), p=p)
                    if algo_0 == algos_head[algo_1_idx]:
                        continue
                    log_msg = f"Crossover driven by performance: {algo_0.sid} (Rank {rank[algo_0_idx]}) x {self.algos[algo_1_idx].sid} (Rank {rank[algo_1_idx]})"
                else:
                    # Diversity-based
                    distances = [- idea_distance(base_idea=algo_0.idea, new_idea=algo_1.idea) for algo_1 in self.algos]
                    distance_rank = np.argsort(np.argsort(distances)) + 1
                    distance_based_p = 1 / distance_rank
                    distance_based_p /= np.sum(distance_based_p)
                    algo_1_idx = np.random.choice(len(self.algos), p=distance_based_p)
                    if distances[algo_1_idx] == 0:
                        continue
                    log_msg = f"Crossover driven by diversity: {algo_0.sid} (Rank {rank[algo_0_idx]}) x {self.algos[algo_1_idx].sid}, Distance: {distances[algo_1_idx]}"
                prompt = Prompt(self.prompt_crossover([algo_0, self.algos[algo_1_idx]]))
                if prompt not in self.used_prompts:
                    self.used_prompts.append(prompt)
                if prompt not in curr_used_prompts:
                    curr_used_prompts.append(prompt)
                else:
                    continue
                dataset_dict['prompt'].append([
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt.prompt}
                ])
                self.log_info(log_msg)
                n_new_crossover += 1

            # --------------------------------- Creation --------------------------------- #
            if len(dataset_dict['prompt']) == 0:
                self.log_info('No prompts have been added, add creation')
                prompt = self.prompt_creation
                dataset_dict['prompt'].append([
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': self.prompt_creation}
                ])
                if prompt not in self.used_prompts:
                    self.used_prompts.append(Prompt(prompt))

        self.train_dataset = Dataset.from_dict(dataset_dict)
        self.messages = dataset_dict['prompt']
        self.log_info(f"[Dataset] {len(self.messages)} prompts prepared for this step")
        self.train_epoch += 1

    def query_all(self):
        """
        Sequentially send each prompt in self.train_dataset to the LLM via self.client,
        then run self.reward_func on each reply.

        Returns:
            contents: List[str]         — the raw assistant replies
            rewards_list: List[List]    — the reward_func outputs (one list per prompt)
        """
        all_messages    = [entry['prompt'] for entry in self.train_dataset]
        all_contents    = []

        # 1) Query LLM for each prompt
        for messages in all_messages:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            all_contents.append(resp.choices[0].message.content)

        # 2) Wrap into the shape reward_func expects
        completions_batch = [[{'content': c}] for c in all_contents]

        # 3) Call reward_func once on the whole batch
        rewards = self.reward_func(all_messages, completions_batch)

        return all_contents, rewards

    # =========================================================================
    # DPO 模式：只用 vLLM 生成，不调用 GRPO 更新
    # =========================================================================
    def generate_with_vllm(self):
        """
        用 vLLM 生成算法并评估，只生成不更新模型。
        用于 DPO 模式：生成 → 评估 → 构建偏好对 → DPO 更新
        
        流程：
        1. 用 vLLM 生成多个候选算法
        2. 评估每个算法的性能
        3. 保存到 self.algos（用于构建偏好对）
        """
        import torch
        from transformers import PreTrainedModel
        
        # 获取模型（可能是 unsloth wrapper）
        model_to_use = self.model
        if hasattr(self.model, 'model'):
            model_to_use = self.model.model
        
        # 检查是否有 vLLM
        if hasattr(model_to_use, 'generate'):
            # 尝试用 vLLM 批量生成
            try:
                all_contents = self._generate_batch_vllm(model_to_use)
            except Exception as e:
                print(f"vLLM generation failed: {e}, falling back to sequential")
                all_contents = self._generate_sequential(model_to_use)
        else:
            all_contents = self._generate_sequential(model_to_use)
        
        # 包装成 reward_func 需要的格式
        completions_batch = [[{'content': c}] for c in all_contents]
        all_messages = [entry['prompt'] for entry in self.train_dataset]
        
        # 调用 reward_func 评估算法
        rewards = self.reward_func(all_messages, completions_batch)
        
        return all_contents, rewards
    
    def _generate_batch_vllm(self, model):
        """用 vLLM 批量生成，每个prompt生成n_generations次"""
        all_messages = [entry['prompt'] for entry in self.train_dataset]
        all_contents = []
        
        # 获取 tokenizer
        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = getattr(model, 'tokenizer', None)
        
        if tokenizer is None:
            raise ValueError("No tokenizer found")
        
        n_generations = self.calm_args.n_generations
        self.log_info(f"[Generate] Generating {n_generations} candidates per prompt, total {len(all_messages)} prompts")
        
        # 批量生成，每个prompt生成n_generations次
        for messages in all_messages:
            for gen_idx in range(n_generations):
                try:
                    # 构建输入
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # 生成
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    # 解码
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # 去掉输入部分，只保留生成的内容
                    input_len = inputs['input_ids'].shape[1]
                    content = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                    all_contents.append(content)
                    
                except Exception as e:
                    print(f"Generation error: {e}")
                    all_contents.append("")
        
        self.log_info(f"[Generate] Total generated: {len(all_contents)} candidates")
        return all_contents
    
    def _generate_sequential(self, model):
        """顺序生成（备选方案）"""
        return self._generate_batch_vllm(model)  # 复用上面的逻辑
    
    def reward_func(self, prompts, completions, **kwargs):
        self.log_step += 1
        if len(self.algos) >= self.population_size:
            self.age_stuck += 1
            self.log_info(f'Stuck counter += 1, arrives at {self.age_stuck}')
        self.log_info(f"=== Step {self.log_step} (Epoch {self.train_epoch}) ===")
        self.log_info(f"[LLM] Generated {len(completions)} candidates this step")
        
        res = []
        mean_perfs = []
        average_idea_word_count = []
        curr_algos = []
        curr_prompts = []
        curr_responses = []
        curr_algos_reward_idx = []
        
        # 统计过滤原因
        filter_stats = {
            'idea_not_exist': 0,
            'code_not_exist': 0,
            'function_not_exist': 0,
            'bug_in_function': 0,
            'random_algorithm': 0,
            'success': 0
        }
        
        for curr_algo_i, (prompt, completion) in enumerate(zip(prompts, completions)):
            prompt = prompt[-1]['content']
            prompt_idx = self.used_prompts.index(prompt)
            prompt = self.used_prompts[prompt_idx]
            curr_prompts.append(prompt)
            algo_name = f'creation({self.log_step})'
            
            response = completion[0]['content']
            curr_responses.append(response)
            idea = extract_first_double_braced(response)
            if idea is None:
                idea = extract_idea_description(response)
            idea_start = 'The idea of the algorithm is to'
            if idea is None or not idea.startswith(idea_start) or len(idea) - len(idea_start) <= 10:
                filter_stats['idea_not_exist'] += 1
                res.append(self.calm_args.reward_idea_not_exist)
                continue

            code = get_code(response)
            if code is None:
                # Reward for not enclosing the code in a Python block
                filter_stats['code_not_exist'] += 1
                res.append(self.calm_args.reward_code_not_exist)
                continue

            step_func = extract_function_from_string(code)
            if step_func is None:
                # Reward for no function found
                filter_stats['function_not_exist'] += 1
                res.append(self.calm_args.reward_function_not_exist)
                continue
            
            algo = HeuristicPolicy(step_func, name=algo_name, problem_name=self.problem_name)
            algo.code = code
            algo.idea = idea
            algo.name = algo_name
            algo.birth = self.log_step
            algo.response = response
            algo.parent_prompt_type = prompt.op
            res.append(None)
            curr_algos.append(algo)
            curr_algos_reward_idx.append(curr_algo_i)

        curr_algo_running_results = ray.get([evaluate_code.remote(algo, self.instances) for algo in curr_algos])
        for algo, curr_algo_i, running_res in zip(curr_algos, curr_algos_reward_idx, curr_algo_running_results):
            code = algo.code
            is_feasible = True
            
            if isinstance(running_res, str):
                # Running error - 不保存到algos
                filter_stats['bug_in_function'] += 1
                res[curr_algo_i] = self.calm_args.reward_bug_in_function
                continue
            elif 'random' in code or 'np.random' in code:
                filter_stats['random_algorithm'] += 1
                res[curr_algo_i] = self.calm_args.reward_random_algorithm
                continue
            else:
                filter_stats['success'] += 1
                prompt = curr_prompts[curr_algo_i]
                perfs = np.array(running_res['performance'])
                algo.perfs = perfs.copy()
                algo.perf = np.mean(perfs)
                
                # 只保存可行的候选用于DPO偏好对构建
                algo.is_feasible = True
                self.algos.append(algo)
            
            base_algos = []
            for base_code in prompt.base_codes:
                for a in self.algos:
                    if a.code.strip() == base_code.strip():
                        base_algos.append(a)
                        break

            best_base_perf = np.max([a.perf for a in base_algos])
            is_global_best = algo.perf > self.global_best_perf
            is_better = algo.perf > best_base_perf
            if is_global_best:
                self.age_stuck = 0
                self.global_best_perf = algo.perf
                self.log_info(f"Better performance: {algo.perf} at step {self.log_step} by {algo_name}")
                self.log_info(f"Idea: {algo.idea}")

                # 只保存best算法
                os.makedirs(path.join(self.save_dir, 'algos'), exist_ok=True)
                algo_filename = f"S{self.log_step}_{algo.sid}.py"
                algo_path = path.join(self.save_dir, 'algos', algo_filename)
                with open(algo_path, 'w+') as fp:
                    fp.write(code)
                self.log_info(f"[SAVE] Saved algo to {algo_path}")

            reward = None

            average_idea_word_count.append(len(algo.idea.split()))

            if prompt.op == 'initialization':
                reward = 0.0
            else:
                delta_perf = np.clip(abs(algo.perf - best_base_perf) / min(abs(algo.perf), abs(best_base_perf)), 1e-10, 1.0)
                if is_better:
                    reward = 1.0 + delta_perf
                else:
                    if algo.perf >= best_base_perf:
                        reward = 0.0
                    else:
                        reward = self.calm_args.reward_random_algorithm / 2 * (delta_perf if algo not in base_algos else (2*.8))
                if prompt.is_injection:
                    # Injected component
                    match = re.search(r"The new component ([A-Za-z()'\- ]+?) has been introduced", curr_responses[curr_algo_i])
                    if match:
                        new_component = match.group(1).strip()
                        if new_component not in self.injected_components:
                            self.injected_components.append(new_component)
                            self.log_info(f'New component {new_component} has been introduced')

            assert reward is not None
            res[curr_algo_i] = reward

            mean_perfs.append(algo.perf)

        # 只统计有真实perf的算法
        valid_algos = [a for a in self.algos if hasattr(a, 'perf') and a.perf is not None]
        perfs = map(str, sorted([a.perf for a in valid_algos], reverse=True))
        self.log_info(f"Number of algos: {len(valid_algos)}, Perfs: {','.join(perfs)}")
        
        # 打印过滤统计
        filter_msg = f"[Filter] idea_not_exist={filter_stats['idea_not_exist']}, code_not_exist={filter_stats['code_not_exist']}, function_not_exist={filter_stats['function_not_exist']}, bug_in_function={filter_stats['bug_in_function']}, random={filter_stats['random_algorithm']}, success={filter_stats['success']}"
        self.log_info(filter_msg)
        
        return res
    
    @property
    def system_prompt(self):
        return f"""\
            Searching superior heuristics on the {self.problem.name} problem in an evolutionary manner through conversation between User and Assistant. In this problem, {self.problem.description} The User provides existing algorithms and requests a new one.\n\n{self.prompt_algo_requirements()}"""
    
    @property
    def prompt_creation(self):
        assert self.algorithm_template is not None, 'No template was provided while prompting for creation'
        return f"""Be very creative and inventive. Generate an efficient algorithm following the template below:\n\n{self.algorithm_template}"""
    
    def prompt_simplification(self, algos):
        return f"""Please create a simplified and more elegant version of an algorithm by distilling and refining the core ideas from the following:\n\n{self.prompt_algo_details(algos)}"""
    
    def prompt_injection(self, algos):
        prompt = f"""Inject a novel, meaningful component into the following algorithm. The component may be self-devised or inspired by ideas from other domains or problems.\n\n{self.prompt_algo_details(algos)}\n\nUse a concise noun phrase to describe the new component in the responded idea like "The new component ... has been introduced."."""
        if len(self.injected_components) > 0:
            prompt += f""" Exclude the following components that have already been explored: {', '.join(self.injected_components[-10:])}."""
        return prompt
    
    def prompt_replacement(self, algos):
        _MODE_SPECS = [
            ('a fixed, instance-independent decision rule', 'an instance-dependent rule that derives its value from the current observation'),
            ('a key hyper-parameter expressed as either a constant literal or a stationary variable', 'a more principled constant justified by theory or practice'),
            ('a fragment that assigns equal or near-equal credits to multiple elements', 'a fragment where credits are deterministically and reasonably differentiated')
        ]
        p1, p2 = _MODE_SPECS[np.random.choice(len(_MODE_SPECS))]
        prompt = f"""For the following algorithm, identify {p1} and rewrite it to {p2}.\n\n{self.prompt_algo_details(algos)}"""
        return prompt

    def prompt_algo_details(self, algos):
        algo_detail = ""
        sort_indices = np.argsort([a.perf for a in algos])[::-1]
        algos = [algos[i] for i in sort_indices]
        if len(algos) == 0:
            return f"""## The Algorithm\n* Performance: {algos[0].perf_str} {self.problem.unit}\n* Idea: {algos[0].idea}\n* Code: ```python\n{algo.code}```\n\n"""
        
        for i, algo in enumerate(algos):
            algo_detail += f"""## Algorithm {i+1}\n* Performance: {algo.perf_str} {self.problem.unit} (Rank: {i + 1})\n* Idea: {algo.idea}\n* Code:```python\n{algo.code}```\n\n"""
            algo.last_used_epoch = self.train_epoch
        return algo_detail.strip()
    
    def prompt_algo_requirements(self):
        return dedent("""\
            ## Your Task
            You should first present a concise conceptual description, followed by a complete code implementation.
            
            * The description must:
                * Be enclosed with a double brace and starts with "The idea of the algorithm is to".
                * Ensure it is self-contained, insightful, and creatively original.
                * Not reference or rely on any prior ideas or existing code.
            * The code must:
                * Strictly follow the input-output variable names and types used in the provided implementation.
                * Be a single Python function formatted within Python code blocks.
                * Exclude any usage examples.
                * Ensure the algorithm is deterministic.
                * Avoid introducing unnecessary, arbitrarily-tuned hyperparameters; any parameters used should be essential and systematically derived from the input.
                      
            Overall, your response should be like:
            {{The idea of the algorithm is to (sepcific description here)}}
            ```python
            your code here
            ```
            Except for the idea and code, do not give additional explanations or comments.\
        """)
    
    def prompt_crossover(self, algos):
        return f"""Please generate a new algorithm that is motivated by the following algorithms but performs better on any same instance.\n{self.prompt_algo_details(algos)}
        """
    
    def get_algos_and_performance(self):
        res = []
        
        seed_algo_fp = self.calm_args.seed_algo_fp
        if seed_algo_fp == '':
            seed_algo_fp = path.join(cfp, 'configs', self.problem_name, 'seed.py')
        else:
            seed_algo_fp = path.join(cfp, seed_algo_fp)
        if path.exists(seed_algo_fp):
            algo_name = 'seed'
            algo = HeuristicPolicy(step_func=seed_algo_fp, name=algo_name, problem_name=self.problem_name)
            perfs = algo.run_one_episode_sync(instances=self.instances)['performance']
            idea = open(seed_algo_fp, 'r').readlines()[0][1:].strip()
            if not idea.startswith('The idea of the algorithm is to'):
                idea = 'The idea of the algorithm is to solve the {self.problem_name} in some way'
            algo.perf = np.mean(perfs)
            algo.perfs = perfs.copy()
            algo.idea = idea
            res.append(algo)
        return res

    # =========================================================================
    # DPO Preference Pair Selection (DAR + Diversity)
    # =========================================================================
    def select_preference_pairs(self, candidates, prompt_context=None):
        """
        Select preference pairs using DAR (Diversity-Aware Rank-based) sampling:
        - Partition candidates into M buckets by performance rank
        - Sample chosen from higher-ranked buckets (with temperature-controlled probability)
        - Sample rejected from buckets at least 2 ranks away
        - Diversity filtering to avoid learning from similar algos
        
        This prevents:
        1. Always picking best vs worst (overfitting to extremes)
        2. Learning from nearly identical algos (waste of training signal)
        """
        if len(candidates) < 2:
            return []

        # Exclude seed algorithms to avoid anchoring
        non_seed_candidates = [c for c in candidates if 'seed' not in c.sid.lower()]
        if len(non_seed_candidates) < 2:
            self.log_info(f"[DAR] Only {len(non_seed_candidates)} non-seed candidates, skipping preference pairs")
            return []

        # Sort by performance (descending)
        candidates_sorted = sorted(non_seed_candidates, key=lambda x: x.perf, reverse=True)
        
        # Index map to avoid O(n^2) list.index lookups
        idx_of = {a.sid: i for i, a in enumerate(candidates_sorted)}
        
        # Compute diversity matrix for all pairs
        n = len(candidates_sorted)
        diversity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._calculate_diversity(candidates_sorted[i], candidates_sorted[j])
                diversity_matrix[i, j] = d
                diversity_matrix[j, i] = d
        
        # DAR: Partition into M buckets
        M = min(self.calm_args.dar_n_buckets, n)  # Number of buckets
        bucket_size = max(1, n // M)
        
        buckets = []
        for i in range(0, n, bucket_size):
            bucket = candidates_sorted[i:i + bucket_size]
            if bucket:
                buckets.append(bucket)
        
        M = len(buckets)
        
        # Temperature-controlled sampling for chosen (y+)
        # Higher-ranked buckets have higher probability (exploitation vs exploration)
        tau = self.calm_args.dar_temperature  # Temperature parameter
        bucket_probs = np.exp(np.arange(M - 2) / tau)  # Exclude last 2 buckets
        bucket_probs /= bucket_probs.sum()
        
        selected_pairs = []
        used_alg_ids = set()
        
        # Adaptive gap range based on performance distribution
        perfs = np.array([a.perf for a in candidates_sorted])
        perf_std = np.std(perfs) if len(perfs) > 1 else 1.0
        adaptive_gap_min = max(self.pref_gap_min, 0.02)  # At least 2%
        adaptive_gap_max = self.pref_gap_max
        
        max_attempts = self.n_preference_pairs * 10
        attempt = 0
        
        self.log_info(f"[DAR Debug] n_candidates={n}, M_buckets={M}, tau={tau:.2f}, "
                      f"perf_std={perf_std:.4f}, gap_range=[{adaptive_gap_min:.3f}, {adaptive_gap_max:.3f}]")
        
        while len(selected_pairs) < self.n_preference_pairs and attempt < max_attempts:
            attempt += 1
            
            # Sample chosen bucket (from top M-2 buckets)
            try:
                chosen_bucket_idx = np.random.choice(M - 2, p=bucket_probs)
            except:
                chosen_bucket_idx = 0
            chosen_bucket = buckets[chosen_bucket_idx]
            
            # Sample chosen algo from chosen bucket (prefer higher-ranked within bucket)
            if len(chosen_bucket) > 1:
                # Weight by reverse rank within bucket
                weights = np.linspace(1.0, 0.5, len(chosen_bucket))
                weights /= weights.sum()
                chosen_algo = np.random.choice(chosen_bucket, p=weights)
            else:
                chosen_algo = chosen_bucket[0]
            
            # Skip if already used too many times
            if chosen_algo.sid in used_alg_ids and len(used_alg_ids) >= self.n_preference_pairs:
                continue
            
            # Sample rejected bucket (skip adjacent buckets, at least 2 ranks away)
            valid_rejected_buckets = []
            for b_idx in range(chosen_bucket_idx + 2, M):
                valid_rejected_buckets.append(b_idx)
            
            if not valid_rejected_buckets:
                continue
            
            # Weight by diversity - prefer diverse buckets
            rejected_bucket_idx = np.random.choice(valid_rejected_buckets)
            rejected_bucket = buckets[rejected_bucket_idx]
            
            # Find best diversity match in rejected bucket
            best_pair = None
            best_combined_score = -1
            
            for rejected_algo in rejected_bucket:
                # Check diversity
                ci = idx_of.get(chosen_algo.sid)
                rj = idx_of.get(rejected_algo.sid)
                if ci is None or rj is None:
                    continue
                div = diversity_matrix[ci, rj]
                
                if div < self.diversity_threshold:
                    continue
                
                # Check performance gap
                gap = abs(chosen_algo.perf - rejected_algo.perf)
                perf_scale = max(abs(chosen_algo.perf), abs(rejected_algo.perf), 1e-6)
                relative_gap = gap / perf_scale
                
                if relative_gap < adaptive_gap_min or relative_gap > adaptive_gap_max:
                    continue
                
                # Combined score: prefer diverse + moderate gap
                combined_score = div * 0.5 + (1 - abs(relative_gap - 0.5)) * 0.5
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_pair = {
                        'better': chosen_algo,
                        'worse': rejected_algo,
                        'gap': gap,
                        'relative_gap': relative_gap,
                        'diversity': div
                    }
            
            if best_pair is not None:
                selected_pairs.append(best_pair)
                used_alg_ids.add(best_pair['better'].sid)
                used_alg_ids.add(best_pair['worse'].sid)
                self.log_info(f"[DAR] Pair: {best_pair['better'].sid}({best_pair['better'].perf:.3f}) > "
                            f"{best_pair['worse'].sid}({best_pair['worse'].perf:.3f}), "
                            f"gap={relative_gap:.4f}, div={div:.3f}")
        
        return selected_pairs[:self.n_preference_pairs]

    def _calculate_diversity(self, algo1, algo2):
        """Calculate diversity between two algorithms based on code structure and idea."""
        # Code-based diversity
        code1 = algo1.code if hasattr(algo1, 'code') else ""
        code2 = algo2.code if hasattr(algo2, 'code') else ""
        code_diversity = 1.0 - self._jaccard_similarity(code1, code2)

        # Idea-based diversity
        idea1 = algo1.idea if hasattr(algo1, 'idea') else ""
        idea2 = algo2.idea if hasattr(algo2, 'idea') else ""
        idea_diversity = 1.0 - self._jaccard_similarity(idea1, idea2)

        return (code_diversity + idea_diversity) / 2

    def _jaccard_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # DPO Dataset Preparation
    # =========================================================================
    def prepare_dpo_dataset(self, preference_pairs):
        """
        Prepare DPO dataset from selected preference pairs.
        Returns: List of (prompt, chosen_response, rejected_response) tuples
        """
        dpo_data = []
        for pair in preference_pairs:
            better_algo = pair['better']
            worse_algo = pair['worse']

            # Get the prompt that generated these algorithms - ensure it's a string
            prompt_text = ""
            if hasattr(better_algo, 'parent_prompt') and better_algo.parent_prompt:
                prompt_obj = better_algo.parent_prompt
                if isinstance(prompt_obj, str):
                    prompt_text = prompt_obj
                elif isinstance(prompt_obj, list) and len(prompt_obj) > 0:
                    prompt_text = prompt_obj[-1].get('content', '') if isinstance(prompt_obj[-1], dict) else str(prompt_obj[-1])
                else:
                    prompt_text = str(prompt_obj)
            elif hasattr(self, 'messages') and self.messages:
                last_msg = self.messages[-1]
                if isinstance(last_msg, dict):
                    prompt_text = last_msg.get('content', '')
                else:
                    prompt_text = str(last_msg)
            
            chosen_response = better_algo.response if hasattr(better_algo, 'response') else self._algo_to_response(better_algo)
            rejected_response = worse_algo.response if hasattr(worse_algo, 'response') else self._algo_to_response(worse_algo)

            dpo_data.append({
                'prompt': prompt_text,
                'chosen': chosen_response,
                'rejected': rejected_response,
                'better_perf': better_algo.perf,
                'worse_perf': worse_algo.perf,
            })

        return dpo_data

    def _algo_to_response(self, algo):
        """Convert algorithm object to response string."""
        if hasattr(algo, 'response') and algo.response:
            return algo.response
        idea = getattr(algo, 'idea', 'The idea of the algorithm is to solve the problem.')
        code = getattr(algo, 'code', 'pass')
        return f"{{The idea of the algorithm is to {idea}}}\n```python\n{code}\n```"

    # =========================================================================
    # DPO Loss Computation
    # =========================================================================
    def compute_dpo_loss(self, preference_pairs, prompts, chosen_responses, rejected_responses):
        """
        Compute DPO loss for preference pairs.
        L = -E[log σ( β * (log π(y_w|x) - log π_ref(y_w|x) - log π(y_l|x) + log π_ref(y_l|x) )) ]
        """
        import torch
        import torch.nn.functional as F

        if not preference_pairs:
            return torch.tensor(0.0, device=self.model.device if hasattr(self.model, 'device') else 'cpu')

        beta = self.calm_args.dpo_beta
        loss_list = []

        for i, (pair, prompt, chosen, rejected) in enumerate(zip(preference_pairs, prompts, chosen_responses, rejected_responses)):
            try:
                chosen_ids = self.tokenize(chosen, prompt)
                rejected_ids = self.tokenize(rejected, prompt)

                ref_chosen_logps = self._get_log_probabilities(self.reference_model, chosen_ids)
                ref_rejected_logps = self._get_log_probabilities(self.reference_model, rejected_ids)
                chosen_logps = self._get_log_probabilities(self.model, chosen_ids)
                rejected_logps = self._get_log_probabilities(self.model, rejected_ids)

                if ref_chosen_logps.numel() == 0 or ref_rejected_logps.numel() == 0 or chosen_logps.numel() == 0 or rejected_logps.numel() == 0:
                    continue

                ref_chosen_logps = ref_chosen_logps.mean()
                ref_rejected_logps = ref_rejected_logps.mean()
                chosen_logps = chosen_logps.mean()
                rejected_logps = rejected_logps.mean()

                chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
                rejected_rewards = beta * (rejected_logps - ref_rejected_logps)

                logits = chosen_rewards - rejected_rewards
                loss = -F.logsigmoid(logits)
                loss_list.append(loss)

            except Exception as e:
                continue

        if not loss_list:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        return torch.stack(loss_list).mean()

    def tokenize(self, text, prompt=None):
        """Tokenize text for model input."""
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return None

        # Combine prompt and response
        if prompt and isinstance(prompt, list):
            full_text = prompt[-1]['content'] + '\n\n' + text
        else:
            full_text = str(text)

        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )

        return {k: v.to(self.model.device) if hasattr(self.model, 'device') else v for k, v in inputs.items()}

    def _get_log_probabilities(self, model, input_ids):
        """Get log probabilities from model for given inputs."""
        import torch
        import torch.nn.functional as F

        try:
            # 确保 input_ids 在正确的设备上
            device = model.device if hasattr(model, 'device') else self.ref_device
            input_ids = {k: v.to(device) for k, v in input_ids.items()}
            
            # 需要梯度来计算DPO loss
            outputs = model(**input_ids)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids['input_ids'][..., 1:].contiguous()

            # Log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # Mean over non-padding tokens
            mask = (shift_labels != self.tokenizer.pad_token_id).float()
            if mask.sum() > 0:
                return (token_log_probs * mask).sum(-1) / mask.sum(-1)
            else:
                return torch.tensor(0.0)

        except Exception as e:
            return torch.tensor(0.0)

    # =========================================================================
    # DPO Training Step
    # =========================================================================
    def dpo_training_step(self, preference_pairs=None):
        """
        Execute one step of DPO training using manual implementation.
        """
        if preference_pairs is None:
            # Collect candidates from current population
            candidates = list(self.algos) if hasattr(self, 'algos') else []
            if not candidates:
                return {'dpo_loss': 0.0, 'n_pairs': 0}
            preference_pairs = self.select_preference_pairs(candidates)

        if not preference_pairs:
            return {'dpo_loss': 0.0, 'n_pairs': 0}

        # Prepare DPO dataset
        dpo_data = self.prepare_dpo_dataset(preference_pairs)

        # Ensure model is in training mode for gradients
        if hasattr(self.model, 'train'):
            self.model.train()
        
        # Use manual DPO training
        return self._manual_dpo_training(preference_pairs, dpo_data)
    
    def _manual_dpo_training(self, preference_pairs, dpo_data):
        """手动实现 DPO 训练的备用方案"""
        import torch
        import torch.nn.functional as F

        beta = self.calm_args.dpo_beta
        loss_list = []

        for pair, d in zip(preference_pairs, dpo_data):
            try:
                chosen_text = d['prompt'] + '\n\n' + d['chosen']
                rejected_text = d['prompt'] + '\n\n' + d['rejected']

                chosen_ids = self.tokenize(chosen_text)
                rejected_ids = self.tokenize(rejected_text)

                if chosen_ids is None or rejected_ids is None:
                    continue

                ref_chosen = self._get_log_probabilities(self.reference_model, chosen_ids)
                ref_rejected = self._get_log_probabilities(self.reference_model, rejected_ids)
                chosen_logps = self._get_log_probabilities(self.model, chosen_ids)
                rejected_logps = self._get_log_probabilities(self.model, rejected_ids)

                if ref_chosen.numel() == 0 or ref_rejected.numel() == 0 or chosen_logps.numel() == 0 or rejected_logps.numel() == 0:
                    continue

                ref_chosen = ref_chosen.mean()
                ref_rejected = ref_rejected.mean()
                chosen_logps = chosen_logps.mean()
                rejected_logps = rejected_logps.mean()

                logits = beta * (chosen_logps - ref_chosen - rejected_logps + ref_rejected)
                loss = -F.logsigmoid(logits)
                loss_list.append(loss)

            except Exception as e:
                print(f"DPO pair failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not loss_list:
            return {'dpo_loss': 0.0, 'n_pairs': 0}

        n_valid = len(loss_list)
        avg_loss = torch.stack(loss_list).mean()

        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()

        final_loss = avg_loss.item()

        dpo_metrics = {
            'dpo_loss': final_loss,
            'n_pairs': n_valid,
        }

        self.log_info(f"[DPO] Loss={final_loss:.4f}, Pairs={n_valid}")

        return dpo_metrics

    # =========================================================================
    # Save/Load Methods
    # =========================================================================
    def save_trace(self):
        """Save training trace to file."""
        import json
        trace_file = path.join(self.save_dir, f'trace_step_{self.log_step}.json')
        trace_data = {
            'step': self.log_step,
            'problem': self.problem_name,
            'messages': self.messages[-20:] if self.messages else [],  # Last 20 messages
            'algos': [{'name': a.name, 'perf': float(a.perf) if a.perf else None} for a in (self.algos or [])],
        }
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        print(f"Trace saved to {trace_file}")

    def save_model(self):
        """Save model checkpoints."""
        if hasattr(self, 'model'):
            model_dir = path.join(self.save_dir, f'model_step_{self.log_step}')
            os.makedirs(model_dir, exist_ok=True)
            try:
                self.model.save_pretrained(model_dir)
                if hasattr(self.model, 'tokenizer') and self.model.tokenizer:
                    self.model.tokenizer.save_pretrained(model_dir)
                print(f"Model saved to {model_dir}")
            except Exception as e:
                print(f"Warning: Failed to save model: {e}")