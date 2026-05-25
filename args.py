from dataclasses import dataclass, fields
from typing import List, Dict, Any
import argparse
import yaml

@dataclass
class Args:
    problem_name: str = 'obp'
    log_name: str = 'default'
    max_steps: int = 500

    api_key: str = ''
    base_url: str = ''

    n_prompts: int = 1
    population_size: int = 10
    ub_simplification: int = 1
    ub_injection: int = 1
    ub_replacement: int = 2
    ub_crossover: int = 4
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct'
    n_generations: int = 1
    seed_algo_fp: str = ''
    template_fp: str = ''

    speed_collapse: float = 0.0005
    max_stuck_threshold: int = -1

    reward_idea_not_exist: float = -1.0
    reward_code_not_exist: float = -0.95
    reward_function_not_exist: float = -0.90
    reward_bug_in_function: float = -0.85
    reward_random_algorithm: float = -0.75

    lr: float = 1e-5

    # DPO related parameters
    use_dpo: bool = True        #   Use DPO instead of GRPO
    dpo_beta: float = 0.2        # DPO temperature parameter
    n_preference_pairs: int = 8 # Max number of preference pairs per step
    diversity_threshold: float = 0.05  # Minimum diversity for preference pairs
    pref_gap_min: float = 0.01   # Minimum performance gap
    pref_gap_max: float = 0.9    # Maximum performance gap

    # DAR sampling parameters
    dar_n_buckets: int = 5       # Number of performance buckets for DAR
    dar_temperature: float = 3.0 # Temperature for bucket sampling (explore vs exploit)

    @staticmethod
    def from_terminal():
        parser = argparse.ArgumentParser()
        for field in fields(Args):
            field_type = field.type
            default_value = field.default

            # Handle list types (parse from comma-separated string)
            if field_type == List[int]:
                parser.add_argument(f'--{field.name}', type=lambda s: [int(x) for x in s.split(',')],
                                    default=list(default_value))
                continue

            # Handle dictionary (parse from comma-separated key=value pairs)
            if field_type == Dict[str, Any]:
                parser.add_argument(f'--{field.name}', type=lambda s: dict(item.split('=') for item in s.split(',')),
                                    default=default_value or {})
                continue

            # Handle boolean flags properly
            if field_type is bool:
                if default_value is False:
                    parser.add_argument(f'--{field.name}', action='store_true')
                else:
                    parser.add_argument(f'--no-{field.name}', dest=field.name, action='store_false')
                continue

            parser.add_argument(f'--{field.name}', type=field_type, default=default_value)

        args = parser.parse_args()
        args_dict = vars(args)

        return Args(**args_dict)

    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        args_dict = {}
        for field in fields(Args):
            field_name = field.name
            field_value = data.get(field_name, field.default)

            # Convert string attributes to lowercase
            # if isinstance(field_value, str):
                # field_value = field_value.lower()

            # Ensure list attributes are properly converted
            if isinstance(field_value, list) and field.type == List[int]:
                field_value = [int(x) for x in field_value]

            # Ensure dictionary attributes are properly converted
            if isinstance(field_value, dict) and field.type == Dict[str, Any]:
                field_value = {str(k): v for k, v in field_value.items()}

            args_dict[field_name] = field_value

        return Args(**args_dict)