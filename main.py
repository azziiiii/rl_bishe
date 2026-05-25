import os
from os import path
import json
import pynvml
import argparse
import traceback
# Initialize NVML
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

# Find the GPU with the most available memory
max_free_mem = 0
best_gpu_index = 0

for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mem = mem_info.free
    if free_mem > max_free_mem:
        max_free_mem = free_mem
        best_gpu_index = i
# Set the best GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)

# Optional: Print selected GPU
print(f"Using GPU {best_gpu_index} with {max_free_mem / 1024**2:.2f} MB free memory")
pynvml.nvmlShutdown()



import numpy as np
import random

os.environ["RAY_DISABLE_LOGGING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
import traceback
from args import Args as CALM_ARGS


def run(calm_args: CALM_ARGS):
    max_seq_length = 4096 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    online = calm_args.model_name.lower().startswith('online')
    if not online:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = calm_args.model_name,
            max_seq_length = max_seq_length,
            load_in_4bit = True, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = lora_rank,
            gpu_memory_utilization = 0.8, # Reduce if out of memory
        )

        peft_model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ], # Remove QKVO if out of memory
            lora_alpha = 2*lora_rank,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        )
        # 保存 tokenizer 到 model 中，方便 Trainer 获取
        peft_model.tokenizer = tokenizer
        model = peft_model
    else:
        model = calm_args.model_name.split('/')[1]
        print(model)
        training_args = None
    
    from calm_trainer import Trainer
    trainer = Trainer(
        model = model,
        calm_args=calm_args,
    )

    # trainer.save_model()
    n_saved = 1
    while trainer.log_step < calm_args.max_steps * (calm_args.n_generations if online else 1):
        try:
            if not online:
                # ============ DPO 模式：只用 DPO 更新，不用 GRPO ============
                # 1. 用 vLLM 生成算法（不调用 train()，跳过 GRPO 更新）
                trainer.generate_with_vllm()
            else:
                trainer.query_all()

            # ============ DPO 训练步骤 ============
            # 预热阶段（前50步）：只生成算法，不更新模型
            warmup_steps = 10
            if calm_args.use_dpo and not online and hasattr(trainer, 'algos') and len(trainer.algos) >= 2 and trainer.log_step >= warmup_steps:
                dpo_metrics = trainer.dpo_training_step()
                if dpo_metrics:
                    trainer.log_info(f"DPO Training: loss={dpo_metrics.get('dpo_loss', 0):.4f}, pairs={dpo_metrics.get('n_pairs', 0)}")

            trainer.prepare_dataset()
            # 每100步打印一次统计信息
            if trainer.log_step % 10 == 0:
                trainer.log_info(f"[Progress] Step {trainer.log_step}/{calm_args.max_steps}")
            n_saved += 1
        except:
            print(traceback.format_exc())

    # ============ 训练结束，输出最终结果 ============
    print("\n" + "="*60)
    print("Training Complete!")
    best_perf = trainer.global_best_perf
    print(f"Global Best performance: {best_perf:.4f}")

    # 保存所有算法的排名
    sorted_algos = sorted(trainer.algos, key=lambda a: a.perf, reverse=True)
    print(f"\n=== Top 10 Algorithms ===")
    for i, algo in enumerate(sorted_algos[:10]):
        print(f"  #{i+1}: perf={algo.perf:.4f} | {algo.idea[:60]}...")

    # 保存算法列表
    all_algos_file = path.join(trainer.save_dir, 'all_algos.json')
    with open(all_algos_file, 'w') as fp:
        json.dump([{
            'perf': a.perf,
            'idea': a.idea,
            'code': a.code if hasattr(a, 'code') else ''
        } for a in sorted_algos], fp, indent=2)
    print(f"\nAll algos saved to: {all_algos_file}")
    print(f"Log saved to: {trainer.log_dir}")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CALM with one or more YAML config files.')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', default='')
    args = parser.parse_args()
    config_path = args.config
    calm_args = CALM_ARGS.from_yaml(config_path)
    calm_args.log_name = path.splitext(path.basename(config_path))[0]
    n = 0
    cfp = path.abspath(path.dirname(__file__))
    while True:
        log_name = calm_args.log_name + f"_{n}"
        if not path.exists(path.join(cfp, 'calm_saved', calm_args.problem_name, log_name)):
            calm_args.log_name = log_name
            break
        n += 1
    run(calm_args)