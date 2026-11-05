#!/usr/bin/env python3
"""
RAFT PPO 强化学习训练脚本 - TRL 0.7.4 兼容版
"""

import os
import sys
import yaml
import torch
import argparse
import random
from pathlib import Path
from typing import List, Dict
import json
import gc  # 【新增】垃圾回收模块
import numpy as np

# 设置环境
# 允许外部通过环境变量覆盖（例如HF_ENDPOINT=https://huggingface.co）
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_CACHE_DIR))
print(f"[Setup] 模型缓存目录: {os.environ['HF_HOME']}")

sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
from datasets import Dataset


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int):
    """设置全局随机种子，提升可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 尽量保证可复现（会牺牲少量性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def main(config_path: str = "configs/ppo_rl_improved.yaml"):
    config = load_config(config_path)
    
    print("=" * 60)
    print("RAFT PPO 强化学习训练")
    print("=" * 60)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    print(f"[Setup] 随机种子: {seed}")
    
    ppo_config = config["ppo"]
    model_config = config["model"]
    memory_config = config["memory"]
    
    # 加载tokenizer
    print("\n[步骤1/4] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("[步骤2/4] 加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=memory_config.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=bnb_config,
        device_map={"": 0},  # 【关键】固定到0号显卡，避免碎片化
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
    )
    
    # 为4bit训练准备梯度，并加载可训练的SFT LoRA。
    # 小batch PPO关闭gradient checkpointing，避免量化模型输入无梯度。
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=False,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_config["sft_model_path"],
        is_trainable=True,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # TRL 0.7.4在PEFT模式下会关闭adapter并以Base模型计算KL，忽略显式
    # ref_model。将包装器标记为普通模型后，PPOTrainer会使用下方冻结的
    # RAFT-SFT参考模型；底层PeftModel及其LoRA训练参数不受影响。
    model.is_peft_model = False
    model.pretrained_model.config.use_cache = False

    print("[PPO] 策略模型加载完成")

    print("[PPO] 加载冻结的SFT参考模型...")
    ref_base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
    )
    ref_peft_model = PeftModel.from_pretrained(
        ref_base_model,
        model_config["sft_model_path"],
        is_trainable=False,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_peft_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    policy_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ref_trainable = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
    print(f"[诊断] 策略模型可训练参数: {policy_trainable:,}")
    print(f"[诊断] 参考模型可训练参数: {ref_trainable:,}")
    print(f"[诊断] TRL is_peft_model: {model.is_peft_model}")
    if policy_trainable == 0:
        raise RuntimeError("策略模型没有可训练参数")
    if ref_trainable != 0:
        raise RuntimeError("参考模型未完全冻结")
    
    # 加载数据
    print("\n[步骤3/4] 加载数据...")
    samples = []
    with open(config["data"]["train_file"], 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    max_samples = min(len(samples), config["data"].get("max_samples", 500))
    samples = samples[:max_samples]
    
    def build_prompt(sample):
        return f"""Below is an instruction that describes a task...

### Instruction:
{sample["instruction"]}

### Input:
{sample["input"]}

### Response:
"""
    
    dataset = Dataset.from_dict({
        "query": [build_prompt(s) for s in samples],
    })
    print(f"[PPO] 加载了 {len(dataset)} 个样本")
    
    # 配置PPO - 使用正确的参数名
    print("\n[步骤4/4] 配置PPO Trainer...")
    
    ppo_config_obj = PPOConfig(
        model_name=model_config["base_model"],
        learning_rate=float(ppo_config["learning_rate"]),
        batch_size=int(ppo_config["batch_size"]),
        mini_batch_size=int(ppo_config["mini_batch_size"]),
        gradient_accumulation_steps=int(ppo_config["gradient_accumulation_steps"]),
        ppo_epochs=int(ppo_config["ppo_epochs"]),
        # 关键修复：使用 cliprange 而不是 clip_range
        cliprange=float(ppo_config["clip_range"]),
        cliprange_value=float(ppo_config["clip_range_value"]),
        gamma=float(ppo_config["gamma"]),
        lam=float(ppo_config["lam"]),
        log_with=None,
        steps=int(ppo_config["total_steps"]),
        optimize_cuda_cache=True,
    )
    
    # 【核心修改】使用8-bit Adam优化器
    import bitsandbytes as bnb
    # 【关键】使用PagedAdamW8bit，OOM时自动将状态转移到CPU
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=float(ppo_config["learning_rate"])
    )
    
    ppo_trainer = PPOTrainer(
        config=ppo_config_obj,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,  # 传入8-bit优化器
    )
    
    print("\n" + "=" * 60)
    print("开始PPO训练")
    print("=" * 60)
    
    # 训练循环
    generation_kwargs = {
        "max_new_tokens": ppo_config["max_new_tokens"],
        "temperature": ppo_config["temperature"],
        "top_p": ppo_config["top_p"],
        "do_sample": True,
        "return_prompt": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    total_steps = int(ppo_config["total_steps"])
    dataloader_iter = iter(ppo_trainer.dataloader)
    high_kl_steps = 0
    for step in range(total_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(ppo_trainer.dataloader)
            batch = next(dataloader_iter)
        
        queries = batch["query"]
        # 【关键】强制限制输入长度，防止长尾数据撑爆显存
        query_tensors = [
            tokenizer.encode(
                q,
                return_tensors="pt",
                truncation=True,
                max_length=256  # 【极限】进一步降到256
            ).squeeze()
            for q in queries
        ]
        
        # 1. 生成回复
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        
        # 2. 【核心新增】生成后强制释放冗余显存，再进入PPO更新
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. 【RAFT专用奖励函数】
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        rewards = []
        
        for response, query in zip(responses, queries):
            reward = 0.0
            
            # (1) 格式合规性奖励 (0-0.5分)
            if "$Chain-of-Thought$" in response and "$Answer$:" in response:
                reward += 0.5
            elif "Step" in response or "Answer:" in response:
                reward += 0.3  # 部分符合
            
            # (2) CoT推理完整性 (0-0.2分)
            if "Step 1:" in response or "第一步" in response:
                reward += 0.1
            if len(response.split("\n")) >= 3:  # 多行回答，有推理结构
                reward += 0.1
            
            # (3) 长度质量：鼓励适中回答，防止冗长输出刷奖励
            response_length = len(response)
            if 200 <= response_length <= 800:
                reward += 0.2
            elif 100 <= response_length < 200:
                reward += 0.1

            # 超过800字符后连续扣分，最多扣0.4
            if response_length > 800:
                reward -= min(0.4, (response_length - 800) / 2000)

            # 极短回答通常没有完整推理
            if response_length < 80:
                reward -= 0.2
            
            rewards.append(torch.tensor(reward, dtype=torch.float32))
        
        avg_reward = sum(rewards) / len(rewards)
        
        # PPO步骤
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        current_kl = float(stats.get("objective/kl", 0))
        avg_r = sum(r.item() for r in rewards) / len(rewards)
        if step % ppo_config["logging_steps"] == 0:
            print(f"[Step {step}/{ppo_config['total_steps']}] Avg Reward: {avg_r:.3f} | "
                  f"KL: {current_kl:.4f} | "
                  f"Entropy: {stats.get('objective/entropy', 0):.4f}")

        if current_kl > ppo_config.get("max_kl", 30):
            high_kl_steps += 1
        else:
            high_kl_steps = 0

        if high_kl_steps >= ppo_config.get("max_high_kl_steps", 3):
            emergency_dir = Path(config["output"]["output_dir"]) / f"early-stop-{step}"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(emergency_dir)
            tokenizer.save_pretrained(emergency_dir)
            print(f"[Early Stop] KL连续{high_kl_steps}步过高，当前KL={current_kl:.4f}，保存到: {emergency_dir}")
            break
        
        # 【保险】每次step后清理缓存
        torch.cuda.empty_cache()
        
        if step > 0 and step % ppo_config["save_steps"] == 0:
            save_dir = Path(config["output"]["output_dir"]) / f"checkpoint-{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"[PPO] 保存检查点: {save_dir}")
    
    # 保存最终模型
    final_dir = Path(config["output"]["output_dir"]) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ PPO训练完成！模型保存到: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAFT PPO强化学习训练")
    parser.add_argument("--config", type=str, default="configs/ppo_rl_improved.yaml")
    args = parser.parse_args()
    main(args.config)
