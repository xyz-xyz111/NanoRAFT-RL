#!/usr/bin/env python3
"""
RAFT 模型评估脚本

对比三个模型：
1. 基线模型 (Base Qwen2-7B)
2. SFT模型 (RAFT微调后)
3. PPO模型 (强化学习后)

评估指标：
- 格式合规性 (CoT格式)
- 引用准确性 (在干扰项中找答案)
- 答案完整性
"""

import os
import sys
import json
import torch
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

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
from peft import PeftModel
from tqdm import tqdm


def set_global_seed(seed: int):
    """设置全局随机种子，提升多次评估的一致性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RaftEvaluator:
    """RAFT模型评估器"""
    
    def __init__(self, model_path: str, model_name: str, device="cuda"):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径或HuggingFace模型名
            model_name: 模型标识名（用于输出）
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        
        print(f"\n{'='*60}")
        print(f"加载模型: {model_name}")
        print(f"{'='*60}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型（4-bit以节省显存）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # 兼容LoRA适配器目录（仅含adapter权重）
        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                peft_cfg = json.load(f)
            base_model_name = peft_cfg["base_model_name_or_path"]

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()
        self.model_device = next(self.model.parameters()).device
        
        print(f"✅ {model_name} 加载完成")
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只返回Response部分
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        return response
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """评估单个样本"""
        instruction = sample["instruction"]
        context = sample["input"]
        reference = sample["output"]
        
        # 构建Prompt
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
"""
        
        # 生成回答
        generated = self.generate(prompt)
        
        # 计算各项指标
        metrics = {
            "instruction": instruction[:100] + "...",
            "reference": reference[:200] + "...",
            "generated": generated[:300] + "...",
            "format_score": self._score_format(generated),
            "cot_score": self._score_cot(generated),
            "answer_score": self._score_answer(generated),
            "length": len(generated),
        }
        
        return metrics
    
    def _score_format(self, text: str) -> float:
        """格式合规性评分 (0-1)"""
        score = 0.0
        if "$Chain-of-Thought$" in text:
            score += 0.5
        if "$Answer$:" in text:
            score += 0.5
        return score
    
    def _score_cot(self, text: str) -> float:
        """CoT推理质量评分 (0-1)"""
        score = 0.0
        # 检查是否有步骤标记
        if "Step 1:" in text or "Step 2:" in text or "第一步" in text:
            score += 0.3
        # 检查是否有推理结构（多行）
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 3:
            score += 0.3
        # 检查长度合理性
        if 200 < len(text) < 1500:
            score += 0.4
        return score
    
    def _score_answer(self, text: str) -> float:
        """答案具体性评分 (0-1)"""
        score = 0.0
        # 检查是否包含具体信息（数字、专业术语等）
        if any(c.isdigit() for c in text):
            score += 0.3
        # 检查是否有结论性语句
        if any(kw in text.lower() for kw in ["conclusion", "因此", "所以", "综上"]):
            score += 0.3
        # 长度合理性
        if 100 < len(text) < 2000:
            score += 0.4
        return score
    
    def evaluate_dataset(self, dataset_path: str, max_samples: int = 50) -> Dict:
        """评估整个数据集"""
        # 加载数据
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        samples = samples[:max_samples]
        print(f"\n评估 {len(samples)} 个样本...")
        
        results = []
        for sample in tqdm(samples, desc=f"评估 {self.model_name}"):
            result = self.evaluate_sample(sample)
            results.append(result)
        
        # 计算平均分
        avg_metrics = {
            "model": self.model_name,
            "samples": len(results),
            "avg_format_score": sum(r["format_score"] for r in results) / len(results),
            "avg_cot_score": sum(r["cot_score"] for r in results) / len(results),
            "avg_answer_score": sum(r["answer_score"] for r in results) / len(results),
            "avg_length": sum(r["length"] for r in results) / len(results),
        }
        
        # 综合得分
        avg_metrics["overall_score"] = (
            avg_metrics["avg_format_score"] * 0.4 +
            avg_metrics["avg_cot_score"] * 0.3 +
            avg_metrics["avg_answer_score"] * 0.3
        )
        
        return avg_metrics, results


def print_comparison_table(results: List[Dict]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("模型评估对比结果")
    print("="*80)
    
    # 表头
    print(f"{'模型':<20} {'格式得分':<12} {'CoT得分':<12} {'答案得分':<12} {'综合得分':<12} {'平均长度':<12}")
    print("-"*80)
    
    # 数据行
    for r in results:
        print(f"{r['model']:<20} "
              f"{r['avg_format_score']:<12.3f} "
              f"{r['avg_cot_score']:<12.3f} "
              f"{r['avg_answer_score']:<12.3f} "
              f"{r['overall_score']:<12.3f} "
              f"{r['avg_length']:<12.0f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RAFT模型评估")
    parser.add_argument("--test_file", type=str, default="data/synthetic/test.jsonl",
                       help="测试集路径")
    parser.add_argument("--max_samples", type=int, default=30,
                       help="评估样本数（建议30-50）")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct",
                       help="基座模型名")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（默认42）")
    parser.add_argument("--sft_model", type=str, default="outputs/raft-sft/final",
                       help="SFT adapter路径")
    parser.add_argument("--ppo_model", type=str, default="outputs/raft-ppo-improved/final",
                       help="PPO adapter路径")
    parser.add_argument("--ppo_label", type=str, default="PPO模型",
                       help="PPO模型在结果表中的名称")
    parser.add_argument("--output_file", type=str, default="outputs/evaluation_results_improved.json",
                       help="评估结果JSON路径")
    args = parser.parse_args()
    set_global_seed(args.seed)
    print(f"[Setup] 随机种子: {args.seed}")
    
    all_results = []
    
    # 1. 评估基线模型
    print("\n" + "="*80)
    print("开始评估: 基线模型 (Base)")
    print("="*80)
    base_eval = RaftEvaluator(args.base_model, "基线模型(Base)")
    base_metrics, _ = base_eval.evaluate_dataset(args.test_file, args.max_samples)
    all_results.append(base_metrics)
    
    # 清理显存
    del base_eval.model
    del base_eval
    torch.cuda.empty_cache()
    
    # 2. 评估SFT模型
    print("\n" + "="*80)
    print("开始评估: SFT模型")
    print("="*80)
    sft_eval = RaftEvaluator(args.sft_model, "SFT模型")
    sft_metrics, _ = sft_eval.evaluate_dataset(args.test_file, args.max_samples)
    all_results.append(sft_metrics)
    
    # 清理显存
    del sft_eval.model
    del sft_eval
    torch.cuda.empty_cache()
    
    # 3. 评估PPO模型
    print("\n" + "="*80)
    print(f"开始评估: {args.ppo_label}")
    print("="*80)
    ppo_eval = RaftEvaluator(args.ppo_model, args.ppo_label)
    ppo_metrics, _ = ppo_eval.evaluate_dataset(args.test_file, args.max_samples)
    all_results.append(ppo_metrics)
    
    # 打印对比表格
    print_comparison_table(all_results)
    
    # 计算提升幅度
    print("\n" + "="*80)
    print("相比基线模型的提升")
    print("="*80)
    
    base_overall = all_results[0]["overall_score"]
    for r in all_results[1:]:
        improvement = (r["overall_score"] - base_overall) / base_overall * 100
        print(f"{r['model']}: 综合得分 {r['overall_score']:.3f} (提升 {improvement:+.1f}%)")
    
    print("="*80)
    
    # 保存详细结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
