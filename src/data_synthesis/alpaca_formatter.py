"""
Alpaca格式化器 - AlpacaFormatter

功能：
1. 将生成的QA数据转换为Alpaca指令格式
2. 在output中嵌入Chain-of-Thought标记
3. 添加元数据用于后续追踪

Alpaca格式标准：
{
    "instruction": "问题描述/指令",
    "input": "上下文/输入",
    "output": "推理过程 + 答案",
    "metadata": {额外信息}
}

RAFT增强格式：
- output必须包含$Chain-of-Thought$推理过程
- input包含故意注入的干扰项
"""

import json
import hashlib
import random
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .question_generator import GeneratedQA
from .distractor_injector import InjectedContext


@dataclass
class AlpacaSample:
    """
    Alpaca格式样本数据结构
    
    严格遵循Alpaca格式，同时扩展RAFT特有字段。
    """
    instruction: str           # 问题/指令
    input: str                 # 上下文（含干扰项）
    output: str                # 推理过程+答案（CoT格式）
    
    # RAFT扩展字段（存储在metadata中）
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，metadata可选是否展开"""
        base = {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }
        # 如果metadata非空，合并进去
        if self.metadata:
            base["metadata"] = self.metadata
        return base
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class AlpacaFormatter:
    """
    Alpaca格式化器
    
    负责将生成的问答对格式化为标准的Alpaca格式，
    这是后续SFT训练的标准输入格式。
    
    Attributes:
        cot_template: Chain-of-Thought输出模板
        include_metadata: 是否包含元数据字段
    """
    
    # CoT标记模板
    COT_TEMPLATES = {
        "markdown": {
            "prefix": "Let me think through this step by step:\n\n",
            "thought_prefix": "**Step {i}**: ",
            "answer_prefix": "\n**Final Answer**: ",
        },
        "latex": {
            "prefix": "$Chain-of-Thought$\n",
            "thought_prefix": "Step {i}: ",
            "answer_prefix": "\n$Answer$: ",
        },
        "plain": {
            "prefix": "Reasoning Process:\n",
            "thought_prefix": "{i}. ",
            "answer_prefix": "\nAnswer: ",
        },
    }
    
    def __init__(
        self,
        cot_style: str = "latex",
        include_metadata: bool = True,
        dataset_name: str = "raft_synthetic",
    ):
        """
        初始化格式化器
        
        Args:
            cot_style: CoT格式风格 ["markdown", "latex", "plain"]
            include_metadata: 是否包含元数据
            dataset_name: 数据集名称
        """
        if cot_style not in self.COT_TEMPLATES:
            raise ValueError(f"未知的CoT风格: {cot_style}")
        
        self.cot_template = self.COT_TEMPLATES[cot_style]
        self.include_metadata = include_metadata
        self.dataset_name = dataset_name
    
    def format_single(
        self,
        qa: GeneratedQA,
        injected_context: InjectedContext,
    ) -> AlpacaSample:
        """
        将单个QA对格式化为Alpaca格式
        
        Args:
            qa: 生成的QA对
            injected_context: 注入干扰项后的上下文
            
        Returns:
            AlpacaSample: 格式化后的样本
        """
        # 构建CoT格式的output
        output = self._format_output_with_cot(qa)
        
        # 构建metadata
        metadata = self._build_metadata(qa, injected_context)
        
        return AlpacaSample(
            instruction=qa.question,
            input=injected_context.context_text,
            output=output,
            metadata=metadata if self.include_metadata else {},
        )
    
    def format_batch(
        self,
        qa_list: List[GeneratedQA],
        context_list: List[InjectedContext],
    ) -> List[AlpacaSample]:
        """
        批量格式化
        
        Args:
            qa_list: QA对列表
            context_list: 对应的上下文列表
            
        Returns:
            List[AlpacaSample]: 格式化后的样本列表
        """
        if len(qa_list) != len(context_list):
            raise ValueError("qa_list和context_list长度必须相同")
        
        samples = []
        for qa, ctx in zip(qa_list, context_list):
            try:
                sample = self.format_single(qa, ctx)
                samples.append(sample)
            except Exception as e:
                print(f"[Warning] 格式化失败，跳过: {e}")
                continue
        
        return samples
    
    def _format_output_with_cot(self, qa: GeneratedQA) -> str:
        """
        将QA格式化为带CoT的输出
        
        Args:
            qa: QA对
            
        Returns:
            str: 格式化后的output
        """
        tmpl = self.cot_template
        
        # 解析chain_of_thought为步骤
        # 假设CoT可能有多个段落或步骤标记
        cot_text = qa.chain_of_thought.strip()
        
        # 尝试分割为步骤（如果原文有步骤标记）
        steps = self._parse_cot_steps(cot_text)
        
        # 构建格式化文本
        lines = [tmpl["prefix"]]
        for i, step in enumerate(steps, 1):
            step_line = tmpl["thought_prefix"].format(i=i) + step
            lines.append(step_line)
        
        # 添加最终答案
        lines.append(tmpl["answer_prefix"] + qa.answer.strip())
        
        return "\n".join(lines)
    
    def _parse_cot_steps(self, cot_text: str) -> List[str]:
        """
        解析CoT文本为步骤列表
        
        尝试多种分割策略：
        1. 按"Step N"或"N."标记分割
        2. 按换行分割
        3. 保持原样
        """
        import re
        
        # 尝试匹配 "Step 1:" 或 "1." 或 "(1)" 模式
        step_pattern = r'(?:Step\s*\d+[\.:\)]\s*|\d+[\.\)]\s+)'
        
        if re.search(step_pattern, cot_text, re.IGNORECASE):
            # 有步骤标记，按标记分割
            parts = re.split(step_pattern, cot_text, flags=re.IGNORECASE)
            # 过滤空字符串
            steps = [p.strip() for p in parts if p.strip()]
            return steps if steps else [cot_text]
        
        # 尝试按换行分割
        lines = [line.strip() for line in cot_text.split('\n') if line.strip()]
        if len(lines) > 1:
            return lines
        
        # 保持原样
        return [cot_text]
    
    def _build_metadata(
        self,
        qa: GeneratedQA,
        ctx: InjectedContext,
    ) -> Dict[str, Any]:
        """
        构建元数据
        
        Args:
            qa: QA对
            ctx: 注入上下文
            
        Returns:
            Dict: 元数据字典
        """
        # 同一来源文档/问题生成的多个增强版本必须进入同一数据划分。
        # 优先使用上游提供的稳定ID；旧缓存没有ID时使用来源文本哈希兼容。
        source_id = (
            qa.metadata.get("source_group_id")
            or qa.metadata.get("document_id")
            or qa.metadata.get("source_id")
        )
        if not source_id:
            normalized_source = " ".join(qa.source_chunk.split())
            source_id = hashlib.sha256(
                normalized_source.encode("utf-8")
            ).hexdigest()[:16]
        injected_text = " ".join(ctx.context_text.split())

        return {
            "sample_id": str(uuid.uuid4())[:8],
            "dataset": self.dataset_name,
            "created_at": datetime.now().isoformat(),
            "reasoning_type": qa.reasoning_type,
            "difficulty": ctx.difficulty,
            "distractor_ratio": round(ctx.mixing_ratio, 3),
            "num_golden_chunks": len(ctx.golden_chunks),
            "num_distractor_chunks": len(ctx.distractor_chunks),
            "golden_positions": ctx.golden_positions,
            "source_model": qa.metadata.get("model", "unknown"),
            "source_group_id": str(source_id),
            "golden_context_present": all(
                " ".join(chunk.split()) in injected_text
                for chunk in ctx.golden_chunks
            ),
        }
    
    def save_to_jsonl(
        self,
        samples: List[AlpacaSample],
        output_path: str,
    ):
        """
        保存为JSONL格式（推荐，便于流式处理）
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"[AlpacaFormatter] 已保存 {len(samples)} 条样本到 {output_path}")
    
    def save_to_json(
        self,
        samples: List[AlpacaSample],
        output_path: str,
    ):
        """
        保存为JSON数组格式
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [s.to_dict() for s in samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[AlpacaFormatter] 已保存 {len(samples)} 条样本到 {output_path}")
    
    @staticmethod
    def load_from_jsonl(input_path: str) -> List[AlpacaSample]:
        """
        从JSONL加载样本
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            List[AlpacaSample]: 样本列表
        """
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata = data.pop("metadata", {})
                samples.append(AlpacaSample(metadata=metadata, **data))
        
        return samples
    
    def split_and_save(
        self,
        samples: List[AlpacaSample],
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        按来源组划分训练/验证/测试集并保存，避免同源增强样本泄漏。
        
        Args:
            samples: 全部样本
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例（剩余为测试集）
            seed: 随机种子
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio必须在0和1之间")
        if not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
            raise ValueError("val_ratio无效，且必须为测试集保留比例")

        groups: Dict[str, List[AlpacaSample]] = {}
        for sample in samples:
            group_id = sample.metadata.get("source_group_id")
            if not group_id:
                # 兼容手工构造的数据；相同问题仍会被放在同一组。
                normalized_question = " ".join(sample.instruction.split())
                group_id = hashlib.sha256(
                    normalized_question.encode("utf-8")
                ).hexdigest()[:16]
                sample.metadata["source_group_id"] = group_id
            groups.setdefault(str(group_id), []).append(sample)

        group_ids = sorted(groups)
        random.Random(seed).shuffle(group_ids)

        n_groups = len(group_ids)
        n_train_groups = int(n_groups * train_ratio)
        n_val_groups = int(n_groups * val_ratio)
        train_group_ids = group_ids[:n_train_groups]
        val_group_ids = group_ids[n_train_groups:n_train_groups + n_val_groups]
        test_group_ids = group_ids[n_train_groups + n_val_groups:]
        train_ids = set(train_group_ids)
        val_ids = set(val_group_ids)
        test_ids = set(test_group_ids)

        train_samples = [sample for gid in train_group_ids for sample in groups[gid]]
        val_samples = [sample for gid in val_group_ids for sample in groups[gid]]
        test_samples = [sample for gid in test_group_ids for sample in groups[gid]]

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise RuntimeError("检测到source_group_id跨数据集泄漏")

        n_total = len(samples)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各集合
        self.save_to_jsonl(train_samples, output_dir / "train.jsonl")
        self.save_to_jsonl(val_samples, output_dir / "val.jsonl")
        self.save_to_jsonl(test_samples, output_dir / "test.jsonl")
        
        # 保存统计信息
        stats = {
            "total": n_total,
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "total_groups": n_groups,
            "train_groups": len(train_ids),
            "val_groups": len(val_ids),
            "test_groups": len(test_ids),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1 - train_ratio - val_ratio, 10),
            "seed": seed,
            "group_key": "metadata.source_group_id",
            "group_leakage": False,
        }
        
        with open(output_dir / "split_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[AlpacaFormatter] 数据集划分完成: {stats}")


class AlpacaPromptBuilder:
    """
    Alpaca提示词构建器（用于推理时）
    
    构建与训练时格式一致的prompt，确保模型行为一致。
    """
    
    # Alpaca标准提示词模板
    PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    
    @classmethod
    def build_prompt(
        cls,
        instruction: str,
        input_text: str = "",
    ) -> str:
        """
        构建标准Alpaca格式提示词
        
        Args:
            instruction: 指令/问题
            input_text: 输入/上下文
            
        Returns:
            str: 完整提示词
        """
        return cls.PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
        )
    
    @classmethod
    def build_chat_prompt(
        cls,
        instruction: str,
        input_text: str = "",
        system_msg: str = "You are a helpful assistant.",
    ) -> List[Dict[str, str]]:
        """
        构建Chat格式提示词（用于Chat模型）
        
        Args:
            instruction: 指令
            input_text: 输入
            system_msg: 系统消息
            
        Returns:
            List[Dict]: 消息列表
        """
        user_content = instruction
        if input_text:
            user_content += f"\n\nContext:\n{input_text}"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]


if __name__ == "__main__":
    # 使用示例
    formatter = AlpacaFormatter(cot_style="latex")
    
    # 模拟数据
    from dataclasses import dataclass
    
    @dataclass
    class MockQA:
        question: str
        chain_of_thought: str
        answer: str
        reasoning_type: str
        metadata: dict
    
    @dataclass  
    class MockContext:
        context_text: str
        golden_chunks: list
        distractor_chunks: list
        golden_positions: list
        difficulty: str
        mixing_ratio: float
    
    qa = MockQA(
        question="如何在Selenium中处理Shadow DOM元素？",
        chain_of_thought="Step 1: 识别Shadow DOM边界。Step 2: 使用JavaScript执行器。Step 3: 定位内部元素。",
        answer="使用execute_script方法穿透Shadow DOM边界。",
        reasoning_type="multi_step",
        metadata={"model": "gpt-3.5-turbo"},
    )
    
    ctx = MockContext(
        context_text="Document 1: ...\n\nDocument 2: ...",
        golden_chunks=["..."],
        distractor_chunks=["...", "..."],
        golden_positions=[0],
        difficulty="medium",
        mixing_ratio=0.67,
    )
    
    sample = formatter.format_single(qa, ctx)
    print(sample.to_json())
