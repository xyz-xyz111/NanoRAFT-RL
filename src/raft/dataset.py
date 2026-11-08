"""
RAFT 数据集处理器

功能：
1. 加载Alpaca格式数据
2. 格式化为指令微调格式
3. 处理干扰项上下文和CoT
"""

import json
from typing import List, Dict
from datasets import Dataset


def load_alpaca_data(file_path: str) -> List[Dict]:
    """
    加载Alpaca格式的JSONL数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        List[Dict]: 数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_alpaca_prompt(instruction: str, input_text: str, response_template: str = "### Response:\n") -> str:
    """
    格式化为Alpaca指令格式
    
    Args:
        instruction: 指令/问题
        input_text: 输入上下文（含干扰项）
        response_template: 响应模板前缀
        
    Returns:
        str: 格式化后的prompt
    """
    if input_text and input_text.strip():
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n{response_template}"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n{response_template}"
    
    return prompt


def format_training_example(example: Dict, response_template: str = "### Response:\n") -> Dict[str, str]:
    """
    将单个样本格式化为训练格式
    
    Args:
        example: Alpaca格式样本
        response_template: 响应模板
        
    Returns:
        Dict: 包含text字段的字典
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # 构建完整prompt
    prompt = format_alpaca_prompt(instruction, input_text, response_template)
    
    # 完整的文本（prompt + output）
    full_text = prompt + output
    
    return {
        "text": full_text,
        "prompt": prompt,
        "output": output,
    }


def create_dataset(data_path: str, response_template: str = "### Response:\n") -> Dataset:
    """
    创建HuggingFace Dataset
    
    Args:
        data_path: 数据文件路径
        response_template: 响应模板
        
    Returns:
        Dataset: HuggingFace数据集
    """
    print(f"[Dataset] 正在加载数据: {data_path}")
    
    # 加载原始数据
    raw_data = load_alpaca_data(data_path)
    print(f"[Dataset] 加载了 {len(raw_data)} 条样本")
    
    # 格式化
    formatted_data = [format_training_example(ex, response_template) for ex in raw_data]
    
    # 创建Dataset
    dataset = Dataset.from_list(formatted_data)
    
    print(f"[Dataset] 数据集创建完成: {len(dataset)} 条")
    
    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 1024):
    """
    对数据集进行tokenize
    
    Args:
        dataset: HuggingFace数据集
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        Dataset: tokenized数据集
    """
    print(f"[Dataset] 正在tokenize，max_length={max_length}")
    
    def tokenize_function(examples):
        # 分别编码完整样本和Prompt，用Prompt长度构造response-only labels。
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        prompt_tokens = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        labels = []
        response_token_counts = []
        for input_ids, attention_mask, prompt_ids in zip(
            result["input_ids"],
            result["attention_mask"],
            prompt_tokens["input_ids"],
        ):
            sample_labels = input_ids.copy()
            prompt_length = min(len(prompt_ids), max_length)

            # Instruction、检索上下文和Response标记都不参与损失。
            sample_labels[:prompt_length] = [-100] * prompt_length

            # Padding位置同样不参与损失。
            for index, mask_value in enumerate(attention_mask):
                if mask_value == 0:
                    sample_labels[index] = -100

            labels.append(sample_labels)
            response_token_counts.append(
                sum(1 for token_id in sample_labels if token_id != -100)
            )

        result["labels"] = labels
        result["response_token_count"] = response_token_counts
        
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    before_filter = len(tokenized_dataset)
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: example["response_token_count"] > 0
    )
    dropped = before_filter - len(tokenized_dataset)
    tokenized_dataset = tokenized_dataset.remove_columns(["response_token_count"])

    if dropped:
        print(f"[Dataset] 丢弃 {dropped} 条Response被完全截断的样本")
    print(f"[Dataset] Tokenize完成，仅Response token参与损失")
    
    return tokenized_dataset


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python dataset.py <data.jsonl>")
        sys.exit(1)
    
    dataset = create_dataset(sys.argv[1])
    print(f"\n样本示例:")
    print(dataset[0]["text"][:500] + "...")
