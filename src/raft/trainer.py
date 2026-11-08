"""
RAFT SFT 训练器

功能：
1. 配置TrainingArguments
2. 创建Trainer
3. 训练循环和显存监控
"""

import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
)
from .dataset import create_dataset, tokenize_dataset


def create_training_arguments(config: dict) -> TrainingArguments:
    """
    创建训练参数
    
    Args:
        config: 配置字典
        
    Returns:
        TrainingArguments: 训练参数
    """
    train_config = config["training"]
    output_dir = train_config["output_dir"]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # 训练轮数
        num_train_epochs=train_config["num_epochs"],
        
        # Batch size
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        
        # 梯度累积
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        
        # 学习率
        learning_rate=train_config["learning_rate"],
        
        # 学习率调度
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        
        # 优化器（8-bit分页AdamW节省显存）
        optim=train_config["optim"],
        
        # 权重衰减
        weight_decay=train_config["weight_decay"],
        
        # 梯度裁剪
        max_grad_norm=train_config["max_grad_norm"],
        
        # 日志和保存
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        eval_steps=train_config["eval_steps"],
        save_total_limit=train_config["save_total_limit"],
        
        # 评估
        evaluation_strategy=train_config["evaluation_strategy"],
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        greater_is_better=train_config.get("greater_is_better", False),
        
        # 混合精度
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        
        # 报告
        report_to=train_config["report_to"],
        
        # 其他
        remove_unused_columns=False,
        # 确保在Windows上也能正常运行
        dataloader_num_workers=0,
        # 兼容旧版本Accelerate
        seed=42,
    )
    
    return args


def train(
    model,
    tokenizer,
    train_config: dict,
    train_dataset,
    eval_dataset=None,
):
    """
    执行训练
    
    Args:
        model: PEFT模型
        tokenizer: 分词器
        train_config: 训练配置
        train_dataset: 训练数据集
        eval_dataset: 验证数据集（可选）
    """
    print("=" * 60)
    print("开始RAFT SFT训练")
    print("=" * 60)
    
    # 打印训练信息
    print(f"训练样本数: {len(train_dataset)}")
    if eval_dataset:
        print(f"验证样本数: {len(eval_dataset)}")
    
    # 创建训练参数
    training_args = create_training_arguments(train_config)
    
    callbacks = []
    early_stopping_patience = train_config["training"].get(
        "early_stopping_patience", 0
    )
    if eval_dataset is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=train_config["training"].get(
                    "early_stopping_threshold", 0.0
                ),
            )
        )
        print(
            "Early Stopping已启用: "
            f"patience={early_stopping_patience}, "
            f"threshold={train_config['training'].get('early_stopping_threshold', 0.0)}"
        )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # 数据集已构造response-only labels；默认collator会原样保留labels。
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存最终模型
    output_dir = train_config["training"]["output_dir"]
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\n模型已保存到: {final_dir}")
    
    # 保存分词器
    tokenizer.save_pretrained(final_dir)
    
    # 打印显存使用情况
    if torch.cuda.is_available():
        print("\n最终显存使用情况:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  预留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return trainer


def print_training_summary(config: dict):
    """打印训练配置摘要"""
    print("\n" + "=" * 60)
    print("训练配置摘要")
    print("=" * 60)
    
    train_config = config["training"]
    
    print(f"模型: {config['model']['base_model']}")
    print(f"LoRA r: {config['lora']['r']}, alpha: {config['lora']['alpha']}")
    print(f"Batch size: {train_config['per_device_train_batch_size']} x {train_config['gradient_accumulation_steps']} (累积)")
    print(f"学习率: {train_config['learning_rate']}")
    print(f"训练轮数: {train_config['num_epochs']}")
    print(
        "Early Stopping: "
        f"patience={train_config.get('early_stopping_patience', 0)}, "
        f"threshold={train_config.get('early_stopping_threshold', 0.0)}"
    )
    print(f"最大长度: {config['data']['max_seq_length']}")
    print("=" * 60)
