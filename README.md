# NanoRAFT-RL

围绕 RAFT 训练范式设计并实现检索增强微调与 PPO 对齐工程，覆盖数据构造、Qwen2-7B 参数高效训练和统一评估。

## 实验背景

本项目面向“检索增强监督微调 + 强化学习对齐”的闭环训练问题，在单卡 24GB 显存条件下实现以下链路：

- 检索增强样本（含干扰上下文）驱动的 SFT 训练
- SFT 模型上的 PPO 对齐训练
- 基线/SFT/PPO 三模型统一评估

除模型效果外，重点解决训练稳定性和工程可重复执行问题：

- 首次模型下载后可复用本地缓存
- 在网络波动和镜像切换场景下仍能稳定执行
- 对 LoRA adapter 产物进行一致化评估

## 核心工程贡献

原项目提供 RAFT 数据构造和 SFT 主链路；在此基础上扩展出可在单张 RTX 4090 上运行、可稳定训练并可统一评估的完整工程：

- **解决单卡资源限制：**采用 4-bit QLoRA、LoRA adapter、梯度累积和本地模型缓存，完成 `RAFT 数据 → SFT → PPO → 评估` 全流程；
- **解决 PPO 参考策略失效：**定位 TRL 0.7.4 与 PEFT 组合下参考策略未按预期参与 KL 计算的问题，显式加载并冻结 RAFT-SFT 参考模型，同时修正 Trainer 的参考模型路径和生成输入处理；
- **提高训练稳定性：**重新配置学习率、PPO epoch、clip range、batch 和生成参数，并加入长度奖励、超长惩罚、连续高 KL 提前停止及训练参数自检；
- **补齐评估能力：**将评估脚本改造成可通过命令行指定模型路径、结果标签和输出文件的统一入口，使 Base、RAFT-SFT 与 PPO 使用相同数据和指标进行比较；
- **最终结果：**PPO 在统一评估中的综合得分达到 `0.826`，相对 Base 的 `0.360` 提升约 `129.4%`；完成 60 步稳定训练，观测 KL 保持在约 `-0.6～0.6`，平均输出长度为 `614`。

## 数据集说明（医学领域）

本项目使用的数据集位于 `data/`，核心是以 PubMed RCT 医学摘要为基础构建的检索增强训练语料。
项目数据文件（如 `data.zip`）可通过百度网盘获取：https://pan.baidu.com/s/1mK_s9YtKE3CwbHdDh_F5TA?pwd=xcwt。

### 1) 数据来源与任务属性

- 原始来源：`data/raw/pubmed-rct-master`（PubMed RCT 语料）
- 医学属性：样本来自随机对照试验（RCT）摘要，覆盖临床研究中高证据等级文本
- 结构标签：每篇摘要按 `BACKGROUND / OBJECTIVE / METHODS / RESULTS / CONCLUSIONS` 组织，天然适配“研究设计-结果-结论”链路理解

### 2) 当前仓库中的数据落地形态

- `data/processed/pubmed`：已转为 Markdown 的结构化医学文档共 **25,000** 篇
- 其中 `pubmed_20k_rct`：`train=15,000`、`dev=2,500`、`test=2,500`
- 其中 `pubmed_200k_rct`：`dev=2,500`、`test=2,500`（train 保留压缩包用于按需扩展）
- `data/synthetic`：RAFT 训练样本 **100** 条，按 `70/20/10` 划分为 `train/val/test`
- `data/synthetic_smoke`：烟雾测试集 3 条，用于快速链路验证

### 3) 对医学领域的直接贡献

- 面向循证医学的信息处理：以 RCT 摘要作为训练基础，有利于模型学习“证据级别更高”的医学文本表达
- 面向临床文献筛选场景：训练样本中引入干扰上下文（distractors），更贴近真实医生/研究者在海量文献中的检索与甄别过程
- 面向医学推理可解释性：结构化章节 + CoT 输出约束，使模型回答更容易对应到“研究背景-方法-结果-结论”的证据链
- 面向中文医疗 AI 应用：在英文医学文献基础上生成中文指令与推理回答，增强跨语种医学知识迁移能力

## 实验设置

在 RTX 4090 24GB、Python 3.10、torch 2.1.2+cu121 环境下完成整套训练与评估链路：

- SFT：`outputs/raft-sft/final`
- PPO：`outputs/raft-ppo-improved/final`
- 评估：`outputs/evaluation_results_improved.json`

评估配置为 `data/synthetic/test.jsonl`，实际样本数为 10（`--max_samples 30` 但测试集仅 10 条）。

## 实验结果评估

模型评估结果如下：

| 模型 | 格式得分 | CoT得分 | 答案得分 | 综合得分 | 平均长度 |
|---|---:|---:|---:|---:|---:|
| 基线模型(Base) | 0.000 | 0.560 | 0.640 | 0.360 | 749 |
| RAFT-SFT模型 | 0.850 | 0.960 | 0.700 | **0.838** | 385 |
| PPO模型 | 0.850 | 0.930 | 0.690 | **0.826** | 614 |

结果解读：

- RAFT-SFT 将综合得分从 `0.360` 提升到 `0.838`（相对提升约 `132.8%`），主要收益来自格式遵循、证据组织和 CoT 结构学习。
- PPO 训练 KL 稳定在约 `-0.6～0.6`，综合得分达到 `0.826`，接近 RAFT-SFT 的 `0.838`。
- PPO 相对基线提升约 `129.4%`；答案分数仍略低于 SFT，说明奖励函数还有进一步加入答案正确性和证据一致性信号的空间。
- 测试集只有 10 条样本，本结果用于验证训练与评估闭环，并不代表具有统计显著性的模型能力结论。

### PPO 稳定训练设计

针对 TRL 0.7.4 与 PEFT 的参考策略处理方式，项目采用以下稳定训练设计：

1. 显式加载并冻结 RAFT-SFT 参考模型，使 KL 在相同初始策略之间计算；
2. 将学习率从 `1e-5` 降为 `5e-7`，PPO epoch 从 2 降为 1，clip range 从 `0.2` 降为 `0.1`；
3. 生成时设置 `return_prompt=False`，避免将输入提示误作模型回复；
4. 对超过 800 字符的回答加入连续长度惩罚；
5. 监控连续高 KL 并自动提前停止，保留中间检查点。

## 创新性（AI 方法、医学应用与工程实现）

1. 检索增强 + 高干扰上下文训练（RAFT核心）
- 训练样本为“问题 + 多文档上下文 + 干扰项”联合输入，显式训练模型在噪声医学文本中定位有效证据的能力。

2. 医学结构先验融入训练数据
- PubMed RCT 五段式结构（BACKGROUND/OBJECTIVE/METHODS/RESULTS/CONCLUSIONS）被保留到训练链路，帮助模型学习医学研究的标准叙述范式，而不是仅记忆表面文本模式。

3. 分阶段训练与消融对比：SFT -> PPO
- 先用监督学习学习结构化回答，再用 PPO 探索奖励对齐；三模型统一评估用于分析两个训练阶段各自带来的能力变化。

4. 面向医学问答的轻量奖励工程
- PPO 阶段奖励函数同时考虑格式合规、推理完整性、答案具体性与幻觉惩罚，避免只追求表面流畅而忽略医学场景中关键的证据一致性。

5. 适配器感知评估与三模型统一对比
- 评估脚本兼容 LoRA adapter 目录自动加载，并统一比较 Base/SFT/PPO，能直接观察“检索增强训练”和“强化学习对齐”各自带来的增益。

6. 单卡 24GB 的可执行大模型对齐闭环
- 使用 `Qwen2-7B + 4bit + LoRA + 8bit optimizer` 完成端到端训练，并按阶段设计显存优化策略，降低大模型训练的硬件门槛。

7. 稳定执行优先的工程化设计
- 统一缓存目录到 `models/cache`、严格执行 `total_steps`、固定全局随机种子、支持离线运行验证，提升跨次实验的一致性和可追溯性。

---

## 关键工程问题说明

### 1) 模型加载问题怎么解决？首次与后续有什么区别？

核心策略：统一 Hugging Face 缓存目录到项目内 `models/cache`。

目前脚本行为：

- `scripts/run_raft_sft.py`：默认使用 `models/cache`
- `scripts/run_ppo_training.py`：默认使用 `models/cache`
- `scripts/run_evaluation.py`：默认使用 `models/cache`
- `scripts/run_data_synthesis.py`：默认使用 `models/cache`

这意味着：

- 首次运行：会下载缺失模型文件到 `models/cache`
- 后续运行：优先复用本地缓存，不再重复下载

如需后续严格离线（完全不访问外网），可额外加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 2) `data.zip` 解压位置如何设置？

是，但要注意目录层级。

`/root/autodl-tmp/data.zip` 的顶层是 `processed/`、`synthetic/` 等，不带 `data/` 前缀。
因此正确命令是：

```bash
cd /root/autodl-tmp/NanoRAFT-RL
unzip -o /root/autodl-tmp/data.zip -d data
```

解压后应出现：

- `data/processed/index/...`
- `data/synthetic/train.jsonl`
- `data/synthetic/val.jsonl`
- `data/synthetic/test.jsonl`

若直接 `-d .`，会变成 `processed/` 落在仓库根目录，路径不匹配脚本默认配置。

### 3) GitHub 拉取代码后能否完整运行（拿到数据后）？

谨慎结论：

- 可以完整运行：`SFT -> PPO -> Eval`（前提：拿到 `data.zip`，并有 24GB 显存）
- 不能 100% 保证跨机器“完全一致数值”：生成式训练存在随机性
- 不能 100% 保证“全流程含数据重建”稳定：数据重建依赖外部 API 与网络（Zhipu/HF）

换句话说：

- 对“工程可跑通并产出模型与评估文件”这一目标：可以保证（在同类环境）
- 对“任何机器、任何时刻、所有分数逐位一致”：不能保证

---

## 推荐运行路径（拿到 data.zip 后）

### 0. 环境

```bash
cd /root/autodl-tmp/NanoRAFT-RL
source /etc/network_turbo
pip install -r requirements.txt
```

### 1. 数据就位

数据分发说明：`data.zip` 由作者通过百度网盘提供，下载地址为 https://pan.baidu.com/s/1mK_s9YtKE3CwbHdDh_F5TA?pwd=xcwt。

```bash
unzip -o /root/autodl-tmp/data.zip -d data
```

### 2. SFT

```bash
python -u scripts/run_raft_sft.py --config configs/raft_sft.yaml
```

### 3. PPO（稳定版配置）

```bash
python -u scripts/run_ppo_training.py --config configs/ppo_rl_improved.yaml
```

说明：稳定版针对 TRL 0.7.4 显式加载冻结的 RAFT-SFT 参考模型，并加入 KL 监控和长度惩罚。当前实现会严格按照 `total_steps` 执行（dataloader 耗尽后自动续跑）。

### 4. 评估

```bash
python -u scripts/run_evaluation.py \
  --test_file data/synthetic/test.jsonl \
  --max_samples 30 \
  --base_model Qwen/Qwen2-7B-Instruct \
  --ppo_model outputs/raft-ppo-improved/final \
  --ppo_label 改进PPO模型 \
  --output_file outputs/evaluation_results_improved.json \
  --seed 42
```

---

## 严格“从原始文档全量重建”路径（更慢、更不稳定）

这条路径会重新做数据合成，依赖外部 API 与网络，不建议在验收阶段首选。

```bash
cd /root/autodl-tmp/NanoRAFT-RL
source /etc/network_turbo
pip install -r requirements.txt

export ZHIPU_API_KEY="<YOUR_KEY>"
python -u scripts/run_data_synthesis.py --config configs/data_synthesis.yaml --force-regenerate

python -u scripts/run_raft_sft.py --config configs/raft_sft.yaml
python -u scripts/run_ppo_training.py --config configs/ppo_rl_improved.yaml
python -u scripts/run_evaluation.py --test_file data/synthetic/test.jsonl --max_samples 30 --base_model Qwen/Qwen2-7B-Instruct --seed 42
```

---

## 缓存机制说明（首次下载后复用）

默认缓存目录：

- `models/cache`

建议保留该目录，避免每次重拉 7B 分片。常用检查：

```bash
du -sh models/cache
find models/cache -maxdepth 3 -type f | head
```

如需验证“只用本地缓存”：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 关键输出

- SFT模型：`outputs/raft-sft/final`
- PPO模型：`outputs/raft-ppo-improved/final`
- 评估结果：`outputs/evaluation_results_improved.json`
- 运行日志：`logs/`

---

## 常见问题（简版）

1. 模型又开始下载，感觉没走缓存
- 检查是否在项目目录运行；确认 `models/cache` 存在并有权重分片。

2. `data.zip` 解压后脚本找不到数据
- 大概率解压层级错了。必须是 `.../NanoRAFT-RL/data/processed`，不是 `.../NanoRAFT-RL/processed`。

3. 如何保证多次跑结果更稳定
- SFT 默认 `seed=42`；PPO 与评估可通过配置/参数固定种子（`seed=42`）。
- 仍需注意：生成模型在不同驱动/算子实现下可能有微小差异，属于正常范围。

4. 网络抖动导致 HF 或 API 失败
- 先执行 `source /etc/network_turbo`；必要时切换 `HF_ENDPOINT` 到官方或镜像。

---

## 变更与踩坑记录

完整工程问题与修复记录见：

- `engineering_notes.md`
