# NanoRAFT-RL 工程问题与修复记录

日期：2026-02-28  
环境：RTX 4090 24GB，Python 3.10.8，torch 2.1.2+cu121

## 1) 克隆后核心依赖缺失
- 症状：
  - 运行时大量导入失败（`transformers`、`peft`、`datasets`、`trl`、`langchain`、`llama_index` 等）。
- 根因：
  - 全新环境仅预装了部分包，项目依赖未完整安装。
- 修复：
  - 执行 `pip install -r requirements.txt` 安装项目依赖。
- 结果：
  - 核心脚本恢复可导入、可执行。

## 2) `run_data_synthesis.py` 导入报错：`No module named langchain_openai`
- 症状：
  - 报错 `ModuleNotFoundError: No module named 'langchain_openai'`。
- 根因：
  - `requirements.txt` 缺少 `langchain-openai`，但代码中有 `from langchain_openai import ChatOpenAI`。
- 修复：
  - 在 `requirements.txt` 增加 `langchain-openai==0.0.2`。
  - 重新安装依赖。
- 结果：
  - 数据合成脚本通过 CLI / import 阶段。

## 3) `sentence-transformers==2.2.2` 与新版 `huggingface_hub` 不兼容
- 症状：
  - 报错 `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`。
- 根因：
  - `sentence-transformers==2.2.2` 依赖旧版 `huggingface_hub` API。
- 修复：
  - 在 `requirements.txt` 固定 `huggingface_hub==0.19.4`。
  - 重新安装兼容版本。
- 结果：
  - `sentence_transformers` 导入恢复正常。

## 4) Hugging Face 模型下载网络不可达
- 症状：
  - 下载时出现 `Network is unreachable`。
- 根因：
  - 默认网络路径访问 `huggingface.co` 不稳定。
- 修复：
  - 启用加速命令：`source /etc/network_turbo`。
  - 设置 `HF_ENDPOINT=https://hf-mirror.com`。
- 结果：
  - `huggingface.co` 与 `hf-mirror.com` 连通性恢复。

## 5) API Key 安全加固
- 症状：
  - `configs/data_synthesis.yaml` 中存在明文智谱 API Key。
- 根因：
  - 敏感信息直接写入配置文件。
- 修复：
  - 清空配置中的 key，改为环境变量输入（`ZHIPU_API_KEY`）。
- 结果：
  - 在不把密钥写入仓库的前提下继续运行数据流程。

## 6) 索引持久化中断导致向量文件损坏
- 症状：
  - 加载 `data/processed/index/default__vector_store.json` 时出现 `JSONDecodeError`。
- 根因：
  - 合成过程在保存索引时被中断，产生截断 JSON。
- 修复：
  - 从 `data.zip` 恢复索引文件：
    - `data/processed/index/default__vector_store.json`
    - `data/processed/index/docstore.json`
    - `data/processed/index/graph_store.json`
    - `data/processed/index/index_store.json`
- 结果：
  - 索引损坏问题消除，可继续使用现有数据资产。

## 7) embedding / 索引校验阶段耗时过长，误判为卡住
- 症状：
  - 校验阶段耗时很久，表面上像“卡死”。
- 根因：
  - 首次运行包含大型 embedding 模型下载与索引加载。
- 修复：
  - 训练路径先使用已提供的 `data/synthetic` 完成训练与评估闭环，再选择是否全量重跑数据合成。
- 结果：
  - 避免重复长时间阻塞，先保障端到端流程可执行。

## 8) SFT 加载 Qwen2 失败（transformers 版本过低）
- 症状：
  - 加载 `Qwen/Qwen2-7B-Instruct` 时出现 `KeyError: 'qwen2'`。
- 根因：
  - `transformers==4.36.2` 不包含 Qwen2 配置映射。
- 修复：
  - 升级到 `transformers==4.37.2`。
  - 同步更新 `requirements.txt`。
- 结果：
  - Qwen2 架构加载兼容。

## 10) 数据合成 smoke 在镜像源发生 SSL 握手超时
- 症状：
  - 从 `hf-mirror.com` 拉取 `all-MiniLM-L6-v2` 时出现 `ProxyError` / SSL 握手超时。
- 根因：
  - 当前代理链路下镜像端点阶段性不稳定。
- 修复：
  - 保留 `source /etc/network_turbo`，并在该步骤改用 `huggingface.co` 直连（取消 `HF_ENDPOINT`）。
- 结果：
  - 数据合成 smoke 成功（3/3 样本生成 + 干扰项注入 + 切分保存）。

## 12) 评估脚本需兼容 LoRA adapter 目录
- 症状：
  - SFT/PPO 输出为 adapter 目录，不是完整 base model checkpoint。
- 根因：
  - 原评估脚本默认对所有路径都执行 `AutoModelForCausalLM.from_pretrained(model_path)`。
- 修复：
  - 更新 `scripts/run_evaluation.py`：
    - 检测 `adapter_config.json`
    - 4-bit 加载 base model
    - 通过 `PeftModel.from_pretrained` 挂载 adapter
    - 统一生成设备处理
- 结果：
  - 基线 / SFT / PPO 三模型评估完整通过。

## 13) HF 镜像写死降低可移植性
- 症状：
  - SFT / PPO 脚本总是强制 `HF_ENDPOINT=https://hf-mirror.com`。
- 根因：
  - 脚本内硬编码环境变量，外部无法覆盖。
- 修复：
  - 改为仅在未设置 `HF_ENDPOINT` 时才使用镜像默认值。
- 结果：
  - 可在命令层灵活选择镜像或官方端点。

## 14) PPO 重跑时直连 `huggingface.co` 代理握手失败
- 症状：
  - 访问 `huggingface.co` tokenizer / model 文件时报 `requests.exceptions.ProxyError`，SSL 握手超时。
- 根因：
  - 当前代理路由在直连端点上存在间歇性 TLS 握手失败。
- 修复：
  - 重跑阶段不再频繁切换端点，改走可确定的本地缓存加载路径。
- 结果：
  - 训练与评估主路径摆脱网络抖动影响。

## 15) 离线模式下缓存路径不一致导致缺失分片
- 症状：
  - 离线运行 PPO 时出现 `LocalEntryNotFoundError`，缺失 `model-00004-of-00004.safetensors`。
- 根因：
  - `~/.cache/huggingface` 只有部分分片，完整 15G 快照在项目缓存 `models/cache`。
- 修复：
  - 显式指定缓存环境变量：
    - `TRANSFORMERS_CACHE=/root/autodl-tmp/NanoRAFT-RL/models/cache`
    - `HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/NanoRAFT-RL/models/cache`
    - 配合 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- 结果：
  - PPO 与评估均从本地缓存完整加载并成功结束。

## 16) 缓存行为依赖命令行环境变量，对新使用者不友好
- 症状：
  - 稳定运行依赖每条命令手动设置缓存相关环境变量。
- 根因：
  - 脚本间缓存处理策略不一致。
- 修复：
  - 在以下脚本统一默认缓存行为：
    - `scripts/run_data_synthesis.py`
    - `scripts/run_ppo_training.py`
    - `scripts/run_evaluation.py`
  - 统一默认项目缓存目录：`models/cache`（同时保留外部环境变量覆盖能力）。
- 结果：
  - 首次下载后可默认复用本地缓存，无需每次额外设置。

## 17) PPO 的 `total_steps` 可能因 dataloader 耗尽提前结束
- 症状：
  - 配置期望固定步数（如 60），实际接近单轮批次数就结束。
- 根因：
  - 训练循环使用 `for ... in ppo_trainer.dataloader`，单轮遍历结束即退出。
- 修复：
  - 更新训练循环为 `for step in range(total_steps)`，在 `StopIteration` 时自动重建迭代器继续训练。
  - 为 PPO 与评估脚本加入全局随机种子设置（默认 `seed=42`）。
- 结果：
  - PPO 可严格按配置步数执行，多次运行结果更稳定、可对比性更好。

---

后续若出现新的运行问题，将继续追加到本文件。
