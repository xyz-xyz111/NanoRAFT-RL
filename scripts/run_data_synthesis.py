#!/usr/bin/env python3
"""
数据合成流水线主入口脚本

使用方法:
1. 将原始文档放入 data/raw/ 目录
2. 设置 OPENAI_API_KEY 环境变量
3. 运行: python scripts/run_data_synthesis.py [--config configs/data_synthesis.yaml]

流程:
1. 加载并切分文档
2. 构建向量索引
3. 使用教师模型生成问答对
4. 注入干扰项
5. 格式化为Alpaca格式
6. 划分训练/验证/测试集
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[Warning] tqdm未安装，将不显示进度条")

# 添加src到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_synthesis import (
    DocumentLoader,
    DistractorRetriever,
    QuestionGenerator,
    DistractorInjector,
    AlpacaFormatter,
    apply_quality_gate,
    save_quality_artifacts,
    PubMedRCTParser,
)
from src.data_synthesis.question_generator import GeneratedQA
from src.utils.logging_utils import setup_logger


def _save_qa_cache(qa_list: list, cache_file: Path):
    """保存问答对到缓存文件"""
    with open(cache_file, 'w', encoding='utf-8') as f:
        for qa in qa_list:
            # 将dataclass转换为字典
            qa_dict = {
                'question': qa.question,
                'chain_of_thought': qa.chain_of_thought,
                'answer': qa.answer,
                'reasoning_type': qa.reasoning_type,
                'source_chunk': qa.source_chunk,
                'metadata': qa.metadata,
            }
            f.write(json.dumps(qa_dict, ensure_ascii=False) + '\n')


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_pubmed_data(config: dict, logger):
    """预处理PubMed数据（如果需要）"""
    pubmed_config = config.get("pubmed", {})
    
    if not pubmed_config.get("enabled", False):
        return
    
    raw_txt = pubmed_config.get("raw_txt_path", "")
    output_dir = pubmed_config.get("output_dir", "data/processed/pubmed")
    
    # 检查是否已经预处理过
    if Path(output_dir).exists() and any(Path(output_dir).rglob("*.md")):
        logger.info(f"[PubMed] 检测到已预处理的Markdown文件，跳过预处理")
        return
    
    # 检查原始txt文件是否存在
    if not Path(raw_txt).exists():
        logger.warning(f"[PubMed] 原始txt文件不存在: {raw_txt}")
        logger.info("[PubMed] 请先从Kaggle下载数据集并解压")
        return
    
    logger.info(f"[PubMed] 开始预处理数据: {raw_txt}")
    
    try:
        parser = PubMedRCTParser(raw_txt)
        parser.parse()
        parser.to_markdown_files(
            output_dir,
            max_files=pubmed_config.get("max_articles")
        )
        logger.info(f"[PubMed] 预处理完成，输出到: {output_dir}")
    except Exception as e:
        logger.error(f"[PubMed] 预处理失败: {e}")
        raise


def setup_hf_mirror():
    """设置Hugging Face镜像（解决国内访问问题）"""
    import os
    # 优先使用hf-mirror.com镜像
    if not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[Setup] 已设置Hugging Face镜像: https://hf-mirror.com")

    # 设置统一缓存目录，保证首次下载后可复用
    default_cache_dir = PROJECT_ROOT / "models" / "cache"
    os.environ.setdefault("HF_HOME", str(default_cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(default_cache_dir))
    print(f"[Setup] 模型缓存目录: {os.environ['HF_HOME']}")
    
    # 禁用不必要的警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 禁用NLTK自动下载（避免网络错误）
    os.environ["NLTK_DATA"] = os.path.expanduser("~/nltk_data")
    
    # 创建NLTK目录结构（避免下载）
    try:
        nltk_dir = os.path.expanduser("~/nltk_data/tokenizers/punkt")
        os.makedirs(nltk_dir, exist_ok=True)
        # 创建空的english.pickle避免报错
        open(os.path.join(nltk_dir, "english.pickle"), "a").close()
        py3_dir = os.path.join(nltk_dir, "PY3")
        os.makedirs(py3_dir, exist_ok=True)
        open(os.path.join(py3_dir, "english.pickle"), "a").close()
    except:
        pass


def main(config_path: str = "configs/data_synthesis.yaml"):
    """
    数据合成主流程
    
    Args:
        config_path: 配置文件路径
    """
    config = load_config(config_path)
    main_with_config(config)


def main_with_config(config: dict):
    """
    数据合成主流程（使用已加载的配置）
    
    Args:
        config: 配置字典
    """
    # 设置镜像（在导入transformers之前）
    setup_hf_mirror()
    
    print(f"[Main] 配置加载完成")
    
    # 设置日志
    logger = setup_logger(
        name="data_synthesis",
        level=config.get("logging", {}).get("level", "INFO"),
        log_dir=config.get("logging", {}).get("log_dir", "logs/data_synthesis"),
    )
    logger.info("=" * 60)
    logger.info("RAFT 数据合成流水线启动")
    logger.info(f"时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # ============================================================
    # 步骤0: 预处理PubMed数据（如启用）
    # ============================================================
    if config.get("pubmed", {}).get("enabled", False):
        preprocess_pubmed_data(config, logger)
    
    # ============================================================
    # 步骤1 & 2: 加载文档并构建/加载索引
    # ============================================================
    
    doc_config = config["document"]
    index_dir = "data/processed/index"
    
    # 检查是否已有缓存的索引
    if Path(index_dir).exists() and any(Path(index_dir).iterdir()):
        logger.info("[步骤1-2/6] 检测到已存在的索引，正在加载...")
        
        loader = DocumentLoader(
            data_dir=doc_config["raw_dir"],
            chunk_size=doc_config["chunking"]["chunk_size"],
            chunk_overlap=doc_config["chunking"]["chunk_overlap"],
            embedding_model=config["embedding"]["model_name"],
        )
        
        try:
            index = loader.load_index(index_dir)
            nodes = loader.nodes  # 从加载的索引获取nodes
            
            # 验证索引是否可以正常检索（检查维度匹配）
            try:
                test_retriever = loader.get_retriever(similarity_top_k=1)
                test_results = test_retriever.retrieve("test query")
                logger.info(f"[步骤1-2/6] 索引验证通过，包含 {len(nodes)} 个chunks")
            except Exception as dim_e:
                if "shapes" in str(dim_e) and "not aligned" in str(dim_e):
                    logger.warning(f"[步骤1-2/6] 索引维度不匹配: {dim_e}")
                    logger.warning("[步骤1-2/6] 可能是embedding模型变更导致，将删除并重建索引...")
                    import shutil
                    shutil.rmtree(index_dir, ignore_errors=True)
                    need_rebuild = True
                else:
                    raise
        except Exception as e:
            logger.warning(f"加载索引失败: {e}，将重新构建...")
            # 删除损坏的索引
            import shutil
            shutil.rmtree(index_dir, ignore_errors=True)
            # 继续执行重新构建流程
            need_rebuild = True
    else:
        need_rebuild = True
    
    # 如果需要重新构建索引
    if 'need_rebuild' in locals() and need_rebuild:
        logger.info("[步骤1/6] 加载文档...")
        
        loader = DocumentLoader(
            data_dir=doc_config["raw_dir"],
            chunk_size=doc_config["chunking"]["chunk_size"],
            chunk_overlap=doc_config["chunking"]["chunk_overlap"],
            embedding_model=config["embedding"]["model_name"],
        )
        
        try:
            documents = loader.load_documents(
                required_exts=doc_config.get("allowed_extensions")
            )
        except FileNotFoundError as e:
            logger.error(f"文档加载失败: {e}")
            print(f"\n[Error] {e}")
            print("请检查配置中的 raw_dir 路径是否正确")
            print("\n如果使用PubMed数据，请请确保:")
            print("  1. 已从Kaggle下载 pubmed-rct-master.zip")
            print("  2. 已解压到 data/raw/")
            print("  3. 已运行: python scripts/prepare_pubmed_data.py --all ...")
            return
        
        if len(documents) == 0:
            logger.error("未找到任何文档，请检查data/raw/目录")
            return
        
        # 切分文档
        nodes = loader.split_documents()
        
        # ============================================================
        # 步骤2: 构建向量索引
        # ============================================================
        logger.info("[步骤2/6] 构建向量索引...")
        index = loader.build_index()
        
        logger.info("[步骤2/6] 保存向量索引...")
        loader.save_index(index_dir)
        logger.info("[步骤2/6] 索引保存完成")
    
    # ============================================================
    # 步骤3: 初始化问题生成器
    # ============================================================
    logger.info("[步骤3/6] 初始化问题生成器...")
    
    qg_config = config["question_generation"]
    provider = qg_config.get("provider", "openai")
    
    # 检查API Key
    if provider == "zhipu":
        api_key = qg_config.get("zhipu", {}).get("api_key") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            logger.error("使用智谱AI需要提供ZHIPU_API_KEY环境变量或在配置中设置api_key")
            print("\n[错误] 未设置智谱AI API Key")
            print("请设置环境变量: $env:ZHIPU_API_KEY='your-api-key'")
            print("或在 configs/data_synthesis.yaml 中配置 zhipu.api_key")
            return
        logger.info(f"使用智谱AI，模型: {qg_config['teacher_model']['model_name']}")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("未设置OPENAI_API_KEY环境变量...")
        logger.info(f"使用OpenAI兼容API，模型: {qg_config['teacher_model']['model_name']}")
    
    try:
        generator = QuestionGenerator(
            model_name=qg_config["teacher_model"]["model_name"],
            api_key=api_key,
            base_url=qg_config["teacher_model"].get("base_url"),
            temperature=qg_config["teacher_model"]["temperature"],
            max_retries=qg_config["teacher_model"]["max_retries"],
            provider=provider,
        )
        logger.info(f"教师模型初始化成功: {provider} / {qg_config['teacher_model']['model_name']}")
    except Exception as e:
        logger.error(f"教师模型初始化失败: {e}")
        if config.get("local_fallback", {}).get("enabled"):
            logger.info("切换到本地模型...")
            generator = QuestionGenerator.LocalModelGenerator(
                model_path=config["local_fallback"]["model_name"],
                device=config["local_fallback"]["device"],
            )
        else:
            raise
    
    # ============================================================
    # 步骤4: 生成问答对（支持断点续传）
    # ============================================================
    
    # 检查是否有缓存的问答对
    cache_dir = Path("data/processed/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    qa_cache_file = cache_dir / "qa_list_cache.jsonl"
    
    # 选择用于生成问题的chunks（不必全部使用，可采样）
    chunks_for_generation = [node.text for node in nodes]
    
    # 根据配置限制chunks数量（用于测试）
    max_chunks = qg_config["strategy"].get("max_chunks")
    if max_chunks and len(chunks_for_generation) > max_chunks:
        logger.info(f"[测试模式] 限制处理前 {max_chunks} 个chunks（总计 {len(chunks_for_generation)} 个）")
        chunks_for_generation = chunks_for_generation[:max_chunks]
    
    # 检查是否使用缓存
    qa_list = []
    if not config.get("force_regenerate", False) and qa_cache_file.exists():
        logger.info(f"[步骤4/6] 发现缓存的问答对: {qa_cache_file}")
        try:
            with open(qa_cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    qa_data = json.loads(line.strip())
                    from src.data_synthesis.question_generator import GeneratedQA
                    qa = GeneratedQA(**qa_data)
                    qa_list.append(qa)
            logger.info(f"[步骤4/6] 从缓存加载了 {len(qa_list)} 个问答对")
            print(f"✅ 从缓存加载 {len(qa_list)} 个问答对，跳过API调用")
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}，将重新生成")
            qa_list = []
    
    # 如果需要重新生成
    if not qa_list:
        logger.info("[步骤4/6] 生成问答对...")
        logger.info(f"将为 {len(chunks_for_generation)} 个chunks生成问答对")
        logger.info("⚠️  此步骤调用智谱AI API，可能需要较长时间，请耐心等待...")
        
        # 使用进度条包装生成过程
        chunk_iterator = chunks_for_generation
        
        if TQDM_AVAILABLE:
            chunk_iterator = tqdm(chunks_for_generation, desc="生成问答对", unit="chunk")
        
        for chunk in chunk_iterator:
            try:
                qa = generator.generate_from_chunk(
                    chunk_text=chunk,
                    max_length=qg_config["strategy"]["max_chunk_length"],
                )
                if qa:
                    qa_list.append(qa)
                
                # 每生成10个打印一次日志
                if len(qa_list) % 10 == 0 and not TQDM_AVAILABLE:
                    logger.info(f"已生成 {len(qa_list)} 个问答对...")
                
                # 每生成50个保存一次缓存
                if len(qa_list) % 50 == 0:
                    _save_qa_cache(qa_list, qa_cache_file)
                    
            except Exception as e:
                logger.warning(f"生成问答对失败: {e}")
                continue
        
        # 最终保存缓存
        if qa_list:
            _save_qa_cache(qa_list, qa_cache_file)
            logger.info(f"[步骤4/6] 问答对已缓存到: {qa_cache_file}")
    
    logger.info(f"[步骤4/6] 成功准备 {len(qa_list)} 个问答对")
    
    if len(qa_list) == 0:
        logger.error("未生成任何问答对，请检查API配置和文档内容")
        return
    
    # ============================================================
    # 步骤5: 注入干扰项
    # ============================================================
    logger.info("[步骤5/6] 注入干扰项...")
    
    # 初始化检索器（用于获取干扰项）
    retriever = DistractorRetriever(
        index_retriever=loader.get_retriever(similarity_top_k=10),
        num_distractors=config["distractor"]["total_contexts"] - 1,  # 减1留给golden
        shuffle=config["distractor"]["shuffle"],
        strategy=config["distractor"]["selection_strategy"],
    )
    retriever.set_all_nodes(nodes)
    
    # 初始化注入器
    injector = DistractorInjector(
        difficulty=config["distractor"]["difficulty"],
        shuffle=config["distractor"]["shuffle"],
    )
    
    # 为每个QA对注入干扰项
    injected_contexts = []
    successful_qa_list = []
    
    # 使用进度条
    qa_iterator = enumerate(qa_list)
    if TQDM_AVAILABLE:
        qa_iterator = tqdm(list(enumerate(qa_list)), desc="注入干扰项", unit="sample")
    
    for i, qa in qa_iterator:
        if not TQDM_AVAILABLE and i % 10 == 0:
            logger.info(f"处理 {i+1}/{len(qa_list)}...")
        
        try:
            # 使用注入器构建带干扰项的上下文
            golden_chunk = qa.source_chunk
            
            injected = injector.inject_with_retrieval(
                query=qa.question,
                golden_chunk=golden_chunk,
                retriever=retriever,
                total_contexts=config["distractor"]["total_contexts"],
            )
            
            injected_contexts.append(injected)
            successful_qa_list.append(qa)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.warning(f"注入干扰项失败 (sample {i}): {e}")
            logger.debug(f"详细错误: {error_detail}")
            
            # 如果是前5个样本失败，打印详细错误帮助调试
            if i < 5:
                print(f"\n[Debug] Sample {i} 注入失败详情:")
                print(f"  问题: {qa.question[:100]}...")
                print(f"  来源chunk长度: {len(qa.source_chunk)}")
                print(f"  错误: {e}")
            continue
    
    logger.info(f"成功注入干扰项: {len(injected_contexts)}/{len(qa_list)}")
    
    # ============================================================
    # 步骤6: 格式化为Alpaca格式并保存
    # ============================================================
    logger.info("[步骤6/6] 格式化并保存数据集...")
    
    formatter = AlpacaFormatter(
        cot_style=config["output"]["cot_style"],
        include_metadata=config["output"]["include_metadata"],
        dataset_name=config["output"]["dataset_name"],
    )
    
    # 格式化（使用进度条）
    samples = []
    # 只配对实际注入成功的QA，避免中间失败导致问题与上下文错位。
    format_iterator = zip(successful_qa_list, injected_contexts)
    
    if TQDM_AVAILABLE:
        format_iterator = tqdm(list(format_iterator), desc="格式化数据", unit="sample")
    
    for qa, ctx in format_iterator:
        try:
            sample = formatter.format_single(qa, ctx)
            samples.append(sample)
        except Exception as e:
            logger.warning(f"格式化失败: {e}")
            continue
    
    logger.info(f"格式化完成: {len(samples)} 个样本")
    
    # 保存
    output_dir = config["output"]["output_dir"]

    # 质量门：不合格样本只进入rejected_samples.jsonl，不参与训练划分。
    quality_config = config.get("quality_gate", {})
    if quality_config.get("enabled", True):
        quality_result = apply_quality_gate(
            samples,
            min_question_chars=quality_config.get("min_question_chars", 20),
            min_context_chars=quality_config.get("min_context_chars", 100),
            min_output_chars=quality_config.get("min_output_chars", 50),
        )
        save_quality_artifacts(quality_result, output_dir)
        samples = quality_result.accepted
        logger.info(f"质量门结果: {quality_result.report}")
        if not samples:
            raise RuntimeError("质量门过滤后没有可用于训练的样本")
    
    # 划分并保存
    formatter.split_and_save(
        samples=samples,
        output_dir=output_dir,
        train_ratio=config["output"]["split"]["train_ratio"],
        val_ratio=config["output"]["split"]["val_ratio"],
        seed=config["output"]["split"]["seed"],
    )
    
    # 同时保存完整数据集（用于备份）
    formatter.save_to_jsonl(
        samples=samples,
        output_path=f"{output_dir}/all_samples.jsonl",
    )
    
    # ============================================================
    # 完成统计
    # ============================================================
    logger.info("=" * 60)
    logger.info("数据合成完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"总样本数: {len(samples)}")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("✅ 数据合成流水线完成!")
    print(f"📊 总样本数: {len(samples)}")
    print(f"📁 输出目录: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAFT数据合成流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/run_data_synthesis.py
  
  # 使用自定义配置
  python scripts/run_data_synthesis.py --config my_config.yaml
  
  # 强制重新生成（忽略缓存）
  python scripts/run_data_synthesis.py --force-regenerate
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_synthesis.yaml",
        help="配置文件路径 (默认: configs/data_synthesis.yaml)"
    )
    
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="强制重新生成问答对（忽略缓存）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用命令行参数覆盖
    if args.force_regenerate:
        config["force_regenerate"] = True
        print("[参数] 强制重新生成模式：将忽略缓存的问答对")
    
    main_with_config(config)
