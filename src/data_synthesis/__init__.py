"""
数据合成流水线模块

本模块负责：
1. 使用LlamaIndex加载并切分本地文档
2. 利用LangChain编排教师模型生成问题
3. 注入干扰项(Distractors)构建抗噪训练数据
4. 输出Alpaca格式数据（含Chain-of-Thought）

数据格式：
{
    "instruction": "问题描述",
    "input": "上下文（含干扰项）",
    "output": "推理过程 + 答案"
}
"""

from importlib import import_module


_LAZY_IMPORTS = {
    "DocumentLoader": (".document_loader", "DocumentLoader"),
    "DistractorRetriever": (".retriever", "DistractorRetriever"),
    "QuestionGenerator": (".question_generator", "QuestionGenerator"),
    "DistractorInjector": (".distractor_injector", "DistractorInjector"),
    "AlpacaFormatter": (".alpaca_formatter", "AlpacaFormatter"),
    "apply_quality_gate": (".quality_gate", "apply_quality_gate"),
    "save_quality_artifacts": (".quality_gate", "save_quality_artifacts"),
    "PubMedRCTParser": (".pubmed_parser", "PubMedRCTParser"),
    "convert_pubmed_to_format": (".pubmed_parser", "convert_pubmed_to_format"),
}


def __getattr__(name):
    """按需导入，避免预处理阶段被无关的检索/GPU依赖阻塞。"""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "DocumentLoader",
    "DistractorRetriever", 
    "QuestionGenerator",
    "DistractorInjector",
    "AlpacaFormatter",
    "apply_quality_gate",
    "save_quality_artifacts",
    "PubMedRCTParser",
    "convert_pubmed_to_format",
]
