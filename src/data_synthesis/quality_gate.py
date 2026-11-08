"""合成RAFT样本的确定性质量门。"""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class QualityResult:
    accepted: List[Any]
    rejected: List[Dict[str, Any]]
    report: Dict[str, Any]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def apply_quality_gate(
    samples: Iterable[Any],
    min_question_chars: int = 20,
    min_context_chars: int = 100,
    min_output_chars: int = 50,
) -> QualityResult:
    """过滤结构错误、长度异常、重复和证据缺失的样本。"""
    accepted, rejected = [], []
    reason_counts: Counter = Counter()
    seen_questions = set()
    samples = list(samples)

    for sample in samples:
        reasons = []
        question = _normalize(sample.instruction)
        context = _normalize(sample.input)
        output = _normalize(sample.output)
        metadata = sample.metadata or {}

        if len(question) < min_question_chars:
            reasons.append("question_too_short")
        if len(context) < min_context_chars:
            reasons.append("context_too_short")
        if len(output) < min_output_chars:
            reasons.append("output_too_short")
        if not metadata.get("source_group_id"):
            reasons.append("missing_source_group_id")
        if metadata.get("golden_context_present") is False:
            reasons.append("golden_context_missing")
        if question in seen_questions:
            reasons.append("duplicate_question")

        if reasons:
            reason_counts.update(reasons)
            rejected.append({"reasons": reasons, "sample": sample.to_dict()})
        else:
            accepted.append(sample)
            seen_questions.add(question)

    total = len(samples)
    report = {
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / total, 4) if total else 0.0,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "thresholds": {
            "min_question_chars": min_question_chars,
            "min_context_chars": min_context_chars,
            "min_output_chars": min_output_chars,
        },
    }
    return QualityResult(accepted, rejected, report)


def save_quality_artifacts(result: QualityResult, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(result.report, f, ensure_ascii=False, indent=2)
    with open(output_path / "rejected_samples.jsonl", "w", encoding="utf-8") as f:
        for item in result.rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
