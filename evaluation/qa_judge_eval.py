"""
qa_judge_eval.py
================
QA 质量评测模块（LLM-as-a-Judge）。

使用 LLM 作为裁判，对系统生成的答案与标准答案进行比较，
输出一个二元判断（正确/错误）和一个 1-5 分的质量评分。

这是评测 RAG 系统端到端问答质量的核心指标，
也是论文中"第三优先指标"（QA LLM-as-Judge）的具体实现。

评测维度：
    - Correctness (0/1): 答案是否与标准答案在语义上一致。
    - Quality (1-5): 答案的整体质量（准确性、完整性、简洁性）。
"""

import json
import os
from dataclasses import dataclass
from typing import Optional
import re

from openai import OpenAI
from src.uniq_cluster_memory.utils.llm_client import (
    get_llm_client,
    LLM_MODEL,
)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = get_llm_client()
    return _client


@dataclass
class QAJudgeResult:
    """单次 LLM-as-a-Judge 评测结果。"""
    sample_id: str
    question: str
    gt_answer: str
    hypothesis: str
    correctness: int    # 0 或 1
    quality_score: int  # 1-5
    reasoning: str      # 裁判的推理过程


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems in the medical domain.
Your task is to evaluate whether a hypothesis answer correctly answers the question, given the ground truth answer.

You must respond with a JSON object containing:
- "correctness": 1 if the hypothesis is semantically correct (matches the key facts in the ground truth), 0 otherwise.
- "quality_score": An integer from 1 to 5, where:
    1 = Completely wrong or irrelevant
    2 = Partially correct but missing key information
    3 = Mostly correct with minor errors or omissions
    4 = Correct and complete
    5 = Correct, complete, and well-articulated
- "reasoning": A brief explanation (1-2 sentences) for your evaluation.

Be strict about factual accuracy, especially for medical information (e.g., drug names, dosages, diagnoses).
"""

JUDGE_USER_PROMPT = """Question: {question}

Ground Truth Answer: {gt_answer}

Hypothesis Answer: {hypothesis}

Evaluate the hypothesis answer and respond with a JSON object."""


def judge_single(
    sample_id: str,
    question: str,
    gt_answer: str,
    hypothesis: str,
    model: str = LLM_MODEL,
) -> QAJudgeResult:
    """
    使用 LLM 对单个答案进行评测。

    Args:
        sample_id: 样本 ID。
        question: 问题。
        gt_answer: 标准答案。
        hypothesis: 系统生成的答案。
        model: 用于评测的 LLM 模型。

    Returns:
        QAJudgeResult 对象。
    """
    client = _get_client()
    user_content = JUDGE_USER_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        hypothesis=hypothesis,
    )

    try:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # 优先使用 JSON mode；若不支持则降级到普通输出解析
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
        except Exception:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            raw = response.choices[0].message.content

        raw = (raw or "").strip()
        result_dict = _parse_judge_json(raw)
        correctness = int(result_dict.get("correctness", 0))
        quality_score = int(result_dict.get("quality_score", 1))
        reasoning = result_dict.get("reasoning", "")
    except Exception as e:
        # 如果 LLM 调用失败，返回默认的失败结果
        correctness = 0
        quality_score = 1
        reasoning = f"Evaluation failed: {str(e)}"

    return QAJudgeResult(
        sample_id=sample_id,
        question=question,
        gt_answer=gt_answer,
        hypothesis=hypothesis,
        correctness=correctness,
        quality_score=quality_score,
        reasoning=reasoning,
    )


def _parse_judge_json(raw: str) -> dict:
    """从裁判模型输出中尽可能鲁棒地解析 JSON 结果。"""
    if not raw:
        return {"correctness": 0, "quality_score": 1, "reasoning": "Empty judge output."}

    # 直接 JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 代码块包裹 JSON
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue

    # 抽取第一个 {...} 片段尝试解析
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {"correctness": 0, "quality_score": 1, "reasoning": f"Unparseable judge output: {raw[:200]}"}


def aggregate_qa_metrics(results: list[QAJudgeResult]) -> dict:
    """
    聚合多个样本的 QA 评测结果。

    Args:
        results: 多个样本的 QAJudgeResult 列表。

    Returns:
        包含聚合指标的字典。
    """
    if not results:
        return {"accuracy": 0.0, "mean_quality": 0.0, "n_samples": 0}

    accuracy = sum(r.correctness for r in results) / len(results)
    mean_quality = sum(r.quality_score for r in results) / len(results)

    return {
        "accuracy": round(accuracy, 4),
        "mean_quality_score": round(mean_quality, 4),
        "n_samples": len(results),
        "n_correct": sum(r.correctness for r in results),
    }
