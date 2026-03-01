"""
retrieval_eval.py
=================
检索质量评测模块。

计算 Recall@K 指标：衡量检索到的 Top-K Chunks 中，
是否包含了回答问题所需的关键证据（has_answer=True 的 turn）。

Recall@K = (包含答案证据的检索结果数量) / (总的包含答案证据的 Chunk 数量)

这个指标直接衡量了检索模块找到"关键证据"的能力，
是评测 RAG 系统检索质量的核心指标。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalMetrics:
    """检索评测结果。"""
    recall_at_k: float          # Recall@K
    k: int                      # K 值
    n_relevant_retrieved: int   # 检索到的包含答案的 Chunk 数量
    n_relevant_total: int       # 总的包含答案的 Chunk 数量（在整个对话历史中）
    n_retrieved: int            # 实际检索到的 Chunk 数量


def compute_recall_at_k(
    retrieved_has_answer: list[bool],
    total_answer_chunks: Optional[int] = None,
) -> RetrievalMetrics:
    """
    计算单个样本的 Recall@K。

    Args:
        retrieved_has_answer: 每个检索到的 Chunk 是否包含答案证据的 bool 列表。
        total_answer_chunks: 整个对话历史中包含答案证据的 Chunk 总数。
                             必须提供真实值，不能使用近似分母。

    Returns:
        RetrievalMetrics 对象。
    """
    k = len(retrieved_has_answer)
    n_relevant_retrieved = sum(1 for has_ans in retrieved_has_answer if has_ans)

    if total_answer_chunks is None:
        raise ValueError("total_answer_chunks is required to compute Recall@K.")
    if total_answer_chunks < 0:
        raise ValueError("total_answer_chunks must be non-negative.")

    denominator = total_answer_chunks
    if denominator == 0:
        recall = 0.0
    else:
        recall = n_relevant_retrieved / denominator

    return RetrievalMetrics(
        recall_at_k=recall,
        k=k,
        n_relevant_retrieved=n_relevant_retrieved,
        n_relevant_total=denominator,
        n_retrieved=k,
    )


def aggregate_retrieval_metrics(metrics_list: list[RetrievalMetrics]) -> dict:
    """
    聚合多个样本的检索评测结果，计算宏平均（Macro Average）。

    Args:
        metrics_list: 多个样本的 RetrievalMetrics 列表。

    Returns:
        包含聚合指标的字典。
    """
    if not metrics_list:
        return {"mean_recall_at_k": 0.0, "n_samples": 0}

    mean_recall = sum(m.recall_at_k for m in metrics_list) / len(metrics_list)
    k_value = metrics_list[0].k  # 假设所有样本使用相同的 K

    return {
        "mean_recall_at_k": round(mean_recall, 4),
        f"recall_at_{k_value}": round(mean_recall, 4),
        "n_samples": len(metrics_list),
        "n_with_evidence": sum(1 for m in metrics_list if m.n_relevant_total > 0),
    }
