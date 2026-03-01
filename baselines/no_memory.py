"""
no_memory.py
============
无记忆基线（No-Memory Baseline）。

这是最简单的基线：直接将问题发送给 LLM，不提供任何对话历史或检索上下文。
它代表了"没有任何记忆系统"时模型的表现下界。
"""

import os
from openai import OpenAI

from benchmarks.base_task import UnifiedSample
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


def run_no_memory(sample: UnifiedSample, model: str = LLM_MODEL) -> str:
    """
    无记忆基线：直接用问题调用 LLM，不提供任何上下文。

    Args:
        sample: 统一样本格式。
        model: 使用的 LLM 模型名称。

    Returns:
        LLM 生成的答案字符串。
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Answer the question based on your knowledge.",
            },
            {"role": "user", "content": sample.question},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
