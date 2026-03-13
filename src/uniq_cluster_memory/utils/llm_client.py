"""
llm_client.py
=============
统一的 LLM 客户端工厂，支持 Qwen（DashScope）和 OpenAI 兼容接口。

使用方式：
    from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL

    client = get_llm_client()
    response = client.chat.completions.create(model=LLM_MODEL, ...)
"""

import os
from openai import OpenAI


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Qwen DashScope 配置
QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")          # 主力模型：Qwen-Plus（高质量）
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "qwen-turbo")    # 快速模型：Qwen-Turbo（低延迟）
LLM_TIMEOUT_SECONDS = _get_env_float("LLM_TIMEOUT_SECONDS", 60.0)
LLM_MAX_RETRIES = _get_env_int("LLM_MAX_RETRIES", 2)


def get_llm_client() -> OpenAI:
    """返回配置好的 Qwen LLM 客户端。"""
    if not QWEN_API_KEY:
        raise RuntimeError(
            "Missing LLM API key. Set QWEN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY."
        )
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
