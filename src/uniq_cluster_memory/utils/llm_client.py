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

# Qwen DashScope 配置
QWEN_API_KEY = "sk-4902b4cfaf5a48ecbe9b8458c61d1bb7"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"          # 主力模型：Qwen-Plus（高质量）
LLM_MODEL_FAST = "qwen-turbo"    # 快速模型：Qwen-Turbo（低延迟）


def get_llm_client() -> OpenAI:
    """返回配置好的 Qwen LLM 客户端。"""
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )
