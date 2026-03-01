"""
recursive_summary.py
====================
递归摘要基线（Recursive Summary Baseline）。

这个基线将整个对话历史递归地压缩为一个摘要，然后将摘要作为上下文提供给 LLM。
它代表了"全局压缩式记忆"的方法，与我们的"结构化唯一性记忆"形成对比。

流程：
    Dialog History -> Chunked Summaries -> Merged Summary -> LLM Answer

这个基线的核心弱点是：
1. 压缩过程中可能丢失细节（尤其是数值型信息，如检查结果）。
2. 无法处理信息更新（新信息和旧信息会被混合在摘要中）。
3. 无法检测和记录冲突。
"""

import os
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from benchmarks.base_task import UnifiedSample
from src.uniq_cluster_memory.utils.llm_client import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    LLM_MODEL,
)


SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Summarize the following conversation excerpt, "
        "focusing on key medical facts, symptoms, diagnoses, and treatment plans. "
        "Be concise but retain all important details.",
    ),
    ("human", "Conversation:\n{text}\n\nSummary:"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful medical assistant. "
        "Answer the question based ONLY on the provided summary of the conversation history.",
    ),
    (
        "human",
        "Summary of conversation history:\n---\n{summary}\n---\n\nQuestion: {question}\n\nAnswer:",
    ),
])


class RecursiveSummaryBaseline:
    """
    递归摘要基线实现。

    将对话历史分块后逐块摘要，然后将所有块的摘要合并为最终摘要，
    最后基于最终摘要回答问题。
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        llm_model: str = LLM_MODEL,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", " ", ""],
        )
        self._llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )

    def _history_to_text(self, sample: UnifiedSample) -> str:
        """将对话历史转换为纯文本。"""
        lines = []
        for turn in sample.dialog_history:
            prefix = f"[{turn.timestamp}] " if turn.timestamp else ""
            lines.append(f"{prefix}[{turn.role.upper()}]: {turn.content}")
        return "\n".join(lines)

    def _summarize_chunk(self, text: str) -> str:
        """对单个文本块进行摘要。"""
        chain = SUMMARIZE_PROMPT | self._llm
        response = chain.invoke({"text": text})
        return response.content.strip()

    def run(self, sample: UnifiedSample) -> dict:
        """
        对单个样本执行递归摘要 RAG 流程。

        Args:
            sample: 统一样本格式。

        Returns:
            包含以下字段的字典：
            - "answer": LLM 生成的答案字符串。
            - "summary": 最终生成的对话摘要（用于调试和分析）。
        """
        # Step 1: 将对话历史转换为纯文本并切分
        full_text = self._history_to_text(sample)
        chunks = self._text_splitter.split_text(full_text)

        if not chunks:
            return {"answer": "No conversation history available.", "summary": ""}

        # Step 2: 逐块摘要
        chunk_summaries = [self._summarize_chunk(chunk) for chunk in chunks]

        # Step 3: 合并摘要（如果摘要过长，再做一次摘要）
        merged_text = "\n\n".join(chunk_summaries)
        if len(merged_text) > self.chunk_size * 2:
            final_summary = self._summarize_chunk(merged_text)
        else:
            final_summary = merged_text

        # Step 4: 基于最终摘要回答问题
        chain = QA_PROMPT | self._llm
        response = chain.invoke({"summary": final_summary, "question": sample.question})
        answer = response.content.strip()

        return {"answer": answer, "summary": final_summary}
