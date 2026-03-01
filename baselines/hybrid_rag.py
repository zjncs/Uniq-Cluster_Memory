"""
hybrid_rag.py
=============
混合检索基线（Hybrid RAG Baseline）。

结合 BM25（稀疏检索）和 FAISS（密集检索）进行混合检索，
通过 Reciprocal Rank Fusion (RRF) 融合两种检索结果。

这个基线比 Raw-RAG 更强，代表了"最佳工程化 RAG"的水平，
是我们方法需要超越的更高标准。

流程：
    Dialog History -> Chunks
        -> BM25 Retrieval (Sparse)  \
                                     -> RRF Fusion -> Top-K Docs -> LLM Answer
        -> FAISS Retrieval (Dense)  /
"""

import os
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from src.uniq_cluster_memory.utils.embeddings import LocalEmbeddings
from src.uniq_cluster_memory.utils.llm_client import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    LLM_MODEL,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from benchmarks.base_task import UnifiedSample


@dataclass
class HybridRAGConfig:
    """混合 RAG 基线的配置参数。"""
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5                        # 最终返回的 Chunk 数量
    bm25_top_k: int = 10                  # BM25 初步检索数量
    dense_top_k: int = 10                 # FAISS 初步检索数量
    rrf_k: int = 60                       # RRF 融合参数（标准值为 60）
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = LLM_MODEL


QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful medical assistant. "
            "Answer the question based ONLY on the provided context from the conversation history. "
            "If the answer cannot be found in the context, say 'I cannot find this information.'"
        ),
    ),
    (
        "human",
        "Context:\n---\n{context}\n---\n\nQuestion: {question}\n\nAnswer:",
    ),
])


class HybridRAGBaseline:
    """
    混合检索 RAG 基线实现（BM25 + FAISS + RRF）。
    """

    def __init__(self, config: Optional[HybridRAGConfig] = None):
        self.config = config or HybridRAGConfig()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        self._embeddings = LocalEmbeddings(model_name=self.config.embedding_model)
        self._llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=0.0,
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )

    def _build_documents(self, sample: UnifiedSample) -> list[Document]:
        """将对话历史转换为 LangChain Document 列表并切分。"""
        docs = []
        for i, turn in enumerate(sample.dialog_history):
            content = f"[{turn.role.upper()}]: {turn.content}"
            doc = Document(
                page_content=content,
                metadata={
                    "turn_index": i,
                    "role": turn.role,
                    "timestamp": turn.timestamp or "",
                    "has_answer": turn.has_answer or False,
                    "sample_id": sample.sample_id,
                },
            )
            docs.append(doc)
        return self._text_splitter.split_documents(docs)

    @staticmethod
    def _rrf_fusion(
        bm25_results: list[tuple[int, float]],
        dense_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF) 融合两种检索结果。

        Args:
            bm25_results: BM25 检索结果，每项为 (doc_index, score)。
            dense_results: FAISS 检索结果，每项为 (doc_index, score)。
            k: RRF 参数，通常为 60。

        Returns:
            融合后的 (doc_index, rrf_score) 列表，按分数降序排列。
        """
        scores: dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(bm25_results):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

        for rank, (doc_idx, _) in enumerate(dense_results):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def run(self, sample: UnifiedSample) -> dict:
        """
        对单个样本执行混合检索 RAG 流程。

        Args:
            sample: 统一样本格式。

        Returns:
            包含以下字段的字典：
            - "answer": LLM 生成的答案字符串。
            - "retrieved_docs": 检索到的 Document 列表。
            - "retrieved_has_answer": bool 列表，标记每个 Chunk 是否包含答案证据。
            - "total_answer_chunks": 当前样本中含答案证据的 Chunk 总数（Recall@K 分母）。
        """
        split_docs = self._build_documents(sample)
        for idx, doc in enumerate(split_docs):
            doc.metadata["doc_idx"] = idx

        if not split_docs:
            return {
                "answer": "No conversation history available.",
                "retrieved_docs": [],
                "retrieved_has_answer": [],
                "total_answer_chunks": 0,
            }

        # --- BM25 稀疏检索 ---
        tokenized_corpus = [doc.page_content.split() for doc in split_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = sample.question.split()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_k = min(self.config.bm25_top_k, len(split_docs))
        bm25_results = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:bm25_top_k]

        # --- FAISS 密集检索 ---
        vectorstore = FAISS.from_documents(split_docs, self._embeddings)
        dense_top_k = min(self.config.dense_top_k, len(split_docs))
        dense_results_raw = vectorstore.similarity_search_with_score(
            sample.question, k=dense_top_k
        )
        # 将 FAISS 结果映射回原始 doc 索引（使用稳定 doc_idx，避免重复内容冲突）
        dense_results = [
            (int(doc.metadata.get("doc_idx", -1)), score)
            for doc, score in dense_results_raw
            if int(doc.metadata.get("doc_idx", -1)) != -1
        ]

        # --- RRF 融合 ---
        fused = self._rrf_fusion(bm25_results, dense_results, k=self.config.rrf_k)
        top_k = min(self.config.top_k, len(fused))
        retrieved_docs = [split_docs[idx] for idx, _ in fused[:top_k]]

        # --- LLM 生成答案 ---
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        chain = QA_PROMPT | self._llm
        response = chain.invoke({"context": context, "question": sample.question})
        answer = response.content.strip()

        retrieved_has_answer = [
            doc.metadata.get("has_answer", False) for doc in retrieved_docs
        ]
        total_answer_chunks = sum(
            1 for doc in split_docs if doc.metadata.get("has_answer", False)
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "retrieved_has_answer": retrieved_has_answer,
            "total_answer_chunks": total_answer_chunks,
        }
