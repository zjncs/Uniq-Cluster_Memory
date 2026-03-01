"""
raw_rag.py
==========
Raw-RAG 基线（基于 LangChain 实现）。

这是核心对比基线：将对话历史切分为 Chunks，建立向量索引（FAISS），
然后通过语义检索找到最相关的 Chunks，最后将其作为上下文提供给 LLM 生成答案。

流程：
    Dialog History -> Chunks -> FAISS Index -> Retrieval (Top-K) -> LLM Answer

这个基线代表了"标准 RAG"的表现，是我们方法需要超越的主要对比对象。
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

from benchmarks.base_task import UnifiedSample


@dataclass
class RAGConfig:
    """Raw-RAG 基线的配置参数。"""
    chunk_size: int = 512          # 每个 Chunk 的最大 token 数（近似字符数）
    chunk_overlap: int = 64        # 相邻 Chunk 之间的重叠字符数
    top_k: int = 5                 # 检索返回的最相关 Chunk 数量
    embedding_model: str = "all-MiniLM-L6-v2"  # 本地嵌入模型（可替换为 BioLORD）
    llm_model: str = LLM_MODEL  # 用于生成答案的 LLM（默认 Qwen）


# 用于生成答案的 Prompt 模板
QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful medical assistant. "
            "Answer the question based ONLY on the provided context from the conversation history. "
            "If the answer cannot be found in the context, say 'I cannot find this information in the conversation history.'"
        ),
    ),
    (
        "human",
        (
            "Context from conversation history:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    ),
])


class RawRAGBaseline:
    """
    Raw-RAG 基线实现。

    每次调用 run() 时，会为给定的 UnifiedSample 构建一个临时的向量索引，
    然后检索最相关的 Chunks，最后生成答案。

    注意：为了支持 Recall@K 评测，run() 方法同时返回答案和检索到的 Chunks。
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
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
        """
        将 UnifiedSample 的对话历史转换为 LangChain Document 列表。

        每个 turn 被转换为一个 Document，其 metadata 包含角色、时间戳和 has_answer 标记。
        """
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
        return docs

    def run(self, sample: UnifiedSample) -> dict:
        """
        对单个样本执行 RAG 流程。

        Args:
            sample: 统一样本格式。

        Returns:
            包含以下字段的字典：
            - "answer": LLM 生成的答案字符串。
            - "retrieved_docs": 检索到的 Document 列表（用于 Recall@K 评测）。
            - "retrieved_has_answer": bool 列表，标记每个检索到的 Chunk 是否包含答案证据。
            - "total_answer_chunks": 当前样本中含答案证据的 Chunk 总数（Recall@K 分母）。
        """
        # Step 1: 将对话历史转换为 Documents 并切分
        raw_docs = self._build_documents(sample)
        split_docs = self._text_splitter.split_documents(raw_docs)

        if not split_docs:
            return {
                "answer": "No conversation history available.",
                "retrieved_docs": [],
                "retrieved_has_answer": [],
                "total_answer_chunks": 0,
            }

        # Step 2: 构建 FAISS 向量索引
        vectorstore = FAISS.from_documents(split_docs, self._embeddings)

        # Step 3: 检索 Top-K 最相关的 Chunks
        top_k = min(self.config.top_k, len(split_docs))
        retrieved_docs = vectorstore.similarity_search(sample.question, k=top_k)

        # Step 4: 构建上下文并调用 LLM
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        chain = QA_PROMPT | self._llm
        response = chain.invoke({"context": context, "question": sample.question})
        answer = response.content.strip()

        # 记录每个检索到的 Chunk 是否包含答案证据（用于 Recall@K 评测）
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


if __name__ == "__main__":
    # 快速验证：对 LongMemEval 的第一条样本运行 Raw-RAG
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from benchmarks.longmemeval_task import LongMemEvalTask

    data_file = "data/raw/longmemeval/longmemeval_oracle.json"
    task = LongMemEvalTask(data_path=data_file, max_samples=2)
    samples = task.get_samples()

    rag = RawRAGBaseline()
    result = rag.run(samples[0])

    print(f"Question : {samples[0].question}")
    print(f"GT Answer: {samples[0].answer}")
    print(f"RAG Answer: {result['answer']}")
    print(f"Retrieved {len(result['retrieved_docs'])} chunks")
    print(f"Any chunk has answer: {any(result['retrieved_has_answer'])}")
