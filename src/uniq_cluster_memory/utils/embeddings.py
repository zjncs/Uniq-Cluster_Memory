"""
embeddings.py
=============
本地 Embedding 适配器。

由于当前环境的 OpenAI API 代理不支持 Embedding 模型，
此模块提供一个基于 sentence-transformers 的本地 Embedding 实现，
并封装为与 LangChain Embeddings 接口兼容的类。

在生产环境中，可以直接替换为 OpenAIEmbeddings。
"""

from typing import List
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """
    基于 sentence-transformers 的本地 Embedding 实现。
    兼容 LangChain Embeddings 接口，可直接替换 OpenAIEmbeddings。

    Args:
        model_name: sentence-transformers 模型名称。
                    默认使用 'all-MiniLM-L6-v2'（轻量级，适合快速验证）。
                    生产环境建议替换为 BioLORD（医疗领域专用）。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行 Embedding。"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行 Embedding。"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
