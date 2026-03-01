"""
build_index.py
==============
向量索引预构建脚本。

为了加速实验，此脚本预先为所有数据集的对话历史构建 FAISS 向量索引，
并将索引保存到 data/processed/ 目录。在运行实验时，可以直接加载预构建的索引，
而无需在每次实验时重新构建。

用法：
    cd /path/to/uniq_cluster_memory
    PYTHONPATH=. python3 scripts/build_index.py \\
        --dataset longmemeval \\
        --max_samples 50
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.vectorstores import FAISS
from src.uniq_cluster_memory.utils.embeddings import LocalEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from benchmarks.longmemeval_task import LongMemEvalTask
from benchmarks.meddialog_task import MedDialogTask


DATASET_REGISTRY = {
    "longmemeval": {
        "class": LongMemEvalTask,
        "path": "data/raw/longmemeval/longmemeval_oracle.json",
    },
    "meddialog": {
        "class": MedDialogTask,
        "path": "data/raw/meddialog/meddialog_zh_sample50.json",
    },
}


def build_index(
    dataset_name: str,
    max_samples: int | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embedding_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "data/processed",
) -> None:
    """
    为指定数据集构建 FAISS 向量索引。

    Args:
        dataset_name: 数据集名称。
        max_samples: 最大样本数量。
        chunk_size: Chunk 大小。
        chunk_overlap: Chunk 重叠大小。
        embedding_model: OpenAI 嵌入模型名称。
        output_dir: 索引保存目录。
    """
    print(f"Building FAISS index for {dataset_name}...")

    # 加载数据集
    cfg = DATASET_REGISTRY[dataset_name]
    task = cfg["class"](data_path=cfg["path"], max_samples=max_samples)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples")

    # 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", " ", ""],
    )

    all_docs = []
    for sample in tqdm(samples, desc="Chunking"):
        for i, turn in enumerate(sample.dialog_history):
            content = f"[{turn.role.upper()}]: {turn.content}"
            doc = Document(
                page_content=content,
                metadata={
                    "sample_id": sample.sample_id,
                    "turn_index": i,
                    "role": turn.role,
                    "timestamp": turn.timestamp or "",
                    "has_answer": turn.has_answer or False,
                },
            )
            all_docs.append(doc)

    split_docs = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(split_docs)}")

    # 构建 FAISS 索引
    embeddings = LocalEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 保存索引
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"{dataset_name}_faiss_index")
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved to {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index for datasets")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), default="longmemeval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    build_index(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
