"""
base_task.py
============
定义所有 benchmark task 的统一数据接口基类。

所有 benchmark（MedDialog, LongMemEval, Med-LongMem）都必须继承此基类，
并实现其抽象方法，以保证整个项目的数据契约（Data Contract）一致。

统一输出格式（UnifiedSample）是整个项目的核心数据结构。
所有 baselines 和我们自己的方法都以此为输入。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DialogTurn:
    """
    表示一次对话中的单个发言轮次。

    Attributes:
        role: 发言者角色，'user' 或 'assistant'。
        content: 发言内容。
        timestamp: 可选的时间戳字符串（格式：YYYY/MM/DD HH:MM）。
        has_answer: 可选标记，用于 LongMemEval 评测，标识该轮次是否包含答案证据。
    """
    role: str
    content: str
    timestamp: Optional[str] = None
    has_answer: Optional[bool] = None


@dataclass
class UnifiedSample:
    """
    整个项目的核心数据结构：统一样本格式。

    这是所有 benchmark 加载器的输出格式，也是所有 baseline 和
    我们自己的方法（M1-M5 pipeline）的输入格式。

    Attributes:
        sample_id: 样本的唯一标识符。
        source: 数据来源，例如 'longmemeval', 'meddialog', 'med_longmem'。
        question: 需要回答的问题。
        answer: 标准答案（ground truth）。
        dialog_history: 对话历史，由 DialogTurn 列表组成。
        question_date: 提问的日期（可选，主要用于 LongMemEval 时序推理）。
        question_type: 问题类型（可选，例如 'knowledge-update', 'temporal-reasoning'）。
        metadata: 存储其他任务特定元数据的字典。
    """
    sample_id: str
    source: str
    question: str
    answer: str
    dialog_history: list[DialogTurn]
    question_date: Optional[str] = None
    question_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseTask(ABC):
    """
    所有 benchmark task 的抽象基类。

    每个具体的 benchmark（如 LongMemEvalTask）都必须继承此类，
    并实现 load() 方法。
    """

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        Args:
            data_path: 原始数据文件或目录的路径。
            max_samples: 加载的最大样本数量（用于快速调试）。
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self._samples: Optional[list[UnifiedSample]] = None

    @abstractmethod
    def load(self) -> list[UnifiedSample]:
        """
        从 data_path 加载原始数据，并将其转换为 UnifiedSample 列表。

        Returns:
            一个 UnifiedSample 对象的列表。
        """
        raise NotImplementedError

    def get_samples(self) -> list[UnifiedSample]:
        """
        获取已加载的样本列表。如果尚未加载，则先调用 load()。

        Returns:
            一个 UnifiedSample 对象的列表。
        """
        if self._samples is None:
            self._samples = self.load()
        return self._samples

    def __len__(self) -> int:
        return len(self.get_samples())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_path='{self.data_path}', n_samples={len(self)})"
