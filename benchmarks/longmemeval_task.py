"""
longmemeval_task.py
===================
LongMemEval 数据集的加载器。

数据集来源：https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
论文：Benchmarking Chat Assistants on Long-Term Interactive Memory (ICLR 2025)

原始数据格式（每条样本）：
    {
        "question_id": str,
        "question_type": str,
        "question": str,
        "answer": str,
        "question_date": str,
        "haystack_session_ids": list[str],
        "haystack_dates": list[str],
        "haystack_sessions": list[list[dict]],  # list of sessions, each session is a list of turns
        "answer_session_ids": list[str]
    }

输出格式：UnifiedSample（见 base_task.py）
"""

import json
from pathlib import Path
from typing import Optional

from benchmarks.base_task import BaseTask, DialogTurn, UnifiedSample


class LongMemEvalTask(BaseTask):
    """
    LongMemEval 数据集加载器。

    将 LongMemEval 的 JSON 格式转换为统一的 UnifiedSample 格式。
    对话历史由多个 session 组成，每个 session 包含多个 turn。
    我们将所有 session 的 turn 按时间顺序展平为一个 dialog_history 列表，
    并在每个 turn 上附加对应的 session 时间戳。
    """

    def load(self) -> list[UnifiedSample]:
        """
        加载 LongMemEval JSON 文件并转换为 UnifiedSample 列表。

        Returns:
            UnifiedSample 列表。
        """
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"LongMemEval data file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if self.max_samples is not None:
            raw_data = raw_data[: self.max_samples]

        samples = []
        for item in raw_data:
            # 将多个 session 的 turns 展平为一个对话历史列表
            dialog_history = self._flatten_sessions(
                sessions=item["haystack_sessions"],
                dates=item["haystack_dates"],
            )

            sample = UnifiedSample(
                sample_id=item["question_id"],
                source="longmemeval",
                question=item["question"],
                answer=item["answer"],
                dialog_history=dialog_history,
                question_date=item.get("question_date"),
                question_type=item.get("question_type"),
                metadata={
                    "haystack_session_ids": item.get("haystack_session_ids", []),
                    "answer_session_ids": item.get("answer_session_ids", []),
                },
            )
            samples.append(sample)

        return samples

    @staticmethod
    def _flatten_sessions(
        sessions: list[list[dict]],
        dates: list[str],
    ) -> list[DialogTurn]:
        """
        将多个 session 的 turns 展平为一个 DialogTurn 列表。

        Args:
            sessions: session 列表，每个 session 是一个 turn 字典列表。
            dates: 与 sessions 对应的时间戳列表。

        Returns:
            DialogTurn 列表。
        """
        turns = []
        for session, date in zip(sessions, dates):
            for turn_dict in session:
                turn = DialogTurn(
                    role=turn_dict["role"],
                    content=turn_dict["content"],
                    timestamp=date,
                    has_answer=turn_dict.get("has_answer", False),
                )
                turns.append(turn)
        return turns


if __name__ == "__main__":
    # 快速验证：加载前 5 条样本并打印
    import os
    data_file = os.path.join(
        os.path.dirname(__file__),
        "../data/raw/longmemeval/longmemeval_oracle.json",
    )
    task = LongMemEvalTask(data_path=data_file, max_samples=5)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples from LongMemEval")
    s = samples[0]
    print(f"\nSample ID   : {s.sample_id}")
    print(f"Type        : {s.question_type}")
    print(f"Question    : {s.question}")
    print(f"Answer      : {s.answer}")
    print(f"Date        : {s.question_date}")
    print(f"History turns: {len(s.dialog_history)}")
    print(f"First turn  : [{s.dialog_history[0].role}] {s.dialog_history[0].content[:80]}...")
