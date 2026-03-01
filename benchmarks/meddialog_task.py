"""
meddialog_task.py
=================
MedDialog（中文医疗对话）数据集的加载器。

数据集来源：ticoAg/Chinese-medical-dialogue (Parquet 格式)
原始数据格式（每条样本）：
    {
        "instruction": str,  # 患者的问题/主诉
        "input": str,        # 患者的详细描述
        "output": str,       # 医生的回复
        "history": null | list  # 多轮历史（部分样本有，大部分为 null）
    }

由于 MedDialog 本身不包含预设的 QA 对，我们将其适配为：
    - question: 患者的 instruction（主诉）
    - answer: 医生的 output（回复）
    - dialog_history: 由 input（患者描述）和 output（医生回复）构成的两轮对话

这个适配方式为后续的事件抽取（M1）提供了标准化的输入。
"""

import json
from pathlib import Path
from typing import Optional

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from benchmarks.base_task import BaseTask, DialogTurn, UnifiedSample


class MedDialogTask(BaseTask):
    """
    MedDialog 中文医疗对话数据集加载器。

    支持从 JSON 文件（样本）或 Parquet 文件（完整数据集）加载。
    """

    def load(self) -> list[UnifiedSample]:
        """
        加载 MedDialog 数据并转换为 UnifiedSample 列表。

        Returns:
            UnifiedSample 列表。
        """
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"MedDialog data file not found: {data_path}")

        suffix = data_path.suffix.lower()
        if suffix == ".json":
            raw_data = self._load_json(data_path)
        elif suffix == ".parquet":
            raw_data = self._load_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Expected .json or .parquet")

        if self.max_samples is not None:
            raw_data = raw_data[: self.max_samples]

        samples = []
        for idx, item in enumerate(raw_data):
            sample = self._convert_to_unified(item, idx)
            if sample is not None:
                samples.append(sample)

        return samples

    @staticmethod
    def _load_json(data_path: Path) -> list[dict]:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_parquet(data_path: Path) -> list[dict]:
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required to load Parquet files. Run: pip install pyarrow")
        table = pq.read_table(str(data_path))
        return table.to_pandas().to_dict(orient="records")

    @staticmethod
    def _convert_to_unified(item: dict, idx: int) -> Optional[UnifiedSample]:
        """
        将单条 MedDialog 原始记录转换为 UnifiedSample。

        Args:
            item: 原始数据字典。
            idx: 样本索引（用于生成 sample_id）。

        Returns:
            UnifiedSample 对象，或 None（如果数据不完整则跳过）。
        """
        instruction = (item.get("instruction") or "").strip()
        patient_input = (item.get("input") or "").strip()
        doctor_output = (item.get("output") or "").strip()

        # 跳过关键字段为空的样本
        if not instruction or not doctor_output:
            return None

        # 构建对话历史：患者描述 -> 医生回复
        # 如果 input 不为空，将其作为患者的第一轮发言
        dialog_history = []
        if patient_input:
            dialog_history.append(DialogTurn(role="user", content=patient_input))
        dialog_history.append(DialogTurn(role="assistant", content=doctor_output))

        # 处理多轮历史（如果存在）
        history = item.get("history")
        if history and isinstance(history, list):
            historical_turns = []
            for h_turn in history:
                if isinstance(h_turn, list) and len(h_turn) == 2:
                    historical_turns.append(DialogTurn(role="user", content=h_turn[0]))
                    historical_turns.append(DialogTurn(role="assistant", content=h_turn[1]))
            # 将历史轮次前置到对话历史中
            dialog_history = historical_turns + dialog_history

        return UnifiedSample(
            sample_id=f"meddialog_{idx:06d}",
            source="meddialog",
            question=instruction,
            answer=doctor_output,
            dialog_history=dialog_history,
            question_date=None,
            question_type="medical_qa",
            metadata={},
        )


if __name__ == "__main__":
    import os
    # 使用 JSON 样本文件进行快速验证
    data_file = os.path.join(
        os.path.dirname(__file__),
        "../data/raw/meddialog/meddialog_zh_sample50.json",
    )
    task = MedDialogTask(data_path=data_file, max_samples=5)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples from MedDialog")
    s = samples[0]
    print(f"\nSample ID   : {s.sample_id}")
    print(f"Question    : {s.question}")
    print(f"Answer      : {s.answer[:100]}...")
    print(f"History turns: {len(s.dialog_history)}")
    for turn in s.dialog_history:
        print(f"  [{turn.role}] {turn.content[:60]}...")
