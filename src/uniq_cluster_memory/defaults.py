"""
defaults.py
===========
集中管理数据集相关的 pipeline 推荐默认值，避免实验脚本之间出现配置漂移。
"""

from __future__ import annotations

from typing import Optional


def recommended_pipeline_options(dataset_name: Optional[str]) -> dict:
    """
    返回指定数据集的推荐 pipeline 选项。

    目前最重要的是统一：
    - 缺失时间表达的处理策略
    - 每个时间范围内保留的症状条数上限
    """
    normalized = (dataset_name or "").strip().lower()
    if normalized == "med_longmem":
        return {
            "missing_time_scope": "reference_day",
            "max_symptoms_per_scope": 1,
        }
    return {
        "missing_time_scope": "global",
        "max_symptoms_per_scope": None,
    }
