"""
m4_compression/compressor.py
=============================
M4 模块：记忆压缩器。

核心职责：
    对 M3 输出的 CanonicalMemory 列表进行跨时间范围的压缩和摘要，
    减少记忆库的存储开销，同时保留关键信息。

压缩策略：
    1. 时间序列压缩（unique/latest 策略）：
       对同一属性的多个时间点记录，如果时间跨度超过阈值，
       保留最新值 + 历史摘要（如"血糖从 7.2 降至 6.8，趋势改善"）。
    2. 症状去重（append 策略）：
       合并重复或高度相似的症状描述。
    3. 冲突保留：
       有 conflict_flag=True 的记录不压缩，完整保留用于 Conflict-F1 评测。

注意：
    当前实现为轻量版本，主要完成接口定义和基本压缩逻辑。
    论文中的完整压缩方法（LLM-based summarization）在此基础上扩展。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory


class MemoryCompressor:
    """
    M4 记忆压缩器（轻量版）。

    当前策略：
        - 对无冲突的记录，按属性分组后保留最新值。
        - 对有冲突的记录，完整保留（不压缩）。
        - 对 append 策略的记录（如症状），合并为列表形式。
    """

    MAX_HISTORY_PER_ATTRIBUTE = 5  # 每个属性最多保留的历史记录数

    def compress(
        self,
        memories: List[CanonicalMemory],
        patient_id: str,
    ) -> List[CanonicalMemory]:
        """
        对 CanonicalMemory 列表进行压缩。

        Args:
            memories:   M3 输出的 CanonicalMemory 列表。
            patient_id: 患者/对话 ID。

        Returns:
            压缩后的 CanonicalMemory 列表。
        """
        if not memories:
            return []

        # 按属性分组
        attr_groups: Dict[str, List[CanonicalMemory]] = {}
        for mem in memories:
            key = mem.attribute
            if key not in attr_groups:
                attr_groups[key] = []
            attr_groups[key].append(mem)

        compressed: List[CanonicalMemory] = []
        for attr, group in attr_groups.items():
            compressed.extend(self._compress_group(group))

        return compressed

    def _compress_group(
        self,
        group: List[CanonicalMemory],
    ) -> List[CanonicalMemory]:
        """对单个属性的记录组进行压缩。"""
        if not group:
            return []

        policy = group[0].update_policy

        # 有冲突的记录完整保留
        conflict_records = [m for m in group if m.conflict_flag]
        clean_records = [m for m in group if not m.conflict_flag]

        if policy == "append":
            # append 策略：保留所有不重复的值
            seen_values = set()
            result = []
            for mem in group:
                key = mem.value.strip().lower()
                if key not in seen_values:
                    seen_values.add(key)
                    result.append(mem)
            return result

        elif policy == "latest":
            # latest 策略：全局只保留一条最新记录（不按 scope 拆分）
            result: List[CanonicalMemory] = []
            if clean_records:
                best = max(
                    clean_records,
                    key=lambda m: max(m.provenance) if m.provenance else 0,
                )
                result.append(best)
            # 冲突记录用于评测可追溯性，仍保留
            result.extend(conflict_records)
            return result

        elif policy == "unique":
            # 按 time_scope 分组，每个 scope 保留最新/最高置信度的记录
            scope_groups: Dict[str, List[CanonicalMemory]] = {}
            for mem in clean_records:
                scope = mem.time_scope
                if scope not in scope_groups:
                    scope_groups[scope] = []
                scope_groups[scope].append(mem)

            result = []
            for scope, scope_mems in scope_groups.items():
                # unique 保留置信度最高的
                best = max(scope_mems, key=lambda m: m.confidence)
                result.append(best)

            # 如果历史记录过多，只保留最近的 MAX_HISTORY_PER_ATTRIBUTE 条
            if len(result) > self.MAX_HISTORY_PER_ATTRIBUTE:
                # 按 time_scope 排序，保留最新的
                result.sort(key=lambda m: m.time_scope, reverse=True)
                result = result[:self.MAX_HISTORY_PER_ATTRIBUTE]

            # 加回冲突记录
            result.extend(conflict_records)
            return result

        return group
