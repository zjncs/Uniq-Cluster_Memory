"""
test_eval_metrics.py
====================
用 5 条人工构造的 toy 数据，验证 Unique-F1 和 Conflict-F1 评测逻辑的正确性。

每个 Test Case 都对应一个真实的医疗场景，并明确说明：
    - 场景描述（模拟什么情况）
    - 系统输出（predicted）
    - 标准答案（ground_truth）
    - 期望的评测结果（expected）

Test Case 设计（对应您给出的 5 种场景）：
    Case 1: 同一天重复血糖（冗余去重，无冲突）
    Case 2: 不同天血糖（时间范围不同，应保留两条）
    Case 3: 用药覆盖（latest 策略，新值覆盖旧值）
    Case 4: 隐性冲突（同一天同一属性出现矛盾值）
    Case 5: 长跨度指代（指代解析后，应合并为同一记录）
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord
from evaluation.uniqueness_eval import (
    compute_unique_f1,
    explain_uniqueness_result,
    aggregate_unique_f1,
)
from evaluation.conflict_eval import (
    compute_conflict_f1,
    explain_conflict_result,
    aggregate_conflict_f1,
)


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def make_mem(
    patient_id: str,
    attribute: str,
    value: str,
    unit: str = "",
    time_scope: str = "global",
    conflict_flag: bool = False,
    conflict_history: list = None,
    update_policy: str = "unique",
    provenance: list = None,
) -> CanonicalMemory:
    """快速构造 CanonicalMemory 对象的辅助函数。"""
    return CanonicalMemory(
        patient_id=patient_id,
        attribute=attribute,
        value=value,
        unit=unit,
        time_scope=time_scope,
        conflict_flag=conflict_flag,
        conflict_history=conflict_history or [],
        update_policy=update_policy,
        provenance=provenance or [],
    )


def run_test(
    case_name: str,
    predicted: list[CanonicalMemory],
    ground_truth: list[CanonicalMemory],
    expected_unique_f1: float,
    expected_conflict_f1: float,
    tolerance: float = 0.001,
) -> bool:
    """
    运行单个 Test Case，打印诊断报告，并验证期望值。

    Returns:
        True 表示通过，False 表示失败。
    """
    print(f"\n{'='*65}")
    print(f"  {case_name}")
    print(f"{'='*65}")

    # 计算 Unique-F1
    u_metrics = compute_unique_f1(predicted, ground_truth)
    u_report = explain_uniqueness_result(predicted, ground_truth, u_metrics)
    print(u_report)

    # 计算 Conflict-F1
    c_metrics = compute_conflict_f1(predicted, ground_truth)
    c_report = explain_conflict_result(predicted, ground_truth, c_metrics)
    print(c_report)

    # 验证期望值
    u_pass = abs(u_metrics.f1 - expected_unique_f1) < tolerance
    c_pass = abs(c_metrics.f1 - expected_conflict_f1) < tolerance

    u_status = "✅ PASS" if u_pass else f"❌ FAIL (expected {expected_unique_f1}, got {u_metrics.f1})"
    c_status = "✅ PASS" if c_pass else f"❌ FAIL (expected {expected_conflict_f1}, got {c_metrics.f1})"

    print(f"  Unique-F1   : {u_metrics.f1:.4f}  {u_status}")
    print(f"  Conflict-F1 : {c_metrics.f1:.4f}  {c_status}")

    return u_pass and c_pass


# ─── Test Case 1: 同一天重复血糖（冗余去重，无冲突） ─────────────────────────
#
# 场景：患者 P001 在 2024-01-15 这一天，对话中两次提到血糖值 10.5 mmol/L。
#       系统应该只保留一条记录（去重），且不应标记为冲突。
#
# GT：1 条记录（唯一化后）
# Predicted（理想系统）：1 条记录（正确去重）
# Predicted（差系统）：2 条重复记录（冗余）

def case_1_duplicate_glucose() -> bool:
    """Case 1: 同一天重复血糖 - 系统正确去重"""
    gt = [
        make_mem("P001", "blood_glucose", "10.5", unit="mmol/L",
                 time_scope="2024-01-15", conflict_flag=False, update_policy="unique"),
    ]
    # 理想系统：正确去重，只输出 1 条
    predicted_good = [
        make_mem("P001", "blood_glucose", "10.5", unit="mmol/L",
                 time_scope="2024-01-15", conflict_flag=False, update_policy="unique"),
    ]
    # 差系统：未去重，输出 2 条相同记录
    predicted_bad = [
        make_mem("P001", "blood_glucose", "10.5", unit="mmol/L",
                 time_scope="2024-01-15", conflict_flag=False, update_policy="unique"),
        make_mem("P001", "blood_glucose", "10.5", unit="mmol/L",
                 time_scope="2024-01-15", conflict_flag=False, update_policy="unique"),
    ]

    print("\n  [Sub-case 1a] 理想系统（正确去重）")
    r1 = run_test(
        "Case 1a: 同一天重复血糖 - 理想系统",
        predicted_good, gt,
        expected_unique_f1=1.0,
        expected_conflict_f1=0.0,
    )

    print("\n  [Sub-case 1b] 差系统（未去重，冗余输出）")
    u_bad = compute_unique_f1(predicted_bad, gt)
    print(f"  冗余率 (Redundancy): {u_bad.redundancy:.4f}  (期望 > 0)")
    print(f"  Unique-F1: {u_bad.f1:.4f}  (期望 < 1.0，因为 FP=1)")
    r2 = u_bad.redundancy > 0 and u_bad.f1 < 1.0
    status = "✅ PASS" if r2 else "❌ FAIL"
    print(f"  冗余惩罚验证: {status}")

    return r1 and r2


def test_case_1_duplicate_glucose():
    assert case_1_duplicate_glucose()


# ─── Test Case 2: 不同天血糖（时间范围不同，应保留两条） ─────────────────────
#
# 场景：患者 P001 在 2024-01-15 血糖 10.5，在 2024-01-20 血糖 8.3。
#       这是两个不同时间点的合法记录，系统应该保留两条，不应标记为冲突。

def case_2_different_day_glucose() -> bool:
    """Case 2: 不同天血糖 - 应保留两条独立记录"""
    gt = [
        make_mem("P001", "blood_glucose", "10.5", unit="mmol/L",
                 time_scope="2024-01-15", conflict_flag=False, update_policy="unique"),
        make_mem("P001", "blood_glucose", "8.3", unit="mmol/L",
                 time_scope="2024-01-20", conflict_flag=False, update_policy="unique"),
    ]
    predicted = [
        make_mem("P001", "blood_glucose", "10.5", unit="mmol/L",
                 time_scope="2024-01-15", conflict_flag=False, update_policy="unique"),
        make_mem("P001", "blood_glucose", "8.3", unit="mmol/L",
                 time_scope="2024-01-20", conflict_flag=False, update_policy="unique"),
    ]
    return run_test(
        "Case 2: 不同天血糖 - 保留两条独立记录",
        predicted, gt,
        expected_unique_f1=1.0,
        expected_conflict_f1=0.0,
    )


def test_case_2_different_day_glucose():
    assert case_2_different_day_glucose()


# ─── Test Case 3: 用药覆盖（latest 策略，新值覆盖旧值） ──────────────────────
#
# 场景：患者 P002 的用药方案从"阿莫西林 0.5g tid"更新为"头孢克洛 0.25g bid"。
#       update_policy = "latest"，全局只保留最新值。
#       GT 只有一条记录（最新的头孢克洛），旧值被覆盖但不标记为冲突。

def case_3_medication_update() -> bool:
    """Case 3: 用药覆盖 - latest 策略，只保留最新值"""
    gt = [
        make_mem("P002", "medication", "头孢克洛 0.25g bid",
                 time_scope="global", conflict_flag=False, update_policy="latest"),
    ]
    # 理想系统：正确覆盖，只输出最新值
    predicted_good = [
        make_mem("P002", "medication", "头孢克洛 0.25g bid",
                 time_scope="global", conflict_flag=False, update_policy="latest"),
    ]
    # 差系统：同时保留了旧值（未覆盖）
    predicted_bad = [
        make_mem("P002", "medication", "阿莫西林 0.5g tid",
                 time_scope="global", conflict_flag=False, update_policy="latest"),
        make_mem("P002", "medication", "头孢克洛 0.25g bid",
                 time_scope="global", conflict_flag=False, update_policy="latest"),
    ]

    print("\n  [Sub-case 3a] 理想系统（正确覆盖）")
    r1 = run_test(
        "Case 3a: 用药覆盖 - 理想系统",
        predicted_good, gt,
        expected_unique_f1=1.0,
        expected_conflict_f1=0.0,
    )

    print("\n  [Sub-case 3b] 差系统（旧值未被覆盖）")
    u_bad = compute_unique_f1(predicted_bad, gt)
    # 差系统有 2 条记录，但 unique_key 相同，只有 1 条能匹配 GT
    # 另 1 条（旧值）不匹配 GT，计为 FP
    print(f"  Unique-F1: {u_bad.f1:.4f}  (期望 < 1.0，因为旧值是 FP)")
    r2 = u_bad.f1 < 1.0
    status = "✅ PASS" if r2 else "❌ FAIL"
    print(f"  旧值惩罚验证: {status}")

    return r1 and r2


def test_case_3_medication_update():
    assert case_3_medication_update()


# ─── Test Case 4: 隐性冲突（同一天同一属性出现矛盾值） ───────────────────────
#
# 场景：患者 P003 在 2024-02-10 这一天，
#       对话第 3 轮说体温 37.8℃，第 15 轮说体温 38.5℃（同一天，但值不同）。
#       这是一个真实的隐性冲突：同一天体温不应该有两个不同的"权威值"。
#       系统应该：保留最新值 38.5℃，并将 conflict_flag 置为 True。

def case_4_implicit_conflict() -> bool:
    """Case 4: 隐性冲突 - 同一天体温矛盾"""
    conflict_rec = ConflictRecord(
        old_value="37.8",
        new_value="38.5",
        old_provenance=["turn_3"],
        new_provenance=["turn_15"],
        conflict_type="value_change",
        detected_at="2024-02-10T10:00:00",
    )
    gt = [
        make_mem("P003", "body_temperature", "38.5", unit="℃",
                 time_scope="2024-02-10", conflict_flag=True,
                 conflict_history=[conflict_rec], update_policy="unique"),
    ]
    # 理想系统：正确检测冲突，保留最新值并标记
    predicted_good = [
        make_mem("P003", "body_temperature", "38.5", unit="℃",
                 time_scope="2024-02-10", conflict_flag=True,
                 conflict_history=[conflict_rec], update_policy="unique"),
    ]
    # 差系统：未检测到冲突，只保留了旧值且 conflict_flag=False
    predicted_bad = [
        make_mem("P003", "body_temperature", "37.8", unit="℃",
                 time_scope="2024-02-10", conflict_flag=False,
                 update_policy="unique"),
    ]

    print("\n  [Sub-case 4a] 理想系统（正确检测冲突）")
    r1 = run_test(
        "Case 4a: 隐性冲突 - 理想系统",
        predicted_good, gt,
        expected_unique_f1=1.0,
        expected_conflict_f1=1.0,
    )

    print("\n  [Sub-case 4b] 差系统（未检测到冲突）")
    r2 = run_test(
        "Case 4b: 隐性冲突 - 差系统（漏报冲突）",
        predicted_bad, gt,
        expected_unique_f1=0.0,   # value 不匹配（37.8 vs 38.5），所以 Unique-F1=0
        expected_conflict_f1=0.0, # 未检测到冲突，Conflict-F1=0
    )

    return r1 and r2


def test_case_4_implicit_conflict():
    assert case_4_implicit_conflict()


# ─── Test Case 5: 长跨度指代（指代解析后，应合并为同一记录） ─────────────────
#
# 场景：患者 P004 在对话中：
#       第 2 轮："我上次检查血糖是 12.1 mmol/L"（2024-03-01）
#       第 18 轮："那个指标现在降到 9.8 了"（2024-03-15，"那个指标"指代血糖）
#       系统需要正确解析"那个指标"=血糖，并在 2024-03-15 的 time_scope 下
#       更新血糖值为 9.8 mmol/L。
#
# GT：两条记录（两个不同日期的血糖值）
# 理想系统：正确解析指代，输出两条记录
# 差系统：未解析指代，第 18 轮的记录丢失

def case_5_long_span_coreference() -> bool:
    """Case 5: 长跨度指代 - 指代解析后应生成两条独立记录"""
    gt = [
        make_mem("P004", "blood_glucose", "12.1", unit="mmol/L",
                 time_scope="2024-03-01", conflict_flag=False, update_policy="unique"),
        make_mem("P004", "blood_glucose", "9.8", unit="mmol/L",
                 time_scope="2024-03-15", conflict_flag=False, update_policy="unique"),
    ]
    # 理想系统：正确解析指代，两条记录都正确
    predicted_good = [
        make_mem("P004", "blood_glucose", "12.1", unit="mmol/L",
                 time_scope="2024-03-01", conflict_flag=False, update_policy="unique"),
        make_mem("P004", "blood_glucose", "9.8", unit="mmol/L",
                 time_scope="2024-03-15", conflict_flag=False, update_policy="unique"),
    ]
    # 差系统：未解析指代，第 18 轮的记录丢失（FN）
    predicted_bad = [
        make_mem("P004", "blood_glucose", "12.1", unit="mmol/L",
                 time_scope="2024-03-01", conflict_flag=False, update_policy="unique"),
    ]

    print("\n  [Sub-case 5a] 理想系统（正确解析指代）")
    r1 = run_test(
        "Case 5a: 长跨度指代 - 理想系统",
        predicted_good, gt,
        expected_unique_f1=1.0,
        expected_conflict_f1=0.0,
    )

    print("\n  [Sub-case 5b] 差系统（指代解析失败，记录丢失）")
    r2 = run_test(
        "Case 5b: 长跨度指代 - 差系统（FN=1）",
        predicted_bad, gt,
        expected_unique_f1=round(2 * 1.0 * 0.5 / (1.0 + 0.5), 4),  # P=1.0, R=0.5, F1=0.6667
        expected_conflict_f1=0.0,
    )

    return r1 and r2


def test_case_5_long_span_coreference():
    assert case_5_long_span_coreference()


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  Uniq-Cluster Memory: Evaluation Metrics Validation")
    print("  Testing Unique-F1 and Conflict-F1 on 5 Toy Cases")
    print("=" * 65)

    results = {
        "Case 1: 同一天重复血糖": case_1_duplicate_glucose(),
        "Case 2: 不同天血糖":     case_2_different_day_glucose(),
        "Case 3: 用药覆盖":       case_3_medication_update(),
        "Case 4: 隐性冲突":       case_4_implicit_conflict(),
        "Case 5: 长跨度指代":     case_5_long_span_coreference(),
    }

    print("\n" + "=" * 65)
    print("  Final Summary")
    print("=" * 65)
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print("=" * 65)
    if all_pass:
        print("  🎉 All 5 test cases PASSED. Evaluation metrics are validated.")
    else:
        print("  ⚠️  Some test cases FAILED. Please review the diagnostic reports above.")
    print("=" * 65 + "\n")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
