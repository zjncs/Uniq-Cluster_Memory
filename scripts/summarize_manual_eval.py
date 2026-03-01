"""
summarize_manual_eval.py
========================
汇总人工评测标注结果（CSV）。

输入：
    scripts/run_manual_eval_meddialog.py 生成并由人工填写后的 annotation_template.csv

输出：
    - 终端打印关键统计
    - 保存 summary_manual_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


SCORE_FIELDS = [
    "score_accuracy",
    "score_completeness",
    "score_noise",
    "score_clinical_safety",
]


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_pass(v: str) -> bool | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "pass"}:
        return True
    if s in {"0", "false", "no", "n", "fail"}:
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize manual annotation CSV.")
    parser.add_argument(
        "--annotation_csv",
        required=True,
        help="Path to filled annotation_template.csv",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Output summary json path. Default: <annotation_dir>/summary_manual_eval.json",
    )
    args = parser.parse_args()

    csv_path = Path(args.annotation_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    labeled_rows = []
    for r in rows:
        parsed = {k: _to_float(r.get(k, "")) for k in SCORE_FIELDS}
        if any(v is not None for v in parsed.values()):
            parsed["pass_flag"] = _to_pass(r.get("pass_flag", ""))
            parsed["sample_id"] = r.get("sample_id", "")
            labeled_rows.append(parsed)

    if not labeled_rows:
        raise RuntimeError("No labeled rows found. Please fill score columns first.")

    metrics: Dict[str, float] = {}
    for k in SCORE_FIELDS:
        vals = [r[k] for r in labeled_rows if r[k] is not None]
        metrics[f"mean_{k}"] = round(mean(vals), 4) if vals else 0.0

    pass_vals = [r["pass_flag"] for r in labeled_rows if r["pass_flag"] is not None]
    pass_rate = (sum(1 for p in pass_vals if p) / len(pass_vals)) if pass_vals else 0.0

    # 归一化鲁棒性指数（0-100）
    # 以四个 1-5 分维度均值计算，映射到百分制。
    dim_means = [metrics[f"mean_{k}"] for k in SCORE_FIELDS]
    robustness_index = ((mean(dim_means) - 1.0) / 4.0) * 100.0
    robustness_index = max(0.0, min(100.0, robustness_index))

    summary = {
        "n_total_rows": len(rows),
        "n_labeled_rows": len(labeled_rows),
        "coverage_labeled": round(len(labeled_rows) / len(rows), 4) if rows else 0.0,
        **metrics,
        "pass_rate": round(pass_rate, 4),
        "robustness_index_0_100": round(robustness_index, 2),
    }

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = csv_path.parent / "summary_manual_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 56)
    print("Manual Eval Summary")
    print("=" * 56)
    print(f"Rows (total/labeled): {summary['n_total_rows']} / {summary['n_labeled_rows']}")
    print(f"Coverage labeled     : {summary['coverage_labeled']:.2%}")
    print(f"Mean accuracy        : {summary['mean_score_accuracy']:.3f}")
    print(f"Mean completeness    : {summary['mean_score_completeness']:.3f}")
    print(f"Mean noise           : {summary['mean_score_noise']:.3f}")
    print(f"Mean clinical safety : {summary['mean_score_clinical_safety']:.3f}")
    print(f"Pass rate            : {summary['pass_rate']:.2%}")
    print(f"Robustness Index     : {summary['robustness_index_0_100']:.2f}/100")
    print("=" * 56)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

