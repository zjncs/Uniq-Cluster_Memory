"""
run_core_comparison.py
======================
汇总本项目答辩所需的核心对比结果：

1. Med-LongMem: With Memory (UCM) vs No-Memory 的 Unique-F1
2. LongMemEval: With Memory (Hybrid-RAG) vs No-Memory 的 Recall@5

默认优先读取当前仓库中的结果文件，不重复触发昂贵的模型调用。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "main_results"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_round(value: float) -> float:
    return round(float(value), 4)


def aggregate_pipeline_jsonl(path: Path) -> dict:
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    n = len(rows)
    return {
        "system": "UCM",
        "n_samples": n,
        "unique_f1": _safe_round(sum(r.get("unique_f1_strict", 0.0) for r in rows) / n),
        "unique_relaxed_f1": _safe_round(sum(r.get("unique_f1_relaxed", 0.0) for r in rows) / n),
        "mean_attribute_coverage": _safe_round(sum(r.get("attribute_coverage", 0.0) for r in rows) / n),
        "conflict_f1": _safe_round(sum(r.get("conflict_f1", 0.0) for r in rows) / n),
        "avg_latency": _safe_round(sum(r.get("latency", 0.0) for r in rows) / n),
        "source_file": str(path),
    }


def load_system_from_eval(path: Path, system_name: str) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}, got {type(payload).__name__}")

    for item in payload:
        if isinstance(item, dict) and item.get("system") == system_name:
            result = dict(item)
            result["source_file"] = str(path)
            return result
    raise ValueError(f"System {system_name} not found in {path}")


def load_summary(path: Path, system_name: str) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}, got {type(payload).__name__}")

    return {
        "system": system_name,
        "n_samples": payload.get("n_samples", 0),
        "accuracy": _safe_round(payload.get("accuracy", 0.0)),
        "mean_quality_score": _safe_round(payload.get("mean_quality_score", 0.0)),
        "recall_at_5": _safe_round(
            payload.get("recall_at_5", payload.get("mean_recall_at_k", 0.0))
        ),
        "source_file": str(path),
    }


def choose_largest_result_file(search_dir: Path, patterns: Iterable[str]) -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for pattern in patterns:
        for path in search_dir.glob(pattern):
            if not path.is_file():
                continue
            try:
                n_rows = sum(1 for _ in open(path, "r", encoding="utf-8"))
            except OSError:
                continue
            candidates.append((n_rows, path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"No matching result files found under {search_dir} for patterns {list(patterns)}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def build_report(
    medlongmem_with_memory: dict,
    medlongmem_no_memory: dict,
    longmemeval_with_memory: dict,
    longmemeval_no_memory: dict,
) -> dict:
    med_delta = _safe_round(
        medlongmem_with_memory["unique_f1"] - medlongmem_no_memory.get("unique_f1", 0.0)
    )
    recall_delta = _safe_round(
        longmemeval_with_memory["recall_at_5"] - longmemeval_no_memory["recall_at_5"]
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "med_longmem": {
            "metric": "Unique-F1 (strict)",
            "with_memory": medlongmem_with_memory,
            "no_memory": medlongmem_no_memory,
            "delta": {
                "absolute": med_delta,
            },
        },
        "longmemeval": {
            "metric": "Recall@5",
            "with_memory": longmemeval_with_memory,
            "no_memory": longmemeval_no_memory,
            "delta": {
                "absolute": recall_delta,
            },
        },
        "executive_summary": {
            "headline": (
                f"With Memory improves Med-LongMem Unique-F1 by {med_delta:.4f} "
                f"and LongMemEval Recall@5 by {recall_delta:.4f}."
            ),
            "defense_ready_metrics": {
                "f1": medlongmem_with_memory["unique_f1"],
                "f1_baseline": medlongmem_no_memory.get("unique_f1", 0.0),
                "recall_at_5": longmemeval_with_memory["recall_at_5"],
                "recall_at_5_baseline": longmemeval_no_memory["recall_at_5"],
            },
        },
    }


def render_markdown(report: dict) -> str:
    med = report["med_longmem"]
    lme = report["longmemeval"]
    med_with = med["with_memory"]
    med_without = med["no_memory"]
    lme_with = lme["with_memory"]
    lme_without = lme["no_memory"]

    lines = [
        "# Core Comparison Report",
        "",
        report["executive_summary"]["headline"],
        "",
        "## Med-LongMem",
        "",
        f"- Metric: `{med['metric']}`",
        f"- With Memory (`{med_with['system']}`): `{med_with['unique_f1']:.4f}` "
        f"(n={med_with['n_samples']})",
        f"- No Memory (`{med_without['system']}`): `{med_without['unique_f1']:.4f}` "
        f"(n={med_without['n_samples']})",
        f"- Delta: `{med['delta']['absolute']:+.4f}`",
        "",
        "## LongMemEval",
        "",
        f"- Metric: `{lme['metric']}`",
        f"- With Memory (`{lme_with['system']}`): `{lme_with['recall_at_5']:.4f}` "
        f"(n={lme_with['n_samples']})",
        f"- No Memory (`{lme_without['system']}`): `{lme_without['recall_at_5']:.4f}` "
        f"(n={lme_without['n_samples']})",
        f"- Delta: `{lme['delta']['absolute']:+.4f}`",
        "",
        "## Sources",
        "",
        f"- Med-LongMem With Memory: `{med_with['source_file']}`",
        f"- Med-LongMem No Memory: `{med_without['source_file']}`",
        f"- LongMemEval With Memory: `{lme_with['source_file']}`",
        f"- LongMemEval No Memory: `{lme_without['source_file']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the defense-ready core comparison report.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing experiment result files.",
    )
    parser.add_argument(
        "--ucm_jsonl",
        type=Path,
        default=None,
        help="Optional explicit UCM Med-LongMem jsonl result file.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "core_comparison_report.json",
        help="Path to the generated JSON report.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "core_comparison_report.md",
        help="Path to the generated Markdown report.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if args.ucm_jsonl is not None:
        ucm_jsonl = args.ucm_jsonl.resolve()
    else:
        ucm_jsonl = choose_largest_result_file(
            results_dir,
            patterns=("pipeline_med_longmem_w*.jsonl", "pipeline_med_longmem*.jsonl"),
        )

    medlongmem_with_memory = aggregate_pipeline_jsonl(ucm_jsonl)
    medlongmem_no_memory = load_system_from_eval(
        results_dir / "med_longmem_v01_eval.json",
        "No_Memory",
    )

    longmemeval_with_memory = load_summary(
        results_dir / "longmemeval_hybrid_rag_summary.json",
        "Hybrid_RAG",
    )
    longmemeval_no_memory = load_summary(
        results_dir / "longmemeval_no_memory_summary.json",
        "No_Memory",
    )
    longmemeval_no_memory["recall_at_5"] = 0.0

    report = build_report(
        medlongmem_with_memory=medlongmem_with_memory,
        medlongmem_no_memory=medlongmem_no_memory,
        longmemeval_with_memory=longmemeval_with_memory,
        longmemeval_no_memory=longmemeval_no_memory,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print(report["executive_summary"]["headline"])
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
