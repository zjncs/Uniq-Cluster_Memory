"""
eval_med_longmem_v01.py
=======================
Med-LongMem v0.1 评测脚本。

在 20 条 Hard 级样本上，对比以下三种系统的 Unique-F1 和 Conflict-F1：
    1. GT Upper Bound（完美系统，直接使用 GT 作为预测）
    2. Raw-RAG Baseline（使用 LangChain BM25 检索，LLM 抽取 CanonicalMemory）
    3. No-Memory Baseline（不使用记忆，直接从对话中抽取，无去重）

评测逻辑：
    - GT Upper Bound: predicted = canonical_gt，期望 Unique-F1 = 1.0, Conflict-F1 = 1.0
    - Raw-RAG: 将对话文本分块，BM25 检索相关片段，LLM 从检索结果中抽取 CanonicalMemory
    - No-Memory: LLM 直接从完整对话中抽取，不做任何去重或冲突检测
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord
from evaluation.uniqueness_eval import compute_unique_f1, aggregate_unique_f1
from evaluation.conflict_eval import compute_conflict_f1, aggregate_conflict_f1
from src.uniq_cluster_memory.utils.llm_client import (
    get_llm_client,
    LLM_MODEL,
)

client = get_llm_client()

DATA_DIR = Path("data/raw/med_longmem")
RESULTS_DIR = Path("results/main_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 数据加载 ────────────────────────────────────────────────────────────────

def load_sample(dialogue_id: str) -> dict:
    """加载单条样本的所有文件。"""
    sample_dir = DATA_DIR / dialogue_id
    dialogue = []
    with open(sample_dir / "dialogue.jsonl") as f:
        for line in f:
            dialogue.append(json.loads(line))
    canonical_gt = []
    with open(sample_dir / "canonical_gt.jsonl") as f:
        for line in f:
            canonical_gt.append(CanonicalMemory.from_dict(json.loads(line)))
    conflict_gt = []
    with open(sample_dir / "conflict_gt.jsonl") as f:
        for line in f:
            conflict_gt.append(CanonicalMemory.from_dict(json.loads(line)))
    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)
    return {
        "dialogue_id": dialogue_id,
        "dialogue": dialogue,
        "canonical_gt": canonical_gt,
        "conflict_gt": conflict_gt,
        "metadata": metadata,
    }


def dialogue_to_text(dialogue: list[dict]) -> str:
    """将对话轮次列表转换为纯文本。"""
    lines = []
    for turn in dialogue:
        speaker = "Doctor" if turn["speaker"] == "doctor" else "Patient"
        lines.append(f"{speaker} (Turn {turn['turn_id']}): {turn['text']}")
    return "\n".join(lines)


# ─── LLM 抽取逻辑 ────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a medical information extraction system. 
Given a patient-doctor dialogue, extract all unique medical facts as structured records.

For each unique medical attribute mentioned, create ONE canonical record with:
- patient_id: use the dialogue_id provided
- attribute: standardized attribute name (e.g., blood_glucose, blood_pressure_sys, medication, symptom, primary_diagnosis)
- value: the FINAL/MOST RECENT value mentioned for this attribute in this time scope
- unit: measurement unit if applicable (e.g., mmol/L, mmHg, bpm, °C, g/L), empty string if none
- time_scope: the date or time period this value refers to (use ISO date format if mentioned, otherwise "global")
- conflict_flag: true if you detect that the same attribute had DIFFERENT values mentioned for the same time scope
- update_policy: "unique" for measurements/diagnoses, "latest" for medications, "append" for symptoms
- conflict_history: list of {old_value, new_value, conflict_type} if conflict_flag is true, else empty list

CRITICAL RULES:
1. For "unique" policy: if the same attribute appears multiple times for the same date with DIFFERENT values, set conflict_flag=true and keep only the LAST value
2. For "latest" policy (medications): keep only the most recently prescribed medication
3. For "append" policy (symptoms): create separate records for each distinct symptom
4. Resolve coreferences: if "that reading" or "that number" refers to a previous measurement, treat it as the same attribute

Return ONLY a JSON array of canonical memory objects. No explanation."""


def extract_canonical_memories(
    dialogue_text: str,
    dialogue_id: str,
    context_text: str = None,
) -> list[CanonicalMemory]:
    """
    用 LLM 从对话文本（或检索片段）中抽取 CanonicalMemory 列表。

    Args:
        dialogue_text: 用于抽取的文本（完整对话或检索片段）。
        dialogue_id:   对话 ID。
        context_text:  可选的额外上下文（用于 RAG 模式）。
    """
    input_text = dialogue_text
    if context_text:
        input_text = f"RETRIEVED CONTEXT:\n{context_text}\n\nFULL DIALOGUE:\n{dialogue_text}"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Dialogue ID: {dialogue_id}\n\n{input_text}"},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        records = json.loads(raw)
        memories = []
        for r in records:
            conflict_history = []
            for ch in r.get("conflict_history", []):
                conflict_history.append(ConflictRecord(
                    old_value=str(ch.get("old_value", "")),
                    new_value=str(ch.get("new_value", "")),
                    old_provenance=[],
                    new_provenance=[],
                    conflict_type=ch.get("conflict_type", "value_change"),
                    detected_at=r.get("time_scope", ""),
                ))
            memories.append(CanonicalMemory(
                patient_id=str(r.get("patient_id", dialogue_id)),
                attribute=str(r.get("attribute", "")),
                value=str(r.get("value", "")),
                unit=str(r.get("unit", "")),
                time_scope=str(r.get("time_scope", "global")),
                confidence=float(r.get("confidence", 1.0)),
                provenance=[],
                conflict_flag=bool(r.get("conflict_flag", False)),
                conflict_history=conflict_history,
                update_policy=str(r.get("update_policy", "unique")),
            ))
        return memories
    except Exception as e:
        print(f"    [WARN] Extraction failed: {e}")
        return []


# ─── 三种系统的预测逻辑 ──────────────────────────────────────────────────────

def predict_gt_upper_bound(sample: dict) -> list[CanonicalMemory]:
    """GT Upper Bound: 直接返回 canonical_gt（完美预测）。"""
    return sample["canonical_gt"]


def predict_no_memory(sample: dict) -> list[CanonicalMemory]:
    """
    No-Memory Baseline: LLM 直接从完整对话中抽取，无去重逻辑。
    """
    dialogue_text = dialogue_to_text(sample["dialogue"])
    return extract_canonical_memories(dialogue_text, sample["dialogue_id"])


def predict_raw_rag(sample: dict) -> list[CanonicalMemory]:
    """
    Raw-RAG Baseline: BM25 检索相关片段，LLM 从检索结果中抽取。
    使用简单的关键词匹配模拟 BM25（无需向量数据库）。
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    # 将对话按 5 轮一组分块
    dialogue = sample["dialogue"]
    chunks = []
    for i in range(0, len(dialogue), 5):
        chunk_turns = dialogue[i:i+5]
        chunk_text = "\n".join(
            f"{'Doctor' if t['speaker']=='doctor' else 'Patient'} (Turn {t['turn_id']}): {t['text']}"
            for t in chunk_turns
        )
        chunks.append(Document(page_content=chunk_text, metadata={"chunk_id": i // 5}))

    # BM25 检索：用医疗关键词查询
    medical_query = (
        "blood glucose pressure temperature hemoglobin medication diagnosis symptom "
        "reading result value changed updated"
    )
    try:
        retriever = BM25Retriever.from_documents(chunks, k=min(4, len(chunks)))
        retrieved_docs = retriever.invoke(medical_query)
        context_text = "\n\n---\n\n".join(d.page_content for d in retrieved_docs)
    except Exception:
        context_text = None

    full_dialogue_text = dialogue_to_text(sample["dialogue"])
    return extract_canonical_memories(full_dialogue_text, sample["dialogue_id"], context_text)


# ─── 主评测循环 ──────────────────────────────────────────────────────────────

def evaluate_system(
    system_name: str,
    predict_fn,
    samples: list[dict],
) -> dict:
    """
    在所有样本上运行一个系统，计算聚合指标。
    """
    u_metrics_list = []
    c_metrics_list = []
    per_sample_results = []

    for sample in samples:
        did = sample["dialogue_id"]
        print(f"  [{system_name}] {did}...", end=" ", flush=True)
        try:
            predicted = predict_fn(sample)
            gt = sample["canonical_gt"]
            u = compute_unique_f1(predicted, gt)
            c = compute_conflict_f1(predicted, gt)
            u_metrics_list.append(u)
            c_metrics_list.append(c)
            per_sample_results.append({
                "dialogue_id": did,
                "unique_f1_strict": u.f1,
                "unique_f1_relaxed": u.relaxed_f1,
                "attribute_coverage": u.attribute_coverage,
                "unique_precision": u.precision,
                "unique_recall": u.recall,
                "redundancy": u.redundancy,
                "coverage": u.coverage,
                "conflict_f1": c.f1,
                "conflict_precision": c.precision,
                "conflict_recall": c.recall,
                "n_predicted": u.n_predicted,
                "n_gt": u.n_gt,
            })
            print(f"U-F1(S)={u.f1:.3f} U-F1(R)={u.relaxed_f1:.3f} AttrCov={u.attribute_coverage:.3f} C-F1={c.f1:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    u_agg = aggregate_unique_f1(u_metrics_list)
    c_agg = aggregate_conflict_f1(c_metrics_list)

    return {
        "system": system_name,
        "unique_f1": u_agg.mean_f1,
        "unique_relaxed_f1": u_agg.mean_relaxed_f1,
        "mean_attribute_coverage": u_agg.mean_attribute_coverage,
        "unique_precision": u_agg.mean_precision,
        "unique_recall": u_agg.mean_recall,
        "mean_redundancy": u_agg.mean_redundancy,
        "mean_coverage": u_agg.mean_coverage,
        "conflict_f1": c_agg.mean_f1,
        "conflict_precision": c_agg.mean_precision,
        "conflict_recall": c_agg.mean_recall,
        "n_samples": u_agg.n_samples,
        "per_sample": per_sample_results,
    }


def main():
    # 加载所有样本
    summary_path = DATA_DIR / "dataset_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    dialogue_ids = [s["dialogue_id"] for s in summary["samples"]]

    print(f"\n{'='*65}")
    print(f"  Med-LongMem v0.1 Evaluation")
    print(f"  {len(dialogue_ids)} Hard-level samples")
    print(f"{'='*65}\n")

    samples = []
    for did in dialogue_ids:
        samples.append(load_sample(did))

    all_results = []

    # 1. GT Upper Bound
    print("\n[1/3] GT Upper Bound (oracle)")
    gt_result = evaluate_system("GT_Upper_Bound", predict_gt_upper_bound, samples)
    all_results.append(gt_result)

    # 2. No-Memory Baseline
    print("\n[2/3] No-Memory Baseline")
    nm_result = evaluate_system("No_Memory", predict_no_memory, samples)
    all_results.append(nm_result)

    # 3. Raw-RAG Baseline
    print("\n[3/3] Raw-RAG Baseline (BM25 + LLM)")
    rag_result = evaluate_system("Raw_RAG", predict_raw_rag, samples)
    all_results.append(rag_result)

    # 打印汇总表
    print(f"\n{'='*80}")
    print(f"  Final Results on Med-LongMem v0.1 (n={len(samples)}, Hard)")
    print(f"{'='*80}")
    print(f"  [Strict]  attribute + time_scope + value must all match")
    print(f"  [Relaxed] attribute + value match (time_scope ignored)")
    print(f"  [AttrCov] attribute-only coverage")
    print(f"{'='*80}")
    header = f"{'System':<20} {'U-F1(S)':>9} {'U-F1(R)':>9} {'AttrCov':>9} {'Redund':>8} {'C-F1':>8}"
    print(header)
    print("-" * 80)
    for r in all_results:
        row = (
            f"{r['system']:<20} "
            f"{r['unique_f1']:>9.4f} "
            f"{r.get('unique_relaxed_f1', 0.0):>9.4f} "
            f"{r.get('mean_attribute_coverage', 0.0):>9.4f} "
            f"{r['mean_redundancy']:>8.4f} "
            f"{r['conflict_f1']:>8.4f}"
        )
        print(row)
    print("=" * 80)

    # 保存结果
    output_path = RESULTS_DIR / "med_longmem_v01_eval.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
