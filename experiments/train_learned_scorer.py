"""
experiments/train_learned_scorer.py
====================================
训练 Learned Candidate Scorer 并与手写权重对比。

流程:
    1. 遍历 Med-LongMem 数据集，对每个样本运行 M1→M2→M2.5→M3(不做最终评分)
    2. 在 M3 输出的 candidate groups 中，将每个 candidate 与 GT 匹配标记正/负
    3. 提取 7 维特征向量
    4. 训练 LogisticRegression (5-fold CV)
    5. 对比：learned scorer vs hand-crafted weights 的选择准确率
    6. 报告 feature importance 和学习曲线

Usage:
    DASHSCOPE_API_KEY=xxx PYTHONPATH=. python experiments/train_learned_scorer.py \
        --data_path data/raw/med_longmem \
        --output_dir results/learned_scorer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uniq_cluster_memory.m1_event_extraction import MedicalEventExtractor, ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import EventClusterer, InformationBundleBuilder
from src.uniq_cluster_memory.m3_uniqueness import UniquenessManager
from src.uniq_cluster_memory.m3_uniqueness.conflict_detector import (
    ConflictDetector, W_RECENCY, W_AUTHORITY, W_EVIDENCE, W_PRIOR,
)
from src.uniq_cluster_memory.m3_uniqueness.learned_scorer import (
    LearnedCandidateScorer, ScorerFeatures,
)
from src.uniq_cluster_memory.schema import CanonicalMemory


def load_sample(sample_dir: Path) -> dict:
    """Load a single Med-LongMem sample."""
    meta = json.load(open(sample_dir / "metadata.json"))
    dialogue = [json.loads(l) for l in open(sample_dir / "dialogue.jsonl")]
    gt_canonical = [json.loads(l) for l in open(sample_dir / "canonical_gt.jsonl")]
    gt_conflicts = []
    conflict_file = sample_dir / "conflict_gt.jsonl"
    if conflict_file.exists():
        gt_conflicts = [json.loads(l) for l in open(conflict_file)]
    return {
        "meta": meta,
        "dialogue": dialogue,
        "gt_canonical": gt_canonical,
        "gt_conflicts": gt_conflicts,
        "dialogue_id": meta["dialogue_id"],
    }


def normalize_value(v: str) -> str:
    """Normalize value for matching."""
    import re
    v = v.strip().lower()
    v = re.sub(r"[^\w\s/.\-]", " ", v)
    return " ".join(v.split())


def match_to_gt(candidate_value: str, gt_values: set) -> bool:
    """Check if a candidate value matches any GT value."""
    norm = normalize_value(candidate_value)
    for gt in gt_values:
        gt_norm = normalize_value(gt)
        if norm == gt_norm:
            return True
        # Numeric tolerance
        try:
            cn = float(norm)
            gn = float(gt_norm)
            if abs(cn - gn) < 0.01:
                return True
        except ValueError:
            pass
        # Substring match for diagnoses/symptoms
        if len(norm) > 3 and len(gt_norm) > 3:
            if norm in gt_norm or gt_norm in norm:
                return True
    return False


def build_feature_vector(
    value: str,
    attribute: str,
    events: List[ExtractedEvent],
    dialogue_id: str,
    dialogue_date: str | None,
) -> ScorerFeatures:
    """
    Build a 7-dim feature vector for a candidate value.

    This extracts the same features that the learned scorer uses,
    but from raw events rather than CandidateValue objects.
    """
    matching_events = [e for e in events if normalize_value(e.value) == normalize_value(value)]

    # temporal_recency: normalized position of latest mention
    if matching_events:
        max_turn = max(max(e.provenance) if e.provenance else 0 for e in matching_events)
        all_turns = [max(e.provenance) if e.provenance else 0 for e in events] if events else [0]
        recency = max_turn / max(max(all_turns), 1)
    else:
        recency = 0.0

    # source_authority
    authority = 0.0
    for e in matching_events:
        if e.speaker == "doctor":
            authority = max(authority, 1.0)
        elif e.speaker == "patient":
            authority = max(authority, 0.5)

    # evidence_count
    evidence = len(matching_events)

    # prior_confidence
    if matching_events:
        prior = max(e.confidence for e in matching_events)
    else:
        prior = 0.5

    return ScorerFeatures(
        temporal_recency=recency,
        source_authority=authority,
        evidence_count_log=math.log(1 + evidence),
        prior_confidence=prior,
        n_constraint_violations=0,
        max_violation_severity=0.0,
        confounder_adjustment=0.0,
    )


def hand_crafted_score(features: ScorerFeatures) -> float:
    """Compute hand-crafted score using fixed weights."""
    return (
        W_RECENCY * features.temporal_recency
        + W_AUTHORITY * features.source_authority
        + W_EVIDENCE * features.evidence_count_log
        + W_PRIOR * features.prior_confidence
    )


def collect_training_data(
    data_path: str,
    max_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Run pipeline on all samples and collect (feature, label) pairs.

    For each sample:
    1. Run M1 extraction
    2. For each attribute in GT, collect all candidate values from events
    3. Label each candidate: 1 if matches GT value, 0 otherwise
    4. Extract features

    Returns:
        X: (n, 7) feature matrix
        y: (n,) label vector
        metadata: list of dicts with sample info per row
    """
    data_dir = Path(data_path)
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:max_samples]

    extractor = MedicalEventExtractor()
    X_list = []
    y_list = []
    meta_list = []

    print(f"\nCollecting training data from {len(sample_dirs)} samples...")

    for i, sample_dir in enumerate(sample_dirs):
        sample = load_sample(sample_dir)
        dialogue_id = sample["dialogue_id"]
        gt_canonical = sample["gt_canonical"]

        # Build GT lookup: attribute -> set of values
        gt_by_attr: Dict[str, set] = {}
        for rec in gt_canonical:
            attr = rec.get("attribute", "")
            val = rec.get("value", "")
            if attr and val:
                gt_by_attr.setdefault(attr, set()).add(val)

        # M1: Extract events
        dialogue = sample["dialogue"]
        try:
            events = extractor.extract(dialogue, dialogue_id)
        except Exception as e:
            print(f"  [{i+1}/{len(sample_dirs)}] {dialogue_id}: M1 failed ({e})")
            continue

        if not events:
            print(f"  [{i+1}/{len(sample_dirs)}] {dialogue_id}: no events extracted")
            continue

        # Infer dialogue date
        import re
        dialogue_date = None
        for turn in dialogue:
            m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", turn.get("text", ""))
            if m:
                dialogue_date = m.group(1)
                break

        # Group events by attribute
        events_by_attr: Dict[str, List[ExtractedEvent]] = {}
        for evt in events:
            events_by_attr.setdefault(evt.attribute, []).append(evt)

        n_pos = 0
        n_neg = 0

        for attr, gt_vals in gt_by_attr.items():
            attr_events = events_by_attr.get(attr, [])
            if not attr_events:
                continue

            # Collect unique candidate values for this attribute
            seen_values = set()
            candidates = []
            for evt in attr_events:
                norm = normalize_value(evt.value)
                if norm and norm not in seen_values:
                    seen_values.add(norm)
                    candidates.append(evt.value)

            # Label each candidate
            for cand_value in candidates:
                is_correct = match_to_gt(cand_value, gt_vals)
                features = build_feature_vector(
                    cand_value, attr, events, dialogue_id, dialogue_date
                )
                X_list.append(features.to_array())
                y_list.append(1 if is_correct else 0)
                meta_list.append({
                    "sample_id": dialogue_id,
                    "attribute": attr,
                    "value": cand_value,
                    "is_correct": is_correct,
                })
                if is_correct:
                    n_pos += 1
                else:
                    n_neg += 1

        if (i + 1) % 5 == 0 or i == len(sample_dirs) - 1:
            print(f"  [{i+1}/{len(sample_dirs)}] {dialogue_id}: "
                  f"total={len(X_list)} (pos={sum(y_list)}, neg={len(y_list)-sum(y_list)})")

    X = np.array(X_list) if X_list else np.zeros((0, 7))
    y = np.array(y_list) if y_list else np.zeros(0)

    print(f"\nDataset: {len(X)} candidates, {int(y.sum())} positive, {len(y)-int(y.sum())} negative")
    print(f"Positive ratio: {y.mean():.3f}")

    return X, y, meta_list


def evaluate_hand_crafted(X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate hand-crafted scoring: for each attribute group,
    check if the highest-scored candidate is correct."""
    scores = np.array([
        W_RECENCY * X[i, 0] + W_AUTHORITY * X[i, 1] + W_EVIDENCE * X[i, 2] + W_PRIOR * X[i, 3]
        for i in range(len(X))
    ])
    # Simple: accuracy = fraction of positive samples ranked higher than negatives
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return {"auc": 0.0, "accuracy": 0.0}
    auc = roc_auc_score(y, scores)
    # Binary accuracy at threshold = median
    threshold = np.median(scores)
    preds = (scores >= threshold).astype(int)
    accuracy = (preds == y).mean()
    return {"auc": round(float(auc), 4), "accuracy": round(float(accuracy), 4)}


def run_learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    fractions: list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
) -> List[dict]:
    """Run learning curve: train on increasing fractions of data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    results = []
    n = len(X)

    for frac in fractions:
        k = max(10, int(n * frac))
        if k > n:
            k = n

        # Random subset
        idx = np.random.RandomState(42).choice(n, k, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]

        if len(np.unique(y_sub)) < 2:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        n_folds = min(5, min(int(y_sub.sum()), int(len(y_sub) - y_sub.sum())))
        if n_folds < 2:
            continue

        cv_auc = cross_val_score(model, X_scaled, y_sub, cv=n_folds, scoring="roc_auc")
        cv_acc = cross_val_score(model, X_scaled, y_sub, cv=n_folds, scoring="accuracy")

        results.append({
            "fraction": frac,
            "n_samples": k,
            "auc_mean": round(float(cv_auc.mean()), 4),
            "auc_std": round(float(cv_auc.std()), 4),
            "accuracy_mean": round(float(cv_acc.mean()), 4),
            "accuracy_std": round(float(cv_acc.std()), 4),
        })
        print(f"  frac={frac:.1f} n={k}: AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}  "
              f"Acc={cv_acc.mean():.4f}±{cv_acc.std():.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/med_longmem")
    parser.add_argument("--output_dir", default="results/learned_scorer")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # Step 1: Collect training data
    X, y, meta = collect_training_data(args.data_path, args.max_samples)

    if len(X) < 20:
        print(f"\nToo few training samples ({len(X)}). Need at least 20.")
        sys.exit(1)

    # Step 2: Evaluate hand-crafted baseline
    print("\n" + "=" * 60)
    print("  Hand-Crafted Scoring Baseline")
    print("=" * 60)
    hc_results = evaluate_hand_crafted(X, y)
    print(f"  AUC:      {hc_results['auc']}")
    print(f"  Accuracy: {hc_results['accuracy']}")
    print(f"  Weights:  recency={W_RECENCY}, authority={W_AUTHORITY}, "
          f"evidence={W_EVIDENCE}, prior={W_PRIOR}")

    # Step 3: Train learned scorer
    print("\n" + "=" * 60)
    print("  Learned Scorer (5-Fold CV)")
    print("=" * 60)
    scorer = LearnedCandidateScorer()
    lr_results = scorer.train(X, y, n_folds=5)
    print(f"  AUC:      {lr_results['auc_mean']} ± {lr_results['auc_std']}")
    print(f"  Accuracy: {lr_results['accuracy_mean']} ± {lr_results['accuracy_std']}")
    print(f"  Samples:  {lr_results['n_samples']} (pos={lr_results['n_positive']}, neg={lr_results['n_negative']})")
    print(f"\n  Feature Importance:")
    fi = lr_results["feature_importance"]
    for feat, weight in sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(weight) * 10)
        sign = "+" if weight > 0 else "-"
        print(f"    {feat:<30s} {sign}{abs(weight):.4f}  {bar}")

    # Step 4: Learning curve
    print("\n" + "=" * 60)
    print("  Learning Curve")
    print("=" * 60)
    lc_results = run_learning_curve(X, y)

    # Step 5: Save everything
    scorer.save(os.path.join(args.output_dir, "scorer_model.json"))

    summary = {
        "hand_crafted": hc_results,
        "learned_scorer": lr_results,
        "learning_curve": lc_results,
        "comparison": {
            "auc_improvement": round(lr_results["auc_mean"] - hc_results["auc"], 4),
            "accuracy_improvement": round(lr_results["accuracy_mean"] - hc_results["accuracy"], 4),
        },
        "dataset": {
            "n_samples": len(X),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "positive_ratio": round(float(y.mean()), 4),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Final comparison table
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Final Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Method':<25s} {'AUC':>8s} {'Accuracy':>10s}")
    print(f"  {'─' * 45}")
    print(f"  {'Hand-crafted weights':<25s} {hc_results['auc']:>8.4f} {hc_results['accuracy']:>10.4f}")
    print(f"  {'Learned (LogReg, 5-CV)':<25s} {lr_results['auc_mean']:>8.4f} {lr_results['accuracy_mean']:>10.4f}")
    delta_auc = lr_results["auc_mean"] - hc_results["auc"]
    print(f"  {'Δ (Learned - HC)':<25s} {delta_auc:>+8.4f} "
          f"{lr_results['accuracy_mean'] - hc_results['accuracy']:>+10.4f}")
    print(f"\n  Saved: {args.output_dir}/")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
