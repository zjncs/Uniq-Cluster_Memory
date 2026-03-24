"""
m3_uniqueness/learned_scorer.py
================================
可学习的候选值评分模型（Learned Candidate Scorer）。

替代 conflict_detector.py 中手写权重 (W_RECENCY=0.35, ...) 的线性加权，
使用逻辑回归从 Med-LongMem GT 中学习最优权重组合。

特征向量：
    1. temporal_recency     — 时间新近度 [0,1]
    2. source_authority     — 来源权威度 (doctor=1.0, patient=0.5, unknown=0.0)
    3. evidence_count       — log(1 + 证据条数)
    4. prior_confidence     — M1 提取的原始置信度
    5. n_constraint_violations — 形式约束违反数量
    6. max_violation_severity  — 最严重违反的惩罚值 (0=无违反, 1=硬违反)
    7. confounder_adjustment   — 因果去混淆调整量

训练方式：
    - 正样本：GT value 对应的候选
    - 负样本：非 GT value 的候选
    - 模型：sklearn LogisticRegression（轻量，可解释，无需 GPU）
    - 交叉验证：5-fold，报告 AUC 和准确率

这比手写权重有两个优势：
    1. 权重从数据中学习，避免人工调参偏差
    2. 可以报告 feature importance，增加可解释性
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.uniq_cluster_memory.schema import CandidateValue, CanonicalMemory


@dataclass
class ScorerFeatures:
    """单个候选值的特征向量。"""
    temporal_recency: float = 0.0
    source_authority: float = 0.0
    evidence_count_log: float = 0.0
    prior_confidence: float = 0.0
    n_constraint_violations: int = 0
    max_violation_severity: float = 0.0
    confounder_adjustment: float = 0.0

    def to_array(self) -> list:
        return [
            self.temporal_recency,
            self.source_authority,
            self.evidence_count_log,
            self.prior_confidence,
            self.n_constraint_violations,
            self.max_violation_severity,
            self.confounder_adjustment,
        ]

    @staticmethod
    def feature_names() -> list:
        return [
            "temporal_recency",
            "source_authority",
            "evidence_count_log",
            "prior_confidence",
            "n_constraint_violations",
            "max_violation_severity",
            "confounder_adjustment",
        ]


def extract_features(candidate: CandidateValue) -> ScorerFeatures:
    """从 CandidateValue 提取特征。"""
    return ScorerFeatures(
        temporal_recency=candidate.temporal_recency,
        source_authority=candidate.source_authority,
        evidence_count_log=math.log(1 + candidate.evidence_count),
        prior_confidence=candidate.confidence,
    )


def extract_features_from_dict(d: dict) -> ScorerFeatures:
    """从字典格式的候选值提取特征。"""
    return ScorerFeatures(
        temporal_recency=d.get("temporal_recency", 0.0),
        source_authority=d.get("source_authority", 0.0),
        evidence_count_log=math.log(1 + d.get("evidence_count", 1)),
        prior_confidence=d.get("confidence", 0.0),
    )


class LearnedCandidateScorer:
    """
    可学习的候选评分器。

    使用逻辑回归预测每个候选值是否为正确答案，
    然后用预测概率作为 confidence score。
    """

    def __init__(self):
        self._model = None
        self._is_trained = False
        self._feature_names = ScorerFeatures.feature_names()
        self._weights: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """
        训练逻辑回归模型。

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)，1=正确候选，0=错误候选
            n_folds: 交叉验证折数

        Returns:
            包含 accuracy, auc, feature_importance 的字典。
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # 交叉验证
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv_accuracy = cross_val_score(model, X_scaled, y, cv=min(n_folds, len(y)), scoring="accuracy")
        cv_auc = cross_val_score(model, X_scaled, y, cv=min(n_folds, len(y)), scoring="roc_auc")

        # 全量训练
        model.fit(X_scaled, y)
        self._model = model
        self._is_trained = True
        self._weights = model.coef_[0]
        self._intercept = model.intercept_[0]

        # Feature importance
        importance = dict(zip(self._feature_names, self._weights.tolist()))

        return {
            "accuracy_mean": round(float(cv_accuracy.mean()), 4),
            "accuracy_std": round(float(cv_accuracy.std()), 4),
            "auc_mean": round(float(cv_auc.mean()), 4),
            "auc_std": round(float(cv_auc.std()), 4),
            "feature_importance": importance,
            "intercept": round(float(self._intercept), 4),
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }

    def score_candidates(
        self,
        candidates: List[CandidateValue],
    ) -> List[CandidateValue]:
        """
        用训练好的模型对候选值评分。

        Returns:
            按 confidence 降序排列的候选列表。
        """
        if not self._is_trained or not candidates:
            return candidates

        features = [extract_features(c).to_array() for c in candidates]
        X = self._scaler.transform(np.array(features))
        probs = self._model.predict_proba(X)[:, 1]

        for cand, prob in zip(candidates, probs):
            cand.confidence = round(float(prob), 4)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def save(self, path: str) -> None:
        """保存模型参数到 JSON（不依赖 pickle，便于版本控制）。"""
        if not self._is_trained:
            return
        data = {
            "weights": self._weights.tolist(),
            "intercept": float(self._intercept),
            "scaler_mean": self._scaler.mean_.tolist(),
            "scaler_scale": self._scaler.scale_.tolist(),
            "feature_names": self._feature_names,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """从 JSON 加载模型参数。"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        with open(path) as f:
            data = json.load(f)

        self._weights = np.array(data["weights"])
        self._intercept = data["intercept"]
        self._feature_names = data["feature_names"]

        # 重建 scaler
        self._scaler = StandardScaler()
        self._scaler.mean_ = np.array(data["scaler_mean"])
        self._scaler.scale_ = np.array(data["scaler_scale"])
        self._scaler.var_ = self._scaler.scale_ ** 2
        self._scaler.n_features_in_ = len(self._feature_names)
        self._scaler.n_samples_seen_ = 1  # placeholder

        # 重建模型
        model = LogisticRegression()
        model.coef_ = self._weights.reshape(1, -1)
        model.intercept_ = np.array([self._intercept])
        model.classes_ = np.array([0, 1])
        self._model = model
        self._is_trained = True


def build_training_data_from_results(
    results_dir: str,
    gt_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从实验结果和 GT 构建训练数据。

    扫描 results_dir 中的 ablation_full.jsonl（或其他结果文件），
    提取每个样本的 candidate_values，与 GT 匹配生成标签。

    Returns:
        X: (n_candidates, n_features) 特征矩阵
        y: (n_candidates,) 标签向量
    """
    X_list = []
    y_list = []

    results_path = Path(results_dir)
    gt_path = Path(gt_dir)

    # 遍历所有样本目录
    for sample_dir in sorted(gt_path.iterdir()):
        if not sample_dir.is_dir():
            continue
        gt_file = sample_dir / "canonical_gt.jsonl"
        if not gt_file.exists():
            continue

        # 加载 GT values
        gt_values = set()
        with open(gt_file) as f:
            for line in f:
                rec = json.loads(line)
                gt_values.add(rec.get("value", "").strip().lower())

        # 查找对应的预测结果（从 ablation 详细结果中）
        detail_file = results_path / "ablation_full.jsonl"
        if not detail_file.exists():
            continue

        with open(detail_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("sample_id") != sample_dir.name:
                    continue
                # 这里需要从 pipeline 输出中获取 candidate_values
                # 由于当前结果格式不含 candidate_values，
                # 我们在后续版本中会扩展结果格式
                break

    return np.array(X_list) if X_list else np.zeros((0, 7)), np.array(y_list) if y_list else np.zeros(0)
