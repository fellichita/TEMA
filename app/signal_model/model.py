"""Логистическая регрессия на интерпретируемых признаках: обучение, проверка, объяснение.

Модель хранится в JSON (медианы, масштаб, веса), а не в pickle: файл читается
человеком, а предсказание не исполняет чужой код. Вклад признака в решение —
вес, умноженный на стандартизованное значение; сумма вкладов и свободного
члена даёт логит вероятности «слабый сигнал».
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.signal_model.dataset import Technology
from app.signal_model.features import FEATURE_NAMES, FEATURES, describe, extract

MODEL_VERSION = "weak-signal-logreg/1.0.0"
REGULARIZATION = 0.5
THRESHOLD = 0.5
HIGH_CONFIDENCE = 0.75
LABELS = {feature.name: feature.label for feature in FEATURES}


def matrix(technologies: Sequence[Technology], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    rows = [extract(item) for item in technologies]
    return np.array([[np.nan if row[name] is None else row[name] for name in names] for row in rows], dtype=float)


def pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=REGULARIZATION, class_weight="balanced", max_iter=2000))])


@dataclass(frozen=True)
class Contribution:
    name: str
    label: str
    value: str
    weight: float


@dataclass(frozen=True)
class Prediction:
    title: str
    probability: float
    is_signal: bool
    contributions: tuple[Contribution, ...]


class SignalModel:
    """Обученная модель в виде чисел; работает без sklearn."""

    def __init__(self, payload: dict):
        if payload.get("version") != MODEL_VERSION or tuple(payload["features"]) != FEATURE_NAMES:
            raise ValueError("Файл модели не совпадает с текущим набором признаков.")
        self.payload = payload
        self.medians = np.array(payload["medians"], dtype=float)
        self.means = np.array(payload["means"], dtype=float)
        self.scales = np.array(payload["scales"], dtype=float)
        self.weights = np.array(payload["weights"], dtype=float)
        self.intercept = float(payload["intercept"])

    @classmethod
    def load(cls, path: str | Path) -> SignalModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def predict(self, technologies: Sequence[Technology]) -> list[Prediction]:
        if not technologies:
            return []
        raw = matrix(technologies)
        filled = np.where(np.isnan(raw), self.medians, raw)
        standardized = (filled - self.means) / self.scales
        results = []
        for item, row, values in zip(technologies, standardized, raw, strict=True):
            parts = self.weights * row
            probability = 1 / (1 + math.exp(-(self.intercept + float(parts.sum()))))
            contributions = tuple(sorted(
                (Contribution(name, LABELS[name], describe(name, None if np.isnan(value) else float(value)),
                              round(float(part), 3))
                 for name, part, value in zip(FEATURE_NAMES, parts, values, strict=True)),
                key=lambda contribution: -abs(contribution.weight)))
            results.append(Prediction(item.title, round(probability, 4), probability >= THRESHOLD, contributions))
        return results


def fit(technologies: Sequence[Technology]) -> SignalModel:
    labels = np.array([item.label for item in technologies])
    fitted = pipeline().fit(matrix(technologies), labels)
    imputer, scaler, model = (fitted.named_steps[name] for name in ("impute", "scale", "model"))
    return SignalModel({
        "version": MODEL_VERSION, "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": list(FEATURE_NAMES), "labels": [LABELS[name] for name in FEATURE_NAMES],
        "medians": imputer.statistics_.tolist(), "means": scaler.mean_.tolist(),
        "scales": [float(value) if value > 0 else 1.0 for value in scaler.scale_],
        "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "positives": int(labels.sum()), "negatives": int(len(labels) - labels.sum()),
    })


def _scores(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= THRESHOLD
    return {"accuracy": accuracy_score(labels, predicted), "precision": precision_score(labels, predicted, zero_division=0),
            "recall": recall_score(labels, predicted, zero_division=0), "f1": f1_score(labels, predicted, zero_division=0),
            "roc_auc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan"),
            "brier": brier_score_loss(labels, probabilities)}


def cross_validate(technologies: Sequence[Technology], names: Sequence[str] = FEATURE_NAMES, *,
                   repeats: int = 10, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Повторная стратифицированная 5-кратная проверка: среднее и стандартное отклонение."""
    data, labels = matrix(technologies, names), np.array([item.label for item in technologies])
    folds = []
    for train, test in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(data, labels):
        model = pipeline().fit(data[train], labels[train])
        folds.append(_scores(labels[test], model.predict_proba(data[test])[:, 1]))
    return {key: (float(np.mean([fold[key] for fold in folds])), float(np.std([fold[key] for fold in folds])))
            for key in folds[0]}


def out_of_fold(technologies: Sequence[Technology], *, seed: int = 0) -> np.ndarray:
    data, labels = matrix(technologies), np.array([item.label for item in technologies])
    probabilities = np.zeros(len(labels))
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(data, labels):
        probabilities[test] = pipeline().fit(data[train], labels[train]).predict_proba(data[test])[:, 1]
    return probabilities


def leave_area_out(technologies: Sequence[Technology]) -> dict[str, dict[str, float]]:
    """Обучение без целой области и проверка на ней: модель не должна зубрить темы."""
    data, labels = matrix(technologies), np.array([item.label for item in technologies])
    areas = np.array([item.area for item in technologies])
    results = {}
    for area in sorted(set(areas)):
        test = areas == area
        model = pipeline().fit(data[~test], labels[~test])
        results[area] = _scores(labels[test], model.predict_proba(data[test])[:, 1])
    return results
