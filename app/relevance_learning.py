"""Самообучение на своих анализах: модель «по теме / не по теме» и доли источников.

Каждый завершённый анализ оставляет размеченные примеры: материал выдачи,
его лексическое совпадение с темой и метку. Метки берутся из сигналов,
независимых от признаков модели:

* решение этапа обнаружения по полному тексту документа (смысловая модель E5
  по всем фрагментам аннотации: «прошёл порог» / «ниже порога» / «ближе к
  исключённой области»);
* принадлежность публикации теме, которую методика подтвердила как тренд;
* для материалов дополнительных источников — только однозначные смысловые
  оценки (явно по теме или явно мимо), пограничные не используются;
* отметки владельца в панели («по теме» / «не по теме») — самые весомые.

Модель — логистическая регрессия без смысловой модели в признаках: лексика
темы, источник, вид материала, свежесть и «слова-шум» (хешированные слова
заголовка без слов запроса). Поэтому она дополняет эвристику тем, чему
научилась на прошлых анализах: какие источники обычно приносят мусор и какие
слова выдают чужую тему. Качество проверяется честно — на последних анализах,
которых модель при обучении не видела, в сравнении с одной эвристикой. Если
модель не лучше эвристики, её вес в итоговой оценке нулевой.

Из тех же примеров считается точность каждого источника; при включённой
адаптации источники с высокой долей материалов по теме получают большую
долю сбора в следующих анализах (общий объём сбора не растёт).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any

from app.topic_relevance import LexicalMatch, TopicProfile, lexical_match, stem

LEARNING_VERSION = "relevance-learner/1.0.0"
DIRECTORY = "learning"
MODEL_FILE = "relevance-model.json"
SETTINGS_FILE = "settings.json"
FEEDBACK_FILE = "feedback.json"
HISTORY_FILE = "history.jsonl"
SAMPLES = "samples"
HASH_BUCKETS = 2048
MAX_SAMPLES_PER_RUN = 600
MAX_TRAINING_SAMPLES = 60_000
MIN_SAMPLES = 200
MIN_RUNS = 2
ITERATIONS = 250
L2 = 1e-3
# Однозначные смысловые оценки материала источника.
SEMANTIC_POSITIVE = 0.84
SEMANTIC_NEGATIVE = 0.76
DISCOVERY_POSITIVE = 0.82
WEIGHTS = {"feedback": 4.0, "trend": 2.0, "discovery": 1.0, "semantic": 0.7}
MIN_SOURCE_ITEMS = 20
DEFAULT_SETTINGS = {"enabled": True, "auto_retrain": True, "adapt_sources": True}
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_PUBLICATION = re.compile(r"[0-9a-f]{64}\Z")
_NOISE_STOP = frozenset("""
the and for with from that this are was were have has into over under about new using based study
analysis via its their our can will more most than also between within without after before toward
для как что это при или его еще уже чем так все был была были они она оно
""".split())


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def learning_directory(data_dir: Path) -> Path:
    return Path(data_dir) / DIRECTORY


def _freshness(published: object, as_of: date | None) -> float:
    if not isinstance(published, str) or as_of is None:
        return 0.0
    try:
        moment = date.fromisoformat(published[:10])
    except ValueError:
        return 0.0
    days = max(0, (as_of - moment).days)
    return max(0.0, 1.0 - days / 1095)


def _bucket(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest(), "big") % HASH_BUCKETS


def noise_words(title: str, summary: str | None, profile_terms: set[str]) -> list[str]:
    """Слова материала, не входящие в тему: из них модель учит «чужие» темы."""
    words = []
    for word in _WORD.findall(f"{title} {summary or ''}".casefold()):
        if word in _NOISE_STOP:
            continue
        root = stem(word)
        if root in profile_terms:
            continue
        words.append(root)
    return list(dict.fromkeys(words))[:60]


def profile_terms(profile: TopicProfile) -> set[str]:
    return {term for formulation in profile.formulations for term, _ in formulation.terms}


def features(item: Mapping[str, Any], match: LexicalMatch, terms: set[str],
             as_of: date | None = None) -> dict[str, float]:
    """Признаки одного материала; смысловой модели среди них нет намеренно."""
    title = str(item.get("title") or "")
    summary = item.get("summary") if isinstance(item.get("summary"), str) else None
    values: dict[str, float] = {
        "bias": 1.0, "lexical": match.score, "coverage_title": match.coverage_title,
        "coverage_text": match.coverage_text, "phrase_title": float(match.phrase_title),
        "phrase_text": float(match.phrase_text), "excluded": float(match.excluded),
        "has_summary": float(bool(summary and summary.strip())),
        "title_length": min(1.0, len(title) / 200),
        "freshness": _freshness(item.get("published_at"), as_of),
        f"source={item.get('source_id')}": 1.0, f"kind={item.get('kind')}": 1.0,
    }
    words = noise_words(title, summary, terms)
    if words:
        scale = 1 / math.sqrt(len(words))
        for word in words:
            key = f"w{_bucket(word)}"
            values[key] = values.get(key, 0.0) + scale
    return values


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def auc(scores: Sequence[float], labels: Sequence[int], weights: Sequence[float] | None = None) -> float | None:
    """Площадь под ROC (вероятность, что пример «по теме» оценён выше примера «мимо»)."""
    pairs = sorted(zip(scores, labels, weights or [1.0] * len(labels), strict=True), key=lambda item: item[0])
    positive = sum(weight for _, label, weight in pairs if label == 1)
    negative = sum(weight for _, label, weight in pairs if label == 0)
    if positive <= 0 or negative <= 0:
        return None
    seen_negative = 0.0
    area = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        group_negative = sum(weight for _, label, weight in pairs[index:end] if label == 0)
        group_positive = sum(weight for _, label, weight in pairs[index:end] if label == 1)
        area += group_positive * (seen_negative + group_negative / 2)
        seen_negative += group_negative
        index = end
    return round(area / (positive * negative), 4)


@dataclass
class RelevanceModel:
    """Обученная модель; пустая (без весов) ничего не меняет в оценке."""

    weights: dict[str, float] = field(default_factory=dict)
    version: str | None = None
    trained_at: str | None = None
    samples: int = 0
    runs: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    vocabulary: dict[str, str] = field(default_factory=dict)
    blend: float = 0.0

    @property
    def ready(self) -> bool:
        return bool(self.weights) and self.blend > 0

    def predict(self, values: Mapping[str, float]) -> float:
        return _sigmoid(sum(self.weights.get(name, 0.0) * value for name, value in values.items()))

    def scorer(self, profile: TopicProfile, as_of: date | None = None) -> Callable[[Mapping[str, Any], LexicalMatch],
                                                                                    float] | None:
        if not self.ready:
            return None
        terms = profile_terms(profile)
        return lambda item, match: self.predict(features(item, match, terms, as_of))

    def source_weights(self) -> dict[str, float]:
        """Доли сбора по точности источников в прошлых анализах (1,0 — как у всех)."""
        rated = {source: stats["precision"] for source, stats in self.sources.items()
                 if stats.get("items", 0) >= MIN_SOURCE_ITEMS and isinstance(stats.get("precision"), (int, float))}
        if len(rated) < 3:
            return {}
        mean = sum(rated.values()) / len(rated)
        if mean <= 0:
            return {}
        return {source: round(max(0.4, min(1.6, 0.5 + 0.5 * precision / mean)), 3)
                for source, precision in rated.items()}

    def to_json(self) -> dict[str, Any]:
        return {"learning_version": LEARNING_VERSION, "version": self.version, "trained_at": self.trained_at,
                "samples": self.samples, "runs": self.runs, "metrics": self.metrics, "sources": self.sources,
                "vocabulary": self.vocabulary, "blend": self.blend,
                "weights": {name: round(value, 6) for name, value in sorted(self.weights.items())}}

    @classmethod
    def from_json(cls, value: object) -> RelevanceModel:
        if not isinstance(value, dict) or value.get("learning_version") != LEARNING_VERSION:
            return cls()
        weights = value.get("weights")
        blend = value.get("blend")
        if (not isinstance(weights, dict) or not all(isinstance(name, str) and isinstance(item, (int, float))
                                                      and math.isfinite(item) for name, item in weights.items())
                or not isinstance(blend, (int, float)) or not 0 <= blend <= 1):
            return cls()
        return cls(weights={name: float(item) for name, item in weights.items()},
                   version=value.get("version") if isinstance(value.get("version"), str) else None,
                   trained_at=value.get("trained_at") if isinstance(value.get("trained_at"), str) else None,
                   samples=int(value.get("samples") or 0), runs=int(value.get("runs") or 0),
                   metrics=_mapping(value.get("metrics")), sources=_mapping(value.get("sources")),
                   vocabulary=_mapping(value.get("vocabulary")),
                   blend=float(blend))

    @classmethod
    def load(cls, data_dir: Path | None) -> RelevanceModel:
        if data_dir is None or not load_settings(data_dir)["enabled"]:
            return cls()
        try:
            return cls.from_json(json.loads((learning_directory(data_dir) / MODEL_FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return cls()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def load_settings(data_dir: Path) -> dict[str, bool]:
    try:
        raw = json.loads((learning_directory(data_dir) / SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(raw, dict):
        settings.update({name: value for name, value in raw.items()
                         if name in DEFAULT_SETTINGS and type(value) is bool})
    return settings


def save_settings(data_dir: Path, values: Mapping[str, object]) -> dict[str, bool]:
    if not isinstance(values, Mapping) or not values or any(
            name not in DEFAULT_SETTINGS or type(value) is not bool for name, value in values.items()):
        raise ValueError("learning_settings")
    settings = load_settings(data_dir)
    settings.update({name: bool(value) for name, value in values.items()})
    _write_json(learning_directory(data_dir) / SETTINGS_FILE, settings)
    return settings


# --- Примеры одного анализа -------------------------------------------------------------------


def _label(item: Mapping[str, Any], relevance: Mapping[str, Any] | None,
           discovery: Mapping[str, tuple[float, str]]) -> tuple[int, float, str] | None:
    """Метка материала из независимых сигналов или ничего, если сигнал неоднозначен."""
    trend = item.get("trend")
    if isinstance(trend, dict) and trend.get("confidence") in {"medium", "high"}:
        return 1, WEIGHTS["trend"], "trend"
    studies = [discovery[study] for study in item.get("study_ids") or () if study in discovery]
    if studies:
        decisions = {decision for _, decision in studies}
        best = max(score for score, _ in studies)
        if "closer_to_excluded_scope" in decisions and "retained" not in decisions:
            return 0, WEIGHTS["discovery"], "discovery"
        if "retained" in decisions and best >= DISCOVERY_POSITIVE:
            return 1, WEIGHTS["discovery"], "discovery"
        if decisions == {"below_semantic_threshold"}:
            return 0, WEIGHTS["discovery"], "discovery"
        return None
    cosine = relevance.get("cosine") if isinstance(relevance, Mapping) else None
    if isinstance(cosine, (int, float)) and math.isfinite(cosine):
        if cosine >= SEMANTIC_POSITIVE:
            return 1, WEIGHTS["semantic"], "semantic"
        if cosine <= SEMANTIC_NEGATIVE:
            return 0, WEIGHTS["semantic"], "semantic"
    return None


def _order_key(run_id: str, publication_id: str) -> str:
    return hashlib.sha256(f"{run_id}\0{publication_id}".encode("utf-8")).hexdigest()


def run_samples(run_id: str, plan: Mapping[str, Any], pool: Sequence[Mapping[str, Any]], *,
                relevance: Mapping[str, Mapping[str, Any]] | None = None,
                discovery: Mapping[str, tuple[float, str]] | None = None,
                created_at: str | None = None) -> dict[str, Any]:
    """Сбалансированные примеры одного анализа для обучения (до 600 на анализ)."""
    relevance = relevance or {}
    discovery = discovery or {}
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for item in pool:
        publication_id = item.get("publication_id")
        title = item.get("title")
        if not isinstance(publication_id, str) or not isinstance(title, str) or not title.strip():
            continue
        label = _label(item, relevance.get(publication_id), discovery)
        if label is None:
            continue
        sample = {"publication_id": publication_id, "title": title[:500],
                  "summary": item["summary"][:500] if isinstance(item.get("summary"), str) else None,
                  "source_id": str(item.get("source_id") or "")[:40], "kind": str(item.get("kind") or "")[:40],
                  "published_at": item.get("published_at") if isinstance(item.get("published_at"), str) else None,
                  "label": label[0], "weight": label[1], "origin": label[2]}
        (positives if label[0] else negatives).append(sample)
    # Поровну «по теме» и «мимо»; недостающее место одного класса отдаётся другому.
    for group in (positives, negatives):
        group.sort(key=lambda sample: _order_key(run_id, sample["publication_id"]))
    half = MAX_SAMPLES_PER_RUN // 2
    take_positive = min(len(positives), max(half, MAX_SAMPLES_PER_RUN - len(negatives)))
    take_negative = min(len(negatives), MAX_SAMPLES_PER_RUN - take_positive)
    chosen = positives[:take_positive] + negatives[:take_negative]
    plan_fields = {name: plan.get(name) for name in ("original_query", "english_query", "synonyms",
                                                    "subdirections", "exclusions", "as_of")}
    return {"learning_version": LEARNING_VERSION, "run_id": run_id, "created_at": created_at,
            "plan": plan_fields, "samples": chosen}


def save_run_samples(data_dir: Path, record: Mapping[str, Any]) -> Path | None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        return None
    path = learning_directory(data_dir) / SAMPLES / f"{run_id}.json"
    _write_json(path, record)
    return path


def sampled_runs(data_dir: Path) -> set[str]:
    directory = learning_directory(data_dir) / SAMPLES
    try:
        return {path.stem for path in directory.glob("*.json")}
    except OSError:
        return set()


def _read_samples(data_dir: Path) -> list[dict[str, Any]]:
    records = []
    directory = learning_directory(data_dir) / SAMPLES
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(record, dict) and record.get("learning_version") == LEARNING_VERSION
                and isinstance(record.get("samples"), list) and isinstance(record.get("plan"), dict)):
            records.append(record)
    records.sort(key=lambda record: (str(record.get("created_at") or ""), str(record.get("run_id"))))
    return records


# --- Отметки владельца ------------------------------------------------------------------------


def load_feedback(data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        raw = json.loads((learning_directory(data_dir) / FEEDBACK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {run: items for run, items in raw.items() if isinstance(run, str) and isinstance(items, dict)}


def set_feedback(data_dir: Path, run_id: str, item: Mapping[str, Any], label: int | None) -> dict[str, Any]:
    """Отметить материал анализа «по теме» (1), «не по теме» (0) или снять отметку (None)."""
    publication_id = item.get("publication_id")
    if (not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None
            or not isinstance(publication_id, str) or _PUBLICATION.fullmatch(publication_id) is None
            or label not in {0, 1, None} or not isinstance(item.get("title"), str) or not item["title"].strip()):
        raise ValueError("feedback")
    feedback = load_feedback(data_dir)
    marks = feedback.setdefault(run_id, {})
    if label is None:
        marks.pop(publication_id, None)
        if not marks:
            feedback.pop(run_id, None)
    else:
        marks[publication_id] = {
            "label": label, "title": item["title"][:500],
            "summary": item["summary"][:500] if isinstance(item.get("summary"), str) else None,
            "source_id": str(item.get("source_id") or "")[:40], "kind": str(item.get("kind") or "")[:40],
            "published_at": item.get("published_at") if isinstance(item.get("published_at"), str) else None,
            "at": datetime.now(UTC).isoformat()}
    _write_json(learning_directory(data_dir) / FEEDBACK_FILE, feedback)
    return {"run_id": run_id, "publication_id": publication_id, "label": label,
            "marked": sum(len(items) for items in feedback.values())}


# --- Обучение ---------------------------------------------------------------------------------


@dataclass
class _Dataset:
    rows: list[dict[str, float]]
    labels: list[int]
    weights: list[float]
    groups: list[int]
    lexical: list[float]
    sources: list[str]
    origins: list[str]
    words: Counter[tuple[int, str]]


def _dataset(records: Sequence[Mapping[str, Any]], feedback: Mapping[str, Mapping[str, Mapping[str, Any]]]
             ) -> _Dataset:
    data = _Dataset([], [], [], [], [], [], [], Counter())
    for group, record in enumerate(records):
        plan = record["plan"]
        profile = TopicProfile.from_plan(plan)
        terms = profile_terms(profile)
        as_of = None
        if isinstance(plan.get("as_of"), str):
            try:
                as_of = date.fromisoformat(plan["as_of"])
            except ValueError:
                as_of = None
        samples = {sample["publication_id"]: dict(sample) for sample in record["samples"]
                   if isinstance(sample, dict) and isinstance(sample.get("publication_id"), str)}
        for publication_id, mark in feedback.get(str(record.get("run_id")), {}).items():
            if isinstance(mark, dict) and mark.get("label") in {0, 1} and isinstance(mark.get("title"), str):
                samples[publication_id] = {**mark, "publication_id": publication_id,
                                           "weight": WEIGHTS["feedback"], "origin": "feedback"}
        for sample in samples.values():
            label = sample.get("label")
            if label not in {0, 1} or not isinstance(sample.get("title"), str):
                continue
            match = lexical_match(profile, sample["title"], sample.get("summary"))
            data.rows.append(features(sample, match, terms, as_of))
            data.labels.append(int(label))
            data.weights.append(float(sample.get("weight") or 1.0))
            data.groups.append(group)
            data.lexical.append(match.score)
            data.sources.append(str(sample.get("source_id") or ""))
            data.origins.append(str(sample.get("origin") or ""))
            for word in noise_words(sample["title"], sample.get("summary"), terms):
                data.words[(_bucket(word), word)] += 1
    return data


def _fit(rows: Sequence[Mapping[str, float]], labels: Sequence[int], weights: Sequence[float]) -> dict[str, float]:
    import numpy as np

    names = sorted({name for row in rows for name in row})
    index = {name: position for position, name in enumerate(names)}
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    indices = np.fromiter((index[name] for row in rows for name in row), dtype=np.int64, count=int(indptr[-1]))
    values = np.fromiter((value for row in rows for value in row.values()), dtype=np.float64, count=int(indptr[-1]))
    target = np.asarray(labels, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    # Классы уравниваются: примеров «мимо» обычно больше, чем «по теме».
    for label in (0, 1):
        mask = target == label
        total = sample_weight[mask].sum()
        if total > 0:
            sample_weight[mask] *= sample_weight.sum() / (2 * total)
    sample_weight /= sample_weight.sum()
    weight = np.zeros(len(names))
    regularized = np.asarray([0.0 if name == "bias" else 1.0 for name in names])
    moment = np.zeros(len(names))
    velocity = np.zeros(len(names))
    rows_of = np.repeat(np.arange(len(rows)), lengths)
    for step in range(1, ITERATIONS + 1):
        margin = np.bincount(rows_of, weights=weight[indices] * values, minlength=len(rows))
        probability = 1 / (1 + np.exp(-np.clip(margin, -30, 30)))
        residual = sample_weight * (probability - target)
        gradient = np.bincount(indices, weights=residual[rows_of] * values, minlength=len(names))
        gradient += L2 * regularized * weight
        # Adam: устойчиво на признаках очень разного масштаба.
        moment = 0.9 * moment + 0.1 * gradient
        velocity = 0.999 * velocity + 0.001 * gradient ** 2
        weight -= 0.1 * (moment / (1 - 0.9 ** step)) / (np.sqrt(velocity / (1 - 0.999 ** step)) + 1e-8)
    return {name: float(value) for name, value in zip(names, weight, strict=True) if abs(value) > 1e-6}


def _evaluate(model: Mapping[str, float], rows: Sequence[Mapping[str, float]], labels: Sequence[int],
              weights: Sequence[float], lexical: Sequence[float]) -> dict[str, Any]:
    scores = [_sigmoid(sum(model.get(name, 0.0) * value for name, value in row.items())) for row in rows]
    correct = sum(weight for score, label, weight in zip(scores, labels, weights, strict=True)
                  if (score >= 0.5) == bool(label))
    baseline_correct = sum(weight for score, label, weight in zip(lexical, labels, weights, strict=True)
                           if (score >= 0.5) == bool(label))
    total = sum(weights) or 1.0
    return {"auc": auc(scores, labels, weights), "baseline_auc": auc(lexical, labels, weights),
            "accuracy": round(correct / total, 4), "baseline_accuracy": round(baseline_correct / total, 4),
            "samples": len(rows), "positive": sum(labels)}


def train(data_dir: Path, *, now: datetime | None = None) -> RelevanceModel:
    """Переобучить модель на всех сохранённых анализах и записать её с историей качества."""
    records = _read_samples(data_dir)
    feedback = load_feedback(data_dir)
    known = {str(record.get("run_id")) for record in records}
    # Отметки к анализам без сохранённых примеров тоже учат модель.
    for run_id, marks in feedback.items():
        if run_id not in known and marks:
            records.append({"run_id": run_id, "created_at": None, "plan": {}, "samples": []})
    data = _dataset(records, feedback)
    if len(data.rows) > MAX_TRAINING_SAMPLES:
        cut = len(data.rows) - MAX_TRAINING_SAMPLES
        for name in ("rows", "labels", "weights", "groups", "lexical", "sources", "origins"):
            setattr(data, name, getattr(data, name)[cut:])
    moment = (now or datetime.now(UTC)).isoformat()
    groups = sorted(set(data.groups))
    model = RelevanceModel(trained_at=moment, samples=len(data.rows), runs=len(groups))
    model.sources = _source_stats(data)
    if len(data.rows) < MIN_SAMPLES or len(groups) < MIN_RUNS or len(set(data.labels)) < 2:
        model.metrics = {"state": "collecting", "needed_samples": MIN_SAMPLES, "needed_runs": MIN_RUNS}
        _save_model(data_dir, model)
        return model
    # Честная проверка: последние анализы (не меньше одного) модель не видит при обучении.
    holdout_groups = set(groups[-max(1, len(groups) // 5):])
    train_rows = [index for index, group in enumerate(data.groups) if group not in holdout_groups]
    test_rows = [index for index, group in enumerate(data.groups) if group in holdout_groups]
    metrics: dict[str, Any] = {"state": "trained", "holdout_runs": len(holdout_groups)}
    if train_rows and test_rows and len({data.labels[index] for index in train_rows}) == 2:
        trial = _fit([data.rows[index] for index in train_rows], [data.labels[index] for index in train_rows],
                     [data.weights[index] for index in train_rows])
        metrics["holdout"] = _evaluate(trial, [data.rows[index] for index in test_rows],
                                       [data.labels[index] for index in test_rows],
                                       [data.weights[index] for index in test_rows],
                                       [data.lexical[index] for index in test_rows])
    model.weights = _fit(data.rows, data.labels, data.weights)
    metrics["training"] = _evaluate(model.weights, data.rows, data.labels, data.weights, data.lexical)
    metrics["origins"] = dict(Counter(data.origins))
    model.metrics = metrics
    model.blend = _blend(metrics, len(data.rows))
    model.version = hashlib.sha256(json.dumps(model.weights, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    model.vocabulary = _vocabulary(model.weights, data.words)
    _save_model(data_dir, model)
    return model


def _blend(metrics: Mapping[str, Any], samples: int) -> float:
    """Вес модели в итоговой оценке: ноль, пока на отложенных анализах она не лучше эвристики."""
    holdout = metrics.get("holdout")
    if not isinstance(holdout, dict):
        return 0.0
    model_auc, baseline = holdout.get("auc"), holdout.get("baseline_auc")
    if not isinstance(model_auc, (int, float)) or model_auc < 0.6:
        return 0.0
    if isinstance(baseline, (int, float)) and model_auc < baseline - 0.01:
        return 0.0
    return round(min(0.5, 0.15 + samples / 20_000), 3)


def _source_stats(data: _Dataset) -> dict[str, dict[str, Any]]:
    totals: dict[str, list[float]] = {}
    for source, label in zip(data.sources, data.labels, strict=True):
        if not source:
            continue
        entry = totals.setdefault(source, [0, 0])
        entry[0] += 1
        entry[1] += label
    return {source: {"items": int(items), "relevant": int(relevant),
                     "precision": round((relevant + 1) / (items + 2), 4)}
            for source, (items, relevant) in sorted(totals.items())}


def _vocabulary(weights: Mapping[str, float], words: Counter[tuple[int, str]]) -> dict[str, str]:
    """Самые частые слова сильнейших хеш-признаков — чтобы владелец видел, чему модель научилась."""
    strongest = sorted((name for name in weights if name.startswith("w")), key=lambda name: -abs(weights[name]))[:60]
    by_bucket: dict[int, Counter[str]] = {}
    for (bucket, word), count in words.items():
        by_bucket.setdefault(bucket, Counter())[word] += count
    vocabulary = {}
    for name in strongest:
        options = by_bucket.get(int(name[1:]))
        if options:
            vocabulary[name] = options.most_common(1)[0][0]
    return vocabulary


def _save_model(data_dir: Path, model: RelevanceModel) -> None:
    directory = learning_directory(data_dir)
    _write_json(directory / MODEL_FILE, model.to_json())
    entry = {"trained_at": model.trained_at, "version": model.version, "samples": model.samples,
             "runs": model.runs, "blend": model.blend, "metrics": {
                 key: model.metrics.get(key) for key in ("state", "holdout", "training")}}
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / HISTORY_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def history(data_dir: Path, limit: int = 50) -> list[dict[str, Any]]:
    try:
        lines = (learning_directory(data_dir) / HISTORY_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def reset(data_dir: Path, *, keep_feedback: bool = True) -> None:
    """Забыть обученную модель и примеры (отметки владельца по умолчанию остаются)."""
    directory = learning_directory(data_dir)
    for path in (directory / MODEL_FILE, directory / HISTORY_FILE):
        path.unlink(missing_ok=True)
    samples = directory / SAMPLES
    if samples.is_dir():
        for path in samples.glob("*.json"):
            path.unlink(missing_ok=True)
    if not keep_feedback:
        (directory / FEEDBACK_FILE).unlink(missing_ok=True)


def summary(data_dir: Path) -> dict[str, Any]:
    """Состояние обучения для панели владельца."""
    model = RelevanceModel.load(data_dir) if load_settings(data_dir)["enabled"] else RelevanceModel()
    try:
        stored = RelevanceModel.from_json(json.loads((learning_directory(data_dir) / MODEL_FILE)
                                                     .read_text(encoding="utf-8")))
    except (OSError, ValueError):
        stored = RelevanceModel()
    feedback = load_feedback(data_dir)
    words = sorted(((stored.weights[name], word) for name, word in stored.vocabulary.items()
                    if name in stored.weights), key=lambda item: item[0])
    return {"settings": load_settings(data_dir), "in_use": model.ready, "version": stored.version,
            "trained_at": stored.trained_at, "samples": stored.samples, "runs": stored.runs,
            "blend": stored.blend, "metrics": stored.metrics, "sources": stored.sources,
            "source_weights": stored.source_weights(),
            "noise_words": [word for weight, word in words[:15] if weight < 0],
            "topic_words": [word for weight, word in reversed(words[-15:]) if weight > 0],
            "sampled_runs": len(sampled_runs(data_dir)),
            "feedback": sum(len(items) for items in feedback.values()),
            "history": history(data_dir, 30)}


class Trainer:
    """Одно переобучение за раз, в фоне; повторный запрос во время обучения ставит ещё одно."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._lock = Lock()
        self._running = False
        self._again = False
        self.last_error: str | None = None
        self.last_model: RelevanceModel | None = None

    @property
    def running(self) -> bool:
        return self._running

    def request(self, start: Callable[[Callable[[], None]], Any] | None = None) -> bool:
        with self._lock:
            if self._running:
                self._again = True
                return False
            self._running = True
        runner = start or (lambda target: target())
        runner(self._loop)
        return True

    def _loop(self) -> None:
        while True:
            try:
                self.last_model = train(self.data_dir)
                self.last_error = None
            except Exception as error:  # Обучение не должно ронять сервис анализа.
                self.last_error = type(error).__name__
            with self._lock:
                if not self._again:
                    self._running = False
                    return
                self._again = False


def discovery_decisions(relevance: Iterable[object]) -> dict[str, tuple[float, str]]:
    """Решения этапа обнаружения по документам: study_id → (оценка, решение)."""
    decisions = {}
    for entry in relevance:
        if not isinstance(entry, dict):
            continue
        study, score, decision = entry.get("study_id"), entry.get("score"), entry.get("decision")
        if (isinstance(study, str) and isinstance(score, (int, float)) and math.isfinite(score)
                and decision in {"retained", "below_semantic_threshold", "closer_to_excluded_scope"}):
            decisions[study] = (float(score), str(decision))
    return decisions
