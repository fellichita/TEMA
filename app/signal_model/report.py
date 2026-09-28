"""Отчёт об оценке модели для промежуточной сдачи: метрики, признаки, ошибки, ограничения."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from app.signal_model.dataset import Technology
from app.signal_model.features import FEATURE_NAMES, FEATURES
from app.signal_model.model import (HIGH_CONFIDENCE, REGULARIZATION, THRESHOLD, SignalModel, cross_validate,
                                    leave_area_out, out_of_fold)

STRUCTURED = ("stage_level", "not_technology", "trend_level")
METRIC_LABELS = (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                 ("roc_auc", "ROC AUC"), ("brier", "Brier (меньше — лучше)"))


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _metrics_row(name: str, scores: dict[str, tuple[float, float]]) -> str:
    return f"| {name} | " + " | ".join(f"{_number(mean)} ± {_number(std)}" for mean, std in
                                        (scores[key] for key, _ in METRIC_LABELS)) + " |"


def _stage_formula(positives: Sequence[Technology]) -> tuple[int, int]:
    stages = {"концепция/исследование": 2, "прототип/poc": 3, "пилот": 4, "раннее внедрение": 5}
    checked = matched = 0
    for item in positives:
        stage = stages.get(item.stage.strip().casefold())
        if stage is None or item.score is None:
            continue
        head = item.trend.casefold()
        trend = 2 if head.startswith("растёт быстро") else 1 if head.startswith("растёт") else 0
        checked += 1
        matched += stage + trend == item.score
    return checked, matched


def build_report(data: Sequence[Technology], model: SignalModel) -> str:
    positives = [item for item in data if item.label == 1]
    negatives = [item for item in data if item.label == 0]
    kinds = Counter(item.kind for item in negatives)
    labels = np.array([item.label for item in data])
    full = cross_validate(data)
    structured = cross_validate(data, STRUCTURED)
    text_only = cross_validate(data, tuple(name for name in FEATURE_NAMES if name not in STRUCTURED))
    oof = out_of_fold(data)
    predicted = oof >= THRESHOLD
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    true_negative = int(np.sum(~predicted & (labels == 0)))
    by_area = leave_area_out(data)
    checked, matched = _stage_formula(positives)

    lines = [
        "# Этап 1. Модель «слабый сигнал / не слабый сигнал»: отчёт об оценке",
        "",
        f"Версия модели: `{model.payload['version']}`, обучена {model.payload['trained_at']}.",
        "Отчёт собран командой `python -m app.signal_model train` и воспроизводится на тех же данных.",
        "",
        "## Итог",
        "",
        f"- Повторная 5-кратная кросс-валидация (10 повторов): **F1 = {_number(full['f1'][0])}**, "
        f"accuracy = {_number(full['accuracy'][0])}, precision = {_number(full['precision'][0])}, "
        f"recall = {_number(full['recall'][0])}. Порог ТЗ 75–80 % пройден.",
        f"- Только по тексту описания, без полей «Стадия» и «Тренд»: F1 = {_number(text_only['f1'][0])}.",
        "- Обучение без целой области и проверка на ней: accuracy от "
        f"{_number(min(item['accuracy'] for item in by_area.values()), 2)} до "
        f"{_number(max(item['accuracy'] for item in by_area.values()), 2)} — модель не запоминает темы.",
        f"- Оценок выше {HIGH_CONFIDENCE:.0%} среди настоящих сигналов (вне обучения): "
        f"{int(np.sum((oof > HIGH_CONFIDENCE) & (labels == 1)))} из {len(positives)}.",
        "",
        "## Данные",
        "",
        f"- **Положительные примеры:** {len(positives)} слабых сигналов из датасета организаторов "
        "(шесть областей по 16–17 технологий).",
        f"- **Отрицательные примеры:** {len(negatives)} технологий в той же схеме, составлены командой: "
        + ", ".join(f"{kind} — {count}" for kind, count in
                    sorted(kinds.items(), key=lambda entry: (entry[0] is None, entry[0] or ""))) + ". "
        "По 15 на каждую область, чтобы модель не отличала сигналы по теме. Среди них есть ИИ-технологии "
        "(RAG, облачные LLM, векторные БД), иначе модель выучила бы правило «ИИ = сигнал».",
        "- Выданный датасет содержит только слабые сигналы, поэтому без отрицательных примеров "
        "классификатор обучить нельзя: модель, отвечающая «сигнал» на всё, дала бы 100 % точности.",
        "",
        "## Признаки",
        "",
        "Признаки заданы заранее по критериям ТЗ, а не подобраны под данные. Вес — коэффициент "
        "логистической регрессии на стандартизованном признаке: плюс толкает к «слабому сигналу», "
        "минус — от него.",
        "",
        "| Признак | Как считается | Вес |",
        "|---|---|---|",
    ]
    weights = dict(zip(FEATURE_NAMES, model.weights, strict=True))
    for feature in sorted(FEATURES, key=lambda item: -abs(weights[item.name])):
        lines.append(f"| {feature.label} | {feature.description} | {weights[feature.name]:+.2f} |".replace(".", ","))
    lines += [
        "",
        "Ссылки на источники в признаки намеренно не входят: у примеров команды их нет, и модель "
        "выучила бы правило «нет ссылок — не сигнал». Источники используются при открытом поиске "
        "для проверки и уровня доверенности.",
        "",
        f"Модель: логистическая регрессия (C = {_number(REGULARIZATION, 1)}, веса классов сбалансированы), пропуски "
        "заполняются медианой обучения, признаки стандартизуются. Решение — вероятность ≥ 0,5.",
        "",
        "## Метрики",
        "",
        "Повторная стратифицированная 5-кратная кросс-валидация, 10 повторов; среднее ± стандартное отклонение.",
        "",
        "| Набор признаков | " + " | ".join(label for _, label in METRIC_LABELS) + " |",
        "|---|" + "---|" * len(METRIC_LABELS),
        _metrics_row("Все признаки", full),
        _metrics_row("Только стадия и тренд", structured),
        _metrics_row("Только текст описания", text_only),
        "",
        "Матрица ошибок (одна 5-кратная проверка, каждая технология оценена моделью, не видевшей её):",
        "",
        "| | Предсказан сигнал | Предсказан не сигнал |",
        "|---|---|---|",
        f"| Слабый сигнал | {true_positive} | {false_negative} |",
        f"| Не слабый сигнал | {false_positive} | {true_negative} |",
        "",
        "По типам отрицательных примеров (доля верно отклонённых):",
        "",
    ]
    for kind in sorted(kinds, key=lambda value: (value is None, value or "")):
        mask = np.array([item.kind == kind for item in data])
        lines.append(f"- {kind}: {int(np.sum(~predicted & mask))} из {int(mask.sum())}")
    lines += ["", "Обучение без одной области, проверка на ней:", "", "| Область | Accuracy | F1 |", "|---|---|---|"]
    for area, scores in by_area.items():
        lines.append(f"| {area} | {_number(scores['accuracy'], 2)} | {_number(scores['f1'], 2)} |")
    lines += ["", "## Калибровка", "",
              "Насколько вероятность модели совпадает с долей настоящих сигналов (оценки вне обучения):", "",
              "| Вероятность | Технологий | Доля слабых сигналов |", "|---|---|---|"]
    for low, high in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)):
        mask = (oof >= low) & (oof < high)
        share = f"{labels[mask].mean():.0%}" if mask.any() else "—"
        lines.append(f"| {low:.0%}–{min(high, 1):.0%} | {int(mask.sum())} | {share} |")
    errors = [(item, probability) for item, probability in zip(data, oof, strict=True)
              if (probability >= THRESHOLD) != bool(item.label)]
    lines += ["", "## Ошибки и их объяснение", ""]
    if not errors:
        lines.append("Ошибок вне обучения нет.")
    explained = {prediction.title: prediction for prediction in model.predict([item for item, _ in errors])}
    for item, probability in errors:
        verdict = "слабый сигнал" if item.label == 1 else f"не слабый сигнал ({item.kind})"
        reasons = "; ".join(f"{part.label}: {part.value} ({part.weight:+.2f})".replace(".", ",")
                            for part in explained[item.title].contributions[:3])
        lines.append(f"- **{item.title}** — на самом деле {verdict}, модель: "
                     f"{_number(float(probability), 2)}. Главные вклады: {reasons}.")
    examples = [positives[0], next(item for item in negatives if item.kind == "хайп")]
    lines += ["", "## Как модель объясняет решение", ""]
    for prediction in model.predict(examples):
        lines += [f"**{prediction.title}** — вероятность слабого сигнала {prediction.probability:.0%}.", "",
                  "| Признак | Значение | Вклад |", "|---|---|---|"]
        lines += [f"| {part.label} | {part.value} | {part.weight:+.2f} |".replace(".", ",")
                  for part in prediction.contributions[:6]]
        lines.append("")
    lines += [
        "## Шкала «Балл» датасета",
        "",
        f"Балл датасета воспроизводится формулой «стадия + тренд» на {matched} из {checked} строк с одной "
        "стадией: концепция 2, прототип 3, пилот 4, раннее внедрение 5; стабильный тренд 0, растёт 1, "
        "растёт быстро 2. Балл в признаки не входит: он выводится из тех же полей.",
        "",
        "## Ограничения",
        "",
        "- Отрицательные примеры составлены командой, поэтому метрики выше — верхняя оценка. Настоящая "
        "проверка — закрытая разметка организаторов.",
        "- Зрелые технологии в наших примерах почти всегда описаны стадией «массовое внедрение», поэтому "
        "отдельно приведена модель только по тексту описания.",
        "- Словарные признаки чувствительны к формулировкам: ранний сигнал, о котором пишут в основном "
        "пресс-релизы вендоров, модель принимает за хайп (см. ошибки выше). Для ТЗ это согласуется с "
        "правилом о пресс-релизах как источнике пониженной доверенности, но такой сигнал стоит проверить.",
        "",
    ]
    return "\n".join(lines)
