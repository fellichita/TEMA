"""Командная строка этапа 1.

Обучение и отчёт:
    python -m app.signal_model train --dataset data/signal_model/dataset.xlsx
Оценка нового файла в той же схеме (например, закрытой выборки):
    python -m app.signal_model predict --dataset путь.xlsx --out predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/signal_model/dataset.xlsx"
DEFAULT_NEGATIVES = ROOT / "data/signal_model/negatives.json"
DEFAULT_MODEL = Path(__file__).with_name("model.json")
DEFAULT_REPORT = ROOT / "docs/methodology/signal-model-report.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.signal_model")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="обучить модель и собрать отчёт")
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    predict = commands.add_parser("predict", help="оценить технологии из xlsx")
    predict.add_argument("--dataset", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from app.signal_model.dataset import read_negatives, read_xlsx
    from app.signal_model.model import SignalModel, fit

    if args.command == "train":
        from app.signal_model.report import build_report

        data = read_xlsx(args.dataset) + read_negatives(args.negatives)
        model = fit(data)
        model.save(args.model)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(build_report(data, model), encoding="utf-8")
        print(f"Модель: {args.model}\nОтчёт: {args.report}")
        return 0
    model = SignalModel.load(args.model)
    technologies = read_xlsx(args.dataset, default_label=None)
    predictions = model.predict(technologies)
    rows = [{"Технология": item.title, "Вероятность слабого сигнала": f"{item.probability:.4f}",
             "Решение": "слабый сигнал" if item.is_signal else "не слабый сигнал",
             "Главные признаки": "; ".join(f"{part.label}: {part.value} ({part.weight:+.2f})"
                                           for part in item.contributions[:3])}
            for item in predictions]
    labelled = [(item, technology.label) for item, technology in zip(predictions, technologies, strict=True)
                if technology.label is not None]
    if labelled:
        correct = sum(item.is_signal == bool(label) for item, label in labelled)
        print(f"Точность на размеченных строках: {correct}/{len(labelled)} = {correct / len(labelled):.1%}")
    output = args.out.open("w", encoding="utf-8-sig", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["Технология"])
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
