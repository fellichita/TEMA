"""Explain empty card fields in saved results without changing or refitting ML.

Run from the repository root:
python -m tools.diagnose_card_blockers --output-dir build/diagnostics/card-blockers
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from app.ml.corpus import read_snapshot
from tools.audit_ml_result import BUCKETS, audit_result


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ("photonic", "storage/ml-release/photonic-release.json", "storage/ml-validation/corpus-portable.json"),
    ("quantum", "storage/ml-release/quantum-release.json", "storage/validation-quantum/corpus.json"),
    ("dna", "storage/ml-release/dna-release.json", "storage/validation-dna/corpus.json"),
)
FIELDS = {"problem": "Проблема", "advantage": "Преимущество", "example": "Кейс", "sources": "Источник"}
RULES = {
    "not_a_problem_statement": "Не распознано утверждение о проблеме либо фрагмент имеет обзорную модальность",
    "benefit_not_reported_as_a_result_or_property": "Обзор, перспектива, предложение или потребность вместо заявленного преимущества",
    "benefit_not_specified": "Не распознан конкретный атрибут преимущества",
    "desired_benefit_not_reported": "Пожелание или цель вместо заявленного результата",
    "benefit_not_linked_to_photonic_implementation": "Не установлена связь преимущества с фотонной реализацией",
    "research_progress_instead_of_specific_benefit": "Исследовательский прогресс вместо конкретного преимущества",
}


def diagnose_result(label, result):
    """Classify saved final fields; rejections of successfully filled fields do not block them."""
    rows = []
    seen = set()
    for bucket in BUCKETS:
        for rank, group in enumerate(result[bucket], 1):
            if group["id"] in seen:
                raise ValueError(f"Повтор группы в выдаче {label}: {group['id']}")
            seen.add(group["id"])
            if bucket == "candidates":
                continue
            sources = {s["id"]: s for s in group["sources"]}
            fields = {}
            for field in FIELDS:
                if field == "sources":
                    fields[field] = {"empty": not bool(sources),
                                     "state": "filled" if sources else "no_source",
                                     "source_ids": list(sources), "guard_rejections": [],
                                     "rule_counts": {}, "guard_applies": False}
                    continue
                quote = group["card"][field]
                rejected = group["rejected_quotes"][field]
                state = "filled" if quote else "guard_rejected" if rejected else "no_candidate"
                # Keep actual saved final reasons. They are not all intermediate
                # matching predicates; quote_assessment may replace an earlier reason.
                relevant_rejections = []
                if not quote:
                    for rejected_quote in rejected:
                        source = sources[rejected_quote["study_id"]]
                        relevant_rejections.append({**deepcopy(rejected_quote),
                                                    "source_title": source["title"], "url": source["url"]})
                fields[field] = {
                    "empty": quote is None, "state": state,
                    "modality": group["evidence_annotations"][field]["modality"],
                    "chosen_quote": deepcopy(quote),
                    "guard_rejections": relevant_rejections,
                    "rule_counts": dict(sorted(Counter(q["reason"] for q in relevant_rejections).items())),
                    "saved_rejected_count": len(rejected),
                    "rejections_of_filled_field_excluded_from_totals": len(rejected) if quote else 0,
                }
            rows.append({"direction": label, "id": group["id"], "title": group["title"],
                         "bucket": bucket, "rank_in_bucket": rank, "study_count": group["study_count"],
                         "selected_source_count": len(sources), "selected_source_ids": list(sources),
                         "growth_data_comparable": result["growth_data_comparable"],
                         "selection_reasons": list(group["selection"]["reasons"]),
                         "empty_fields": [name for name, info in fields.items() if info["empty"]],
                         "fields": fields})
    return rows


def summarize(rows):
    identities = [(row["direction"], row["id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Группа повторена в диагностике.")
    by_field = []
    by_rule = {}
    for field, name in FIELDS.items():
        empty = [r for r in rows if r["fields"][field]["empty"]]
        by_field.append({"field": field, "name": name, "groups_checked": len(rows), "empty": len(empty),
                         "no_candidate_or_source": sum(r["fields"][field]["state"] in {"no_candidate", "no_source"} for r in empty),
                         "guard_rejected": sum(r["fields"][field]["state"] == "guard_rejected" for r in empty),
                         "rejected_fragments": sum(len(r["fields"][field]["guard_rejections"]) for r in empty)})
        for row in empty:
            for rule, count in row["fields"][field]["rule_counts"].items():
                item = by_rule.setdefault(rule, {"rule": rule, "description": RULES.get(rule, rule),
                                               "rejected_fragments": 0, "group_keys": set(), "field_keys": set(),
                                               "fields": set()})
                item["rejected_fragments"] += count
                item["group_keys"].add((row["direction"], row["id"]))
                item["field_keys"].add((row["direction"], row["id"], field))
                item["fields"].add(field)
    rules = []
    for item in sorted(by_rule.values(), key=lambda item: (-item["rejected_fragments"], item["rule"])):
        rules.append({"rule": item["rule"], "description": item["description"],
                      "rejected_fragments": item["rejected_fragments"],
                      "affected_groups": len(item["group_keys"]), "affected_fields": len(item["field_keys"]),
                      "fields": sorted(item["fields"]),
                      "group_keys": [list(key) for key in sorted(item["group_keys"])]})
    return {"groups_outside_top": len(rows),
            "groups_with_empty_fields": sum(bool(r["empty_fields"]) for r in rows),
            "groups_with_all_four_fields": sum(not r["empty_fields"] for r in rows),
            "empty_field_count": sum(item["empty"] for item in by_field),
            "example_research_reference_count": sum(r["fields"]["example"]["modality"] == "research_reference" for r in rows),
            "selection_reason_counts": dict(sorted(Counter(reason for r in rows for reason in r["selection_reasons"]).items())),
            "field_table": by_field, "rule_table": rules}


def markdown_report(report):
    summary = report["summary"]
    source_counts = [row["selected_source_count"] for row in report["groups"]]
    no_candidates = Counter(field for row in report["groups"] for field, info in row["fields"].items()
                            if info["state"] == "no_candidate")
    no_candidate_text = "; ".join(f"{FIELDS[field]} — {count}" for field, count in no_candidates.items()) or "нет"
    lines = ["# Почему поля карточек пусты", "",
             f"Проверены {summary['groups_outside_top']} групп вне основного TOP. "
             f"Пустых полей: {summary['empty_field_count']} у {summary['groups_with_empty_fields']} групп. "
             f"У {summary['groups_with_all_four_fields']} групп все четыре поля заполнены.", "",
             "## Отказы по полям", "",
             "| Поле | Проверено групп | Пусто | Экстрактор не передал кандидата / нет источника | Кандидаты отклонены guard | Отклонённых фрагментов в пустых полях |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in summary["field_table"]:
        lines.append(f"| {item['name']} | {item['groups_checked']} | {item['empty']} | "
                     f"{item['no_candidate_or_source']} | {item['guard_rejected']} | {item['rejected_fragments']} |")
    lines += ["", "## Отказы по правилам", "",
              "Только правила для **пустых итоговых полей**. Отклонения, после которых нашлась замена и поле заполнено, сюда не входят.", "",
              "| Итоговое правило | Смысл | Фрагментов | Пустых полей | Групп |",
              "|---|---|---:|---:|---:|"]
    for item in summary["rule_table"]:
        lines.append(f"| `{item['rule']}` | {item['description']} | {item['rejected_fragments']} | "
                     f"{item['affected_fields']} | {item['affected_groups']} |")
    lines += ["", "Числа в колонках «групп» и «полей» по правилам не складываются: у одного пустого поля могут быть разные причины отклонения разных предложений. "
              "Фрагмент считается по записи (группа, поле, исследование, текст), а не по уникальному тексту во всём корпусе. "
              "`reason` — сохранённая итоговая причина, не полный список промежуточных срабатываний.", "",
              "## Границы вывода", "",
              "Поиск идёт по аннотациям первых 12 отобранных источников группы "
              f"(в этих данных источников от {min(source_counts, default=0)} до {max(source_counts, default=0)}). "
              "Исходный экстрактор требует лексический сигнал, не менее 5 слов и не более 900 символов в предложении; "
              "для преимущества до guard также действует `_supported_advantage`. "
              "Поэтому «не передал кандидата» не означает, что во всём корпусе нет подходящего предложения. "
              f"Случаи без кандидата в текущем наборе: {no_candidate_text}. "
              "Для новых данных причины внутри исходного экстрактора этим журналом не детализируются.", "",
              "Источник — отдельный список `sources`, он не проходит guard цитат. Его заполненность не означает, что найден текст для каждого поля. "
              f"Из заполненных кейсов {summary['example_research_reference_count']} примеров — только `research_reference`; "
              "это библиографическая ссылка, не подтверждение реализации.", "",
              "## Почему это ещё не объясняет весь пустой TOP", "",
              "Сохранённые причины отбора (могут пересекаться): " + "; ".join(
                  f"`{key}` — {value}" for key, value in summary["selection_reason_counts"].items()) + ".", "",
              "Отбор сначала завершает проверку при `off_direction`, затем при доле >8%. "
              f"Групп с пустыми полями {summary['groups_with_empty_fields']}, "
              f"а `incomplete_supported_card` записан у {summary['selection_reason_counts'].get('incomplete_supported_card', 0)}. "
              f"Библиографических примеров {summary['example_research_reference_count']}, "
              f"а `example_is_bibliographic_reference` записан у {summary['selection_reason_counts'].get('example_is_bibliographic_reference', 0)}. "
              "Снятие отказов по полям само по себе не снимает ограничения покрытия и роста; пересчёт или ослабление условий здесь не выполнялись.", "",
              "## Полная детализация и воспроизведение", "",
              f"[report.json](report.json) содержит запись на каждую из {summary['groups_outside_top']} групп: четыре поля, их состояние, выбранный фрагмент, "
              "отклонённые кандидаты, точные правила, ссылки на источники и причины размещения вне TOP. "
              "Для заполненных полей причины отклонения альтернатив не считаются блокировкой.", "",
              "Перед диагностикой выполнен структурный replay результата на исходном корпусе: восстановлены выбранные версии документов, "
              "повторён отбор цитат с исходными термами группы, сверены карточки, аннотации, причины отклонения и fingerprints. "
              "При расхождении скрипт завершается ошибкой и не публикует таблицы. Это проверка работы текущих правил, не новая экспертная оценка их правильности.", "",
              "```sh", ".venv/bin/python -B -m tools.diagnose_card_blockers --output-dir /tmp/card-blockers-new-run", "```", "",
              "Каталог вывода должен быть новым. Веса, модель, приложение и исходные данные скрипт не меняет.", ""]
    return "\n".join(lines)


def build_report(inputs):
    rows, checked = [], []
    labels = set()
    for label, result_path, corpus_path in inputs:
        if label in labels:
            raise ValueError(f"Повтор метки направления: {label}")
        labels.add(label)
        result_path, corpus_path = Path(result_path), Path(corpus_path)
        result_bytes = result_path.read_bytes()
        result = json.loads(result_bytes)
        corpus = read_snapshot(corpus_path)
        audit = audit_result(corpus, result)
        if not audit["ok"]:
            raise ValueError(f"Выдача {label} не прошла replay: {audit['errors']}")
        rows.extend(diagnose_result(label, result))
        checked.append({"direction": label, "result_path": str(result_path), "corpus_path": str(corpus_path),
                        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                        "corpus_sha256": corpus["provenance"]["snapshot_sha256"],
                        "result_fingerprint": result["fingerprint"], "structural_replay_ok": True,
                        "quote_replay_count": audit["checks"].get("quote_selection_replay", 0),
                        "buckets": {bucket: len(result[bucket]) for bucket in BUCKETS}})
    return {"schema_version": 1, "kind": "card_field_blocker_diagnostics", "inputs": checked,
            "summary": summarize(rows), "groups": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError("Каталог результата уже существует. Выберите новый каталог.")
        report = build_report([(label, ROOT / result, ROOT / corpus) for label, result, corpus in DEFAULT_INPUTS])
        if report["summary"]["groups_outside_top"] != 53:
            raise ValueError("В текущих выдачах уже не 53 группы вне TOP; сначала уточните набор для этого аудита.")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + "\n")
    print(f"Проверено 53 группы; таблицы и детализация: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
