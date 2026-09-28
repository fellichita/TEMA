"""Консоль для проверки backend: python -m app.backend --help."""

import argparse
import json
import sys
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from pydantic import ValidationError

from app.backend.config import BackendSettings
from app.backend.contracts import SOURCE_NAMES, SearchRequest
from app.backend.errors import BackendError
from app.backend.history import HistoryRequest
from app.backend.history_progress import ProgressPrinter, render_history_progress
from app.backend.service import Backend
from app.backend.validation import collection_requests, validate_pagination, validate_search_query

_INPUT_ERROR_CODES = frozenset({"invalid_pagination", "invalid_query", "invalid_source"})


def _validation_message(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_input=False, include_context=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "параметры запроса"
        # Our model validators have fixed messages, without request values.
        message = item["msg"].removeprefix("Value error, ")
        details.append(f"{location}: {message}")
    return "; ".join(details)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backend Trendanalyser: сбор и локальное хранение публикаций")
    parser.add_argument("--data-dir", type=Path, help="Отдельный локальный каталог данных Trendanalyser Pilot main2")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Загрузить публикации или патенты")
    collect.add_argument("topic")
    collect.add_argument("--limit", type=int, default=200, help="Лимит исходных записей (1–10000)")
    collect.add_argument("--from-date", help="Начало периода YYYY-MM-DD")
    collect.add_argument("--until-date", help="Конец периода YYYY-MM-DD, по умолчанию сегодня")
    collect.add_argument("--primary-topic-ids", nargs="+", default=[],
                         help="Точные ID основных тем OpenAlex; заменяют текстовый поиск")
    selection = collect.add_mutually_exclusive_group()
    selection.add_argument("--source", choices=SOURCE_NAMES, help="Один источник; по умолчанию crossref")
    selection.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, help="Несколько источников; лимит для каждого")
    commands.add_parser("sources", help="Показать источники и наличие настройки доступа без сетевого запроса")
    commands.add_parser("jobs", help="Показать историю сборов")
    history = commands.add_parser("collect-history", help="Собрать документы по календарным периодам")
    history.add_argument("topic")
    history.add_argument("--from-date", required=True)
    history.add_argument("--until-date", help="Дата отсечения; по умолчанию сегодня UTC")
    history.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=["crossref", "openalex"])
    history.add_argument("--period", choices=["year", "month"], default="month")
    history.add_argument("--limit-per-period", type=int, default=1000)
    history.add_argument("--no-auto-split", action="store_true", help="Не дробить переполненные периоды")
    history.add_argument("--max-periods", type=int, default=1200, help="Бюджет всех периодов, включая разделённые родительские")
    history.add_argument("--primary-topic-ids", nargs="+", default=[],
                         help="Точные ID основных тем OpenAlex; заменяют текстовый поиск")
    resume = commands.add_parser("resume-history", help="Продолжить сохранённый исторический сбор")
    resume.add_argument("history_id")
    resume.add_argument("--retry-incomplete", action="store_true", help="Повторить также периоды с неполной успешной выдачей")
    report = commands.add_parser("history", help="Показать прогресс и полноту периодов без сети")
    report.add_argument("history_id")
    report.add_argument("--format", choices=["json", "text", "progress-json"], default="json",
                        help="Полный JSON, читаемый отчёт или компактный JSON прогресса")
    commands.add_parser("histories", help="Список исторических сборов без сети")
    documents = commands.add_parser("documents", help="Прочитать сохранённые документы")
    scope = documents.add_mutually_exclusive_group()
    scope.add_argument("--job-id")
    scope.add_argument("--history-id")
    documents.add_argument("--query", help="Буквальный поиск подстроки в названии и аннотации")
    documents.add_argument("--limit", type=int, default=20)
    documents.add_argument("--offset", type=int, default=0)
    versions = commands.add_parser("versions", help="Показать версии документа из разных источников")
    versions.add_argument("document_key")
    versions.add_argument("--limit", type=int, default=20)
    versions.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "sources":
            print(json.dumps(Backend.sources(), ensure_ascii=False, indent=2))
            return 0
        settings = BackendSettings(data_dir=args.data_dir) if args.data_dir else BackendSettings()
        # Проверить запрос прежде, чем создавать каталог и базу.
        collect_request = None
        history_request = None
        if args.command == "collect":
            selected_sources = args.sources or [args.source or "crossref"]
            values = {"topic": args.topic, "max_results": args.limit,
                      "source": selected_sources[0], "primary_topic_ids": args.primary_topic_ids}
            if args.from_date:
                values["from_date"] = args.from_date
            if args.until_date:
                values["until_date"] = args.until_date
            collect_request = SearchRequest.model_validate(values)
            collection_requests(collect_request, selected_sources)
        elif args.command == "collect-history":
            values = dict(topic=args.topic, from_date=args.from_date, sources=args.sources,
                          period=args.period, max_results_per_period=args.limit_per_period,
                          auto_split=not args.no_auto_split, max_periods=args.max_periods,
                          primary_topic_ids=args.primary_topic_ids)
            if args.until_date:
                values["until_date"] = args.until_date
            history_request = HistoryRequest.model_validate(values)
        elif args.command in {"documents", "versions"}:
            validate_pagination(args.limit, args.offset)
            if args.command == "documents":
                validate_search_query(args.query)
        with Backend(settings) as backend:
            if args.command in {"collect-history", "resume-history"}:
                run_id = (backend.submit_history(history_request) if args.command == "collect-history" else
                          backend.resume_history(args.history_id, retry_incomplete=args.retry_incomplete))
                print(f"history: {run_id}", file=sys.stderr, flush=True)
                try:
                    progress_printer = ProgressPrinter(sys.stderr)
                    while True:
                        try:
                            report = backend.wait_history(run_id, timeout=1)
                            break
                        except FutureTimeout:
                            progress_printer.update(backend.get_history(run_id))
                except KeyboardInterrupt:
                    backend.cancel_history(run_id)
                    progress_printer.update(backend.wait_history(run_id))
                    print(f'Продолжить: python -m app.backend --data-dir "{settings.data_dir}" resume-history {run_id}', file=sys.stderr)
                    return 130
                progress_printer.update(report)
                print(report.model_dump_json(indent=2))
                return 0 if report.coverage_complete else 1
            if args.command == "collect":
                assert collect_request is not None
                job_ids = backend.submit_collections(collect_request, selected_sources)
                for source, job_id in zip(selected_sources, job_ids, strict=True):
                    print(f"{source}: {job_id}", file=sys.stderr)
                try:
                    jobs = [backend.wait(job_id) for job_id in job_ids]
                except KeyboardInterrupt:
                    for job_id in job_ids:
                        backend.cancel(job_id)
                    for job_id in job_ids:
                        backend.wait(job_id)
                    return 130
                results = [job.model_dump(mode="json") | {"coverage_complete": job.coverage_complete} for job in jobs]
                succeeded = all(job.state == "succeeded" for job in jobs)
                result = results[0] if len(results) == 1 else {"all_succeeded": succeeded, "jobs": results}
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if succeeded else 1
            if args.command == "history":
                if args.format == "text":
                    print(render_history_progress(backend.get_history_progress(args.history_id)))
                    return 0
                result = (backend.get_history_progress(args.history_id) if args.format == "progress-json" else
                          backend.get_history(args.history_id)).model_dump(mode="json")
            elif args.command == "histories":
                result = backend.list_history()
            elif args.command == "jobs":
                result = [job.model_dump(mode="json") | {"coverage_complete": job.coverage_complete}
                          for job in backend.list_jobs()]
            elif args.command == "versions":
                result = backend.list_document_versions(args.document_key, limit=args.limit,
                                                        offset=args.offset).model_dump(mode="json")
            else:
                result = backend.list_documents(job_id=args.job_id, query=args.query,
                                                limit=args.limit, offset=args.offset, history_id=args.history_id).model_dump(mode="json")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ValidationError as error:
        print(f"Некорректные параметры: {_validation_message(error)}", file=sys.stderr)
        return 2
    except BackendError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return 2 if error.code in _INPUT_ERROR_CODES else 1
    except OSError:
        print("Не удалось открыть локальное хранилище. Проверьте доступ и свободное место.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Русские сообщения должны пережить конвейер при любой кодовой странице консоли.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
