"""UI labels and validation independent of Tk and storage."""

from datetime import date, datetime, timezone

SOURCES = {"crossref": "Crossref", "openalex": "OpenAlex", "epo": "EPO (патенты)"}
STATES = {"queued": "В очереди", "running": "Загрузка", "failed": "Ошибка",
          "cancelled": "Отменено", "interrupted": "Прервано", "succeeded": "Завершено"}
ACTIVE = {"queued", "running"}


def publication_date(doc):
    if doc.date_precision == "day" and doc.publication_date:
        return doc.publication_date.strftime("%d.%m.%Y")
    if doc.date_precision == "month":
        return f"{doc.publication_month:02d}.{doc.publication_year} (месяц)"
    if doc.date_precision == "year":
        return f"{doc.publication_year} (год)"
    return "Дата не указана"


def job_state(job):
    if job.state == "succeeded":
        if empty_collection(job):
            return "Завершено — ничего не найдено"
        return "Завершено — полная выдача" if job.coverage_complete else "Завершено — неполная выборка"
    return STATES.get(job.state, job.state)


def error_message(error):
    if type(error).__module__ == "app.backend.errors":
        return f"{error.message} [{error.code}]"
    if isinstance(error, ImportError):
        return "Не установлены зависимости. Установите их из requirements/dev.lock."
    if isinstance(error, OSError):
        return "Ошибка локального хранилища. Проверьте доступ к папке и свободное место."
    if type(error).__name__ == "ValidationError":
        return "Некорректные параметры. Проверьте тему, источники, даты и лимит."
    return "Не удалось выполнить операцию. Повторите попытку."


def collection_values(topic, start, end, limit, sources, today=None):
    today = today or datetime.now(timezone.utc).date()
    errors = {}
    topic = topic.strip()
    if not 2 <= len(topic) <= 500 or any(ord(char) < 32 for char in topic):
        errors["topic"] = "Тема: 2–500 символов, без переносов строк."
    if not sources or any(source not in SOURCES for source in sources):
        errors["sources"] = "Выберите хотя бы один источник."
    if "epo" in sources and ('"' in topic or "\\" in topic):
        errors["topic"] = "EPO не поддерживает кавычки и обратную косую черту."
    dates = {}
    for name, raw in (("start", start), ("end", end)):
        raw = raw.strip()
        if not raw:
            dates[name] = None if name == "start" else today
            continue
        try:
            parsed = date.fromisoformat(raw)
            if parsed.isoformat() != raw:
                raise ValueError()
            dates[name] = parsed
        except ValueError:
            errors[name] = "Введите дату в формате ГГГГ-ММ-ДД."
    start_date, end_date = dates.get("start"), dates.get("end")
    if end_date is not None and end_date > today:
        errors["end"] = "Дата окончания не может быть в будущем."
    if start_date is not None and end_date is not None and start_date > end_date:
        errors["start"] = "Начало периода должно быть не позже окончания."
    try:
        count = int(limit.strip())
        if not 1 <= count <= 10_000:
            raise ValueError()
    except ValueError:
        errors["limit"] = "Введите целое число от 1 до 10 000."
        count = 200
    return {"topic": topic, "from_date": dates.get("start"),
            "until_date": dates.get("end", today), "max_results": count}, errors


def document_detail(snapshot):
    """Render one immutable snapshot without mixing fields from other versions."""
    doc = snapshot.document
    return "\n\n".join([
        doc.title, f"Авторы: {', '.join(doc.authors) or 'не указаны'}",
        f"Публикация: {publication_date(doc)}\nТип: {doc.document_type}\nЯзык: {doc.language or 'не указан'}",
        f"Цитирования: {doc.citation_count if doc.citation_count is not None else 'не указаны'}",
        doc.abstract or "Аннотация отсутствует в данных источника.",
        f"Источник версии: {SOURCES.get(doc.source, doc.source)}\nID источника: {doc.source_id}"
        f"\nDOI: {doc.doi or 'не указан'}\nПатент: {doc.patent_publication or '—'}"
        f"\nСемейство патента: {getattr(doc, 'patent_family_id', None) or '—'}",
        f"Получено: {doc.fetched_at.strftime('%d.%m.%Y %H:%M UTC')}"
        "\nВремя получения сохранённой версии; не дата обновления у источника.",
        f"Версия: {snapshot.revision_id}", doc.url,
    ])


def history_values(topic, start, end, limit, sources, period, auto_split, budget, today=None):
    values, errors = collection_values(topic, start, end, limit, sources, today=today)
    if not start.strip():
        errors['start'] = 'Укажите начало исторического периода.'
    if period not in ('month', 'year'):
        errors['period'] = 'Выберите месяцы или годы.'
    try:
        maximum = int(budget.strip())
        if not 1 <= maximum <= 10_000:
            raise ValueError()
    except ValueError:
        maximum = 1200
        errors['budget'] = 'Введите целое число от 1 до 10 000.'
    beginning, ending = values['from_date'], values['until_date']
    if beginning and ending and beginning <= ending:
        if beginning.year < 1000 or ending.year - beginning.year > 100:
            errors['start'] = 'Допустим период до 100 лет, начиная с 1000 года.'
        periods = ((ending.year - beginning.year) * 12 + ending.month - beginning.month + 1
                   if period == 'month' else ending.year - beginning.year + 1)
        if periods * len(sources) > min(1200, maximum):
            errors['budget'] = 'Начальный план превышает бюджет или предел 1200 периодов × источников.'
    return {'topic': values['topic'], 'from_date': beginning, 'until_date': ending,
            'sources': sources, 'period': period, 'max_results_per_period': values['max_results'],
            'auto_split': auto_split, 'max_periods': maximum}, errors


def empty_collection(job):
    return job.coverage_complete and getattr(job, "scanned", None) == 0


def empty_collection_hint(job):
    request = job.request
    source = SOURCES.get(request.source, request.source)
    start = str(request.from_date) if request.from_date else "без начальной даты"
    return (f'{source} не нашёл записей по запросу «{request.topic}» '
            f'за период {start} — {request.until_date}. '
            'Измените формулировку темы или расширьте период в «Разовый сбор». '
            'Crossref и OpenAlex ищут научные публикации, EPO — патенты; поиска по новостным сайтам здесь нет.')
