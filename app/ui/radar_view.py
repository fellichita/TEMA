"""Разметка ТОПа технологий, журнала исключений и страницы-отчёта по технологии."""

from __future__ import annotations

from html import escape
import re
from typing import Mapping

from app.ui.radar_client import RadarPoint, RadarResult, RadarStatus, RadarTechnology

TRANSLATED = '<span class="ta-mt-mark" title="Машинный перевод локальной моделью">машинный перевод</span>'
GRANULAR_POLICY_VERSION = (1, 3, 0)
_POLICY_VERSION = re.compile(r"radar/([0-9]+)\.([0-9]+)\.([0-9]+)\Z")


def older_selection_policy(version: str | None) -> bool:
    """Only known older radar policies trigger the saved-result notice."""
    match = _POLICY_VERSION.fullmatch(version or "")
    return bool(match and tuple(int(part) for part in match.groups()) < GRANULAR_POLICY_VERSION)


def _percent(value: float) -> str:
    return f"{round(value * 100)}%"


def _name(technology: RadarTechnology) -> str:
    # Термин в оригинале: машинный перевод коротких названий искажает смысл.
    return technology.title


def report_key(technology: RadarTechnology, result: RadarResult) -> str | None:
    for prefix, group in (("t", result.technologies), ("x", result.excluded)):
        for index, item in enumerate(group, start=1):
            if item is technology:
                return f"{prefix}{index}"
    return None


def find_report(result: RadarResult, key: str) -> RadarTechnology | None:
    groups = {"t": result.technologies, "x": result.excluded}
    if len(key) < 2 or key[0] not in groups or not key[1:].isdigit():
        return None
    index = int(key[1:]) - 1
    group = groups[key[0]]
    return group[index] if 0 <= index < len(group) else None


def curve_svg(points: tuple[RadarPoint, ...], *, large: bool = False) -> str:
    """Столбики — сумма коэффициентов по месяцам, линия — сглаживание, пунктир — парабола."""
    if not points:
        return ""
    width, height = (720, 200) if large else (240, 56)
    top, bottom = (12, 176) if large else (4, 52)
    peak = max(max(point.weighted, point.smoothed, point.fitted or 0.0) for point in points) or 1.0
    step = width / len(points)
    bars, smooth, fitted = [], [], []
    for index, point in enumerate(points):
        bar = (bottom - top) * point.weighted / peak
        x = index * step
        # Запятая только в подсказке: в координатах SVG она ломает столбик.
        weight = f"{point.weighted:.2f}".replace(".", ",")
        bars.append(f'<rect x="{x + step * 0.15:.1f}" y="{bottom - bar:.1f}" width="{step * 0.7:.1f}" '
                    f'height="{max(bar, 0.8):.1f}" class="{"ta-curve-bar" if point.weighted else "ta-curve-empty"}">'
                    f'<title>{escape(point.month)}: работ {point.materials}, сумма коэффициентов '
                    f'{weight}</title></rect>')
        centre = x + step / 2
        smooth.append(f"{centre:.1f},{bottom - (bottom - top) * point.smoothed / peak:.1f}")
        if point.fitted is not None:
            fitted.append(f"{centre:.1f},{bottom - (bottom - top) * point.fitted / peak:.1f}")
    lines = f'<polyline class="ta-curve-smooth" points="{" ".join(smooth)}"/>'
    if fitted:
        lines += f'<polyline class="ta-curve-fit" points="{" ".join(fitted)}"/>'
    label = (f"Помесячная кривая: {points[0].month} — {points[-1].month}, всего работ "
             f"{sum(point.materials for point in points)}")
    return (f'<svg class="ta-curve{" ta-curve-large" if large else ""}" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none" role="img" aria-label="{escape(label, quote=True)}">'
            f'{"".join(bars)}{lines}</svg>')


def _curve_line(technology: RadarTechnology) -> str:
    confidence = ("мало данных" if technology.curve_confidence is None
                  else f"{technology.curve_confidence}/100")
    return f"Динамика по кривой эксперта: <b>{confidence}</b> · {escape(technology.trend)}"


def page_link(run: str | None, key: str | None = None) -> str:
    """Адрес страницы того же анализа: сохранённого (`?run=`) или текущего."""
    parts = ([f"run={escape(run, quote=True)}"] if run else []) + ([f"report={key}"] if key else [])
    return "?" + "&amp;".join(parts)


def _source_markup(technology: RadarTechnology, scores: Mapping[str, int]) -> str:
    """Only evidence attached to this technology; optional scores belong to publications."""
    if not technology.sources:
        return '<p class="ta-section-note">Источники по этой технологии не указаны.</p>'
    items = []
    for source in technology.sources:
        details = " · ".join(escape(value) for value in
                             (source.source_type, source.source, source.published) if value)
        score = scores.get(source.url)
        assessment = (f'<span class="ta-radar-source-score">Соответствие запросу: '
                      f'<b>{score}/100</b></span>' if score is not None else "")
        meta = f'<span class="ta-radar-source-meta">{details}</span>' if details else ""
        items.append(
            f'<li><a href="{escape(source.url, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{escape(source.title or source.url)}</a>'
            f'{meta}{assessment}</li>')
    return f'<ul class="ta-radar-source-list">{"".join(items)}</ul>'


def technology_card(technology: RadarTechnology, rank: int, key: str, run: str | None = None,
                    source_scores: Mapping[str, int] | None = None) -> str:
    original = (f'<span class="ta-radar-original">{escape(technology.title)}</span>'
                if technology.title_ru and technology.title_ru != technology.title else "")
    facts = [escape(technology.stage)]
    if technology.first_year:
        facts.append(f"с {technology.first_year} г.")
    if technology.all_time is not None:
        facts.append(f"{technology.all_time} работ всего")
    predictors = "".join(
        f'<li class="{"ta-plus" if part.weight > 0 else "ta-minus"}">{escape(part.label)}: {escape(part.value)}</li>'
        for part in technology.predictors[:3])
    source_count = len(technology.sources)
    kind = "Слабый сигнал" if technology.is_signal else "Кандидат"
    return (f'<article class="ta-card ta-radar-card"><div class="ta-radar-overview">'
            f'<div class="ta-card-top"><span class="ta-index">{rank}</span>'
            f'<span class="ta-publication-meta">{" · ".join(facts)}</span></div>'
            f'<span class="ta-radar-kind">{kind}</span><h3>{escape(_name(technology))}</h3>{original}'
            f'<div class="ta-card-assessment"><div class="ta-meter"><span class="ta-meter-label">Уверенность модели'
            f'</span><span class="ta-scale" style="--ta-value: {_percent(technology.probability)}" aria-hidden="true">'
            f'<i></i></span><b class="ta-meter-value">{_percent(technology.probability)}</b></div>'
            f'<p class="ta-radar-curve-line">{_curve_line(technology)}</p>{curve_svg(technology.points)}'
            f'<ul class="ta-radar-predictors">{predictors}</ul></div></div>'
            f'<details class="ta-radar-details"><summary class="ta-radar-expand">'
            f'Источники · {source_count}</summary><div class="ta-radar-source-panel">'
            f'<div class="ta-radar-source-head">'
            f'<h4>Источники по технологии</h4><a href="{page_link(run, key)}" target="_self">'
            f'Подробный отчёт</a></div>{_source_markup(technology, source_scores or {})}'
            f'</div></details></article>')


def stats_markup(result: RadarResult) -> str:
    return ('<div class="ta-stats ta-radar-stats">'
            f'<dl class="ta-stat"><dt>Кандидатов в слабые сигналы</dt><dd>{result.candidates_total}</dd></dl>'
            f'<dl class="ta-stat"><dt>Направлений в ТОПе</dt><dd>{len(result.technologies)}</dd></dl>'
            f'<dl class="ta-stat"><dt>Уверенность модели выше 75%</dt><dd>{result.high_confidence}</dd></dl>'
            f'<dl class="ta-stat"><dt>Обработано материалов</dt><dd>{result.sources_processed}</dd></dl></div>')


def radar_markup(status: RadarStatus, run: str | None = None,
                 source_scores: Mapping[str, int] | None = None) -> str:
    """Секция ТОПа технологий в любом состоянии фонового расчёта."""
    heading = '<div class="ta-section"><h2>ТОП-15 зарождающихся технологий</h2></div>'
    if status.state == "running" or status.result is None:
        if status.state == "unavailable":
            return heading + f'<p class="ta-section-note">{escape(status.message or "ТОП технологий недоступен.")}</p>'
        progress = f"{status.completed} из {status.total}" if status.total else "ищем кандидатов"
        # Готовый ТОП ждёт только перевода выжимок; «Считаем» тогда неправда.
        text = escape(status.message) if status.message else f"Считаем: {progress}"
        return heading + f'<p class="ta-section-note ta-radar-progress" role="status">{text}</p>'
    result = status.result
    old_policy_notice = (
        '<p class="ta-section-note ta-radar-policy-note">Этот сохранённый анализ создан до обновления '
        'отбора направлений. Запустите новый анализ, чтобы увидеть результаты по новым правилам.</p>'
        if run is not None and older_selection_policy(result.policy_version) else "")
    note = ("В ТОПе до 15 направлений: сначала слабые сигналы, затем прошедшие проверку "
            "нишевые кандидаты с меньшей уверенностью модели. Раскройте карточку, чтобы увидеть "
            "источники по этой технологии. Зрелые темы и фразы без работ исключаются правилами, "
            "даже при высоком проценте.")
    cards = "".join(technology_card(item, rank, f"t{rank}", run, source_scores)
                    for rank, item in enumerate(result.technologies, start=1))
    grid = (f'<div class="ta-grid ta-radar-grid">{cards}</div>' if cards else
            '<p class="ta-section-note">Ни один кандидат не прошёл отбор: все оказались зрелыми, '
            'угасающими или без данных. Причины — в журнале исключений ниже.</p>')
    return (heading + old_policy_notice + f'<p class="ta-section-note">{note}</p>' + stats_markup(result) + grid
            + exclusions_markup(result, run))


def exclusions_markup(result: RadarResult, run: str | None = None) -> str:
    if not result.excluded:
        return ""
    rows = "".join(
        f'<li><a href="{page_link(run, f"x{index}")}" target="_self">{escape(_name(item))}</a>'
        f'<span class="ta-radar-excluded-score">{_percent(item.probability)}'
        f'{" · исключено правилом" if item.rule_excluded else ""}</span>'
        f'<span class="ta-radar-reasons">{escape("; ".join(item.reasons) or "ниже порога модели")}</span></li>'
        for index, item in enumerate(result.excluded, start=1))
    return (f'<details class="ta-sources ta-radar-excluded"><summary>Почему не вошли в ТОП · {len(result.excluded)}'
            f'</summary><p class="ta-section-note">Зрелые технологии, угасающий интерес, хайп без научных '
            f'подтверждений и кандидаты с низкой оценкой модели.</p><ul>{rows}</ul></details>')


def _translated(original: str | None, russian: str | None) -> str:
    if not original:
        return '<p class="ta-report-empty">В найденных текстах подходящего фрагмента нет.</p>'
    if russian:
        return (f'<p>{escape(russian)} {TRANSLATED}</p>'
                f'<p class="ta-report-original">Оригинал: {escape(original)}</p>')
    return f'<p>{escape(original)}</p>'


def report_markup(technology: RadarTechnology, result: RadarResult, key: str, run: str | None = None,
                  source_scores: Mapping[str, int] | None = None) -> str:
    """Страница-отчёт по одной технологии: всё, что ТЗ требует показать по инсайту."""
    in_top = key.startswith("t")
    status = ("Слабый сигнал" if technology.is_signal else "Кандидат") if in_top else "Не вошёл в ТОП"
    reasons_heading = "Что ограничивает уверенность" if in_top else "Почему не вошёл в ТОП"
    predictors = "".join(
        f'<tr><td>{escape(part.label)}</td><td>{escape(part.value)}</td>'
        f'<td class="{"ta-plus" if part.weight > 0 else "ta-minus"}">{part.weight:+.2f}</td></tr>'.replace(".", ",")
        for part in technology.predictors)
    checks = "".join(f'<li class="{"ta-check-ok" if check.passed else "ta-check-no"}">'
                     f'{"✓" if check.passed else "✗"} {escape(check.label)}</li>' for check in technology.checks)
    scores = source_scores or {}
    source_rows = []
    for source in technology.sources:
        score = scores.get(source.url)
        assessment = (f'<span class="ta-report-source-score">Соответствие запросу: '
                      f'<b>{score}/100</b></span>' if score is not None else "")
        source_rows.append(
            f'<tr><td><a href="{escape(source.url, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{escape(source.title or source.url)}</a>{assessment}</td>'
            f'<td>{escape(source.published)}</td><td>{escape(source.source_type)}'
            f' · {escape(source.source)}</td><td>{escape(source.language)}</td>'
            f'<td>{escape(source.trust)}</td></tr>')
    sources = "".join(source_rows)
    reasons = ("".join(f"<li>{escape(reason)}</li>" for reason in technology.reasons)
               if technology.reasons else "")
    original = (f'<p class="ta-report-subtitle">Оригинальное название: {escape(technology.title)}</p>'
                if technology.title_ru and technology.title_ru != technology.title else "")
    facts = [f"<dt>Статус</dt><dd>{status}</dd>",
             f"<dt>Уверенность модели</dt><dd>{_percent(technology.probability)}"
             f"{' · исключено правилом' if technology.rule_excluded else ''}</dd>",
             f"<dt>Динамика по кривой эксперта</dt><dd>{'мало данных' if technology.curve_confidence is None else f'{technology.curve_confidence}/100'} · {escape(technology.trend)}</dd>",
             f"<dt>Стадия</dt><dd>{escape(technology.stage)}</dd>",
             f"<dt>Первое упоминание</dt><dd>{technology.first_year or 'не найдено'}</dd>",
             f"<dt>Работ за всё время</dt><dd>{technology.all_time if technology.all_time is not None else 'нет данных'}</dd>"]
    return (
        f'<article class="ta-report"><p class="ta-report-back"><a href="{page_link(run)}" target="_self">'
        f'← К результатам</a></p>'
        f'<p class="ta-report-kicker">Отчёт по технологии · запрос «{escape(result.query)}» · срез {escape(result.as_of)}</p>'
        f'<h1>{escape(_name(technology))}</h1>{original}'
        f'<dl class="ta-report-facts">{"".join(facts)}</dl>'
        + (f'<section><h2>{reasons_heading}</h2><ul>{reasons}</ul></section>' if reasons else "")
        + f'<section><h2>Описание технологии</h2>{_translated(technology.description, technology.description_ru)}</section>'
        f'<section><h2>Потенциальное преимущество</h2>{_translated(technology.advantage, technology.advantage_ru)}</section>'
        f'<section><h2>Кейс-пример</h2>{_translated(technology.case, technology.case_ru)}</section>'
        f'<section><h2>Почему модель дала такую оценку</h2><p>{escape(technology.rationale)}</p>'
        f'<div class="ta-table-wrap"><table class="ta-table"><thead><tr><th scope="col">Признак</th>'
        f'<th scope="col">Значение</th><th scope="col">Вклад в оценку</th></tr></thead><tbody>{predictors}</tbody>'
        f'</table></div><p class="ta-report-note">Плюс толкает к «слабому сигналу», минус — от него. Модель — '
        f'логистическая регрессия, обученная на датасете организаторов; вероятность не является гарантией '
        f'будущего успеха технологии.</p></section>'
        f'<section><h2>Динамика: логика эксперта</h2><p class="ta-report-note">Столбики — сумма коэффициентов '
        f'достоверности найденных работ по месяцам (научные каталоги 1,0; препринты 0,85; сообщества 0,25), линия — '
        f'сглаживание за 3 месяца, пунктир — парабола.</p>{curve_svg(technology.points, large=True)}'
        f'<ul class="ta-report-checks">{checks}</ul></section>'
        f'<section><h2>Источники</h2><div class="ta-table-wrap"><table class="ta-table"><thead><tr>'
        f'<th scope="col">Материал</th><th scope="col">Дата</th><th scope="col">Тип · источник</th>'
        f'<th scope="col">Язык</th><th scope="col">Доверенность</th></tr></thead><tbody>{sources}</tbody></table>'
        f'</div><p class="ta-report-note">Названия материалов — на языке оригинала. Уровень доверенности: высокий — '
        f'научные каталоги, патенты и препринты; пониженный — агрегаторы и сообщества, они не могут быть '
        f'единственным основанием.</p></section></article>')
