"""Readable passports with actual yearly counts and exact original quotations."""

from collections.abc import Callable
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from app.ui.viewport import ScrollViewport

ROLE_NAMES = {"problem": "Проблема", "advantage": "Преимущество", "case": "Пример исследования",
              "summary": "Объяснение", "limitation": "Ограничение", "novelty": "Новизна",
              "application": "Применение"}
GATE_NAMES = {
    "no_supported_primary_result_observation": "нет подтверждённого собственного результата исследования",
    "primary_observation_has_known_older_antecedents": "известны более ранние предшественники результата",
    "incomplete_or_incomparable_history": "история неполная или несопоставимая",
    "fewer_than_ten_recent_studies": "найдено меньше 10 исследований за последние 3 полных года",
    "fewer_than_two_active_recent_years": "публикации есть менее чем в двух последних полных годах",
    "growth_below_1_5": "рост ниже порога 1,5",
    "nonpositive_recent_slope": "последние годы не показывают устойчивого роста",
    "recent_decline_or_transient_burst": "после роста наблюдается снижение или кратковременный всплеск",
    "candidate_not_specific": "самостоятельность технологии ещё не проверена",
    "unsupported_card_fields": "не все ключевые поля подтверждены источниками",
    "novelty_not_verified": "новизна относительно более ранних аналогов не подтверждена",
    "field_exposure_unavailable": "нет сопоставимой истории объёма направления",
    "earlier_search_not_complete": "поиск более ранних аналогов не завершён",
    "no_reviewed_or_archived_novelty_evidence": "нет проверенной новизны или датированного заявления авторов о техническом отличии",
    "first_observation_not_recent": "первое наблюдение в последние 6 полных лет не установлено",
    "no_observed_field_relative_growth": "не наблюдается устойчивого роста относительно направления",
    "fewer_than_two_recent_studies": "найдено меньше двух исследований за последние 3 полных года",
    "field_relative_growth_not_statistically_supported": "рост относительно направления статистически не подтверждён",
    "weak_signal_first_observation_older_than_three_years": "для слабого сигнала не установлено первое наблюдение в последние 3 полных года",
}

EXPOSURE_REASONS = {
    "field_exposure_missing": "источник не сообщил сопоставимые объёмы направления по всем годам",
    "zero_field_exposure": "в истории направления есть годы с нулевым знаменателем",
    "candidate_exceeds_reference_field_exposure": "число работ кандидата превышает объём выбранного направления; требуется уточнение поисковых границ",
    "no_candidate_observations": "в истории кандидата нет найденных работ",
}
GATE_NAMES.update(EXPOSURE_REASONS)


def novelty_prefix(claim):
    method = claim.get("grounding_method") or ""
    if claim.get("support") == "contradicted":
        return "[Утверждение о новизне опровергнуто] "
    if claim.get("support") == "supported" and method.startswith("reviewed-novelty/"):
        return "[Экспертная оценка новизны] "
    if method.startswith("archived-author-novelty/"):
        return "[Авторская гипотеза · новизна независимо не подтверждена] "
    return "[Новизна требует проверки] "


def signal_metric_text(assessment):
    """Describe missing values separately from zero and heuristic scores from uncertainty."""
    if assessment.get("methodology_version") not in {"3.2.0", "3.3.0", "3.4.0"} and "relative_growth" not in assessment:
        return ()
    relative = assessment.get("relative_growth")
    if not relative or relative.get("status") != "available":
        reason = EXPOSURE_REASONS.get((relative or {}).get("reason") or "", "сопоставимых данных недостаточно")
        growth = "Рост относительно направления не оценён: " + reason + "."
    else:
        ratio = relative.get("raw_ratio")
        growth = (f"Отношение частот с учётом объёма направления: {ratio:.2f}." if ratio is not None else
                  "Несглаженное отношение частот не определено: в базовом периоде не найдено работ кандидата.")
        smoothed = relative.get("smoothed_ratio")
        if smoothed is not None:
            growth += f" Сглаженная оценка отношения: {smoothed:.2f}."
        lower, upper = relative.get("ratio_lower_95"), relative.get("ratio_upper_95")
        low_text = f"{lower:.2f}" if lower is not None else "не определена"
        high_text = f"{upper:.2f}" if upper is not None else "без конечной границы"
        growth += f" Условный 95% интервал отношения: нижняя граница {low_text}, верхняя — {high_text}."
        growth += (" Превышение роста направления поддержано этой моделью."
                   if relative.get("excess_growth_supported") else
                   " Превышение роста направления с учётом неопределённости не подтверждено.")
    priority = assessment.get("signal_priority")
    priority_text = (f"Приоритет проверки гипотезы: {priority:.2f}/100." if priority is not None else
                     "Приоритет проверки гипотезы не определён.")
    return (growth, "Сравнение использует фиксированный запрос к OpenAlex. Объём индексированных работ — приближение "
            "к объёму направления; он не равен полному числу исследований в мире. Интервал учитывает случайность счёта "
            "при допущениях модели, но не смещения поиска и индексации.",
            priority_text + " Это некалиброванная эвристика, а не вероятность истинного слабого сигнала.")


@dataclass(frozen=True)
class _Passage:
    text: str
    prefix: str = ""
    url: str | None = None


def _paragraph(parent, text: str, *, pady=3):
    """Keep a single legal long item from making an enormous native Tk surface."""
    if len(text) > 2000:
        box = ScrolledText(parent, wrap="word", width=1, height=6, font="TkDefaultFont", takefocus=True)
        box.insert("1.0", text)
        box.configure(state="disabled")
        def leave(_event=None, *, reverse=False):
            target = box.tk_focusPrev() if reverse else box.tk_focusNext()
            if target is not None:
                target.focus_set()
            return "break"
        box.bind("<Tab>", leave)
        box.bind("<Shift-Tab>", lambda event: leave(event, reverse=True))
        box.bind("<ISO_Left_Tab>", lambda event: leave(event, reverse=True))
        box.pack(fill="x", pady=pady)
        return box
    label = ttk.Label(parent, text=text, wraplength=880, justify="left")
    label.pack(anchor="w", pady=pady)
    return label


class _PassagePages(ttk.Frame):
    """Bound live widgets, retaining full immutable passages in their original order."""

    def __init__(self, parent, panel, viewport, section: str, passages: tuple[_Passage, ...]):
        super().__init__(parent)
        self.panel, self.viewport, self.section = panel, viewport, section
        self.passages = passages
        self.page = 0
        self.link_pending = False
        self.page_ranges: list[tuple[int, int]] = []
        start = size = 0
        for index, item in enumerate(passages):
            length = len(item.text) + len(item.prefix) + len(item.url or "")
            # Never split or truncate a quote. A single legal long item uses
            # bounded-height text widgets, even if it exceeds the page budget.
            if index > start and (index - start >= 20 or size + length > 20000):
                self.page_ranges.append((start, index))
                start, size = index, 0
            size += length
        self.page_ranges.append((start, len(passages)))
        self.pack(fill="x")
        self.position = ttk.Label(self, wraplength=880, style="Muted.TLabel")
        if len(self.page_ranges) > 1 or section == "Доказательства":
            self.position.pack(anchor="w", pady=4)
        self.buttons: list[ttk.Button] = []
        if len(self.page_ranges) > 1:
            controls = ttk.Frame(self)
            controls.pack(fill="x", pady=4)
            for column, (title, target) in enumerate((("Первая", lambda: 0), ("Назад", lambda: self.page - 1),
                                                     ("Далее", lambda: self.page + 1),
                                                     ("Последняя", lambda: len(self.page_ranges) - 1))):
                def choose(select: Callable[[], int] = target):
                    self.show(select())
                button = ttk.Button(controls, text=title, command=choose)
                button.grid(row=column // 2, column=column % 2, sticky="ew", padx=3, pady=2)
                controls.columnconfigure(column % 2, weight=1)
                self.buttons.append(button)
        self.body = ttk.Frame(self)
        self.body.pack(fill="x")
        self.notice = ttk.Label(self, wraplength=880, style="Muted.TLabel")
        self.show(0, focus=False)

    def _notice(self, text: str):
        if self.winfo_exists() and not self.panel.app.closing:
            self.notice.configure(text=text)
            self.notice.pack(before=self.body, anchor="w", pady=4)

    def _open(self, url: str):
        if self.link_pending:
            self._notice("Источник открывается. Дождитесь завершения перед открытием следующей ссылки.")
            return "break"
        self.link_pending = True
        self._notice("Открываем источник в браузере…")
        def opened(_):
            self.link_pending = False
            self._notice("Источник открыт в браузере.")
        def failed(_error):
            self.link_pending = False
            self._notice("Не удалось открыть источник. Проверьте доступность браузера и повторите открытие ссылки.")
        if not self.panel.app.controller.call("pilot-evidence-link-" + str(self.winfo_toplevel()),
                                               "open_url", opened, failed, url):
            self.link_pending = False
            self._notice("Открытие источника сейчас недоступно. Дождитесь текущей операции и повторите.")
        return "break"

    def show(self, page: int, *, focus: bool = True):
        page = min(max(0, page), len(self.page_ranges) - 1)
        for child in self.body.winfo_children():
            child.destroy()
        self.page = page
        start, end = self.page_ranges[page]
        self.position.configure(text=f"{self.section}: {start + 1}–{end} из {len(self.passages)} · "
                                f"страница {page + 1} из {len(self.page_ranges)}. Полный текст без сокращений.")
        for item in self.passages[start:end]:
            _paragraph(self.body, item.prefix + item.text, pady=(8, 3))
            if item.url is not None:
                def open_link(_event=None, url=item.url):
                    return self._open(url)
                if len(item.url) > 2000:
                    _paragraph(self.body, item.url)
                    ttk.Button(self.body, text="Открыть первоисточник", command=open_link).pack(anchor="w", pady=(0, 4))
                else:
                    link = ttk.Label(self.body, text=item.url, foreground="#45bfd0", cursor="hand2", takefocus=True, wraplength=880)
                    link.pack(anchor="w", pady=(0, 4))
                    for event in ("<Button-1>", "<Return>", "<space>"):
                        link.bind(event, open_link)
        for index, button in enumerate(self.buttons):
            disabled = page == 0 if index < 2 else page == len(self.page_ranges) - 1
            button.state(["disabled"] if disabled else ["!disabled"])
        if focus:
            self.viewport.focus_when_visible(self.position)


def show_passport(panel, card, payload):
    window = tk.Toplevel(panel.app.root)
    window.title(card["candidate"]["label"])
    window.geometry("980x720")
    panel.app.child_windows.append(window)
    viewport = ScrollViewport(window, padding=18)
    viewport.pack(fill="both", expand=True)
    frame = viewport.content
    ttk.Label(frame, text=card["candidate"]["label"], style="Section.TLabel", wraplength=880).pack(anchor="w")
    from app.ui.pilot_panel import category_label
    _paragraph(frame, category_label(card) + " · " + card["candidate"]["definition"], pady=8)
    if card["category"] in {"weak_signal_candidate", "emerging_candidate"}:
        _paragraph(frame, "Автоматическая гипотеза: техническое отличие заявлено авторами сохранённых источников. "
                   "Это не независимое подтверждение новизны. Проверьте ранние аналоги и выводы в паспорте.")
    from app.ui.pilot_materials import sensitivity_dialog, show_matches
    identifier = card["candidate"]["candidate_id"]
    run_id = payload.get("view_id", payload["result"]["run_id"])
    top_ids = payload["result"].get("top_trend_ids")
    if top_ids is None:
        top_ids = [item["candidate"]["candidate_id"] for item in payload["result"].get("cards", ())
                   if item["category"] == "confirmed_trend"][:15]
    _paragraph(frame, (f"Позиция в сохранённом TOP-15: {top_ids.index(identifier) + 1}."
                       if identifier in top_ids else "Кандидат вне TOP-15. Его наличие в результате не означает подтверждение слабого сигнала."))
    actions = ttk.Frame(frame)
    actions.pack(fill="x", pady=8)
    ttk.Button(actions, text="Отчёты и препринты по технологии", command=lambda: show_matches(panel, run_id, identifier)).pack(side="left")
    ttk.Button(frame, text="Развить и перепроверить кандидата",
               command=lambda: panel.refine_candidate(run_id, identifier)).pack(anchor="w", pady=6)
    if card.get("historical_snapshot_id"):
        ttk.Button(actions, text="Проверить устойчивость", command=lambda: sensitivity_dialog(panel, run_id, identifier)).pack(side="left", padx=8)
        from app.ui.pilot_review import start_review
        ttk.Button(frame, text="Проверить ранние аналоги и оценить новизну", command=lambda: start_review(panel, run_id, identifier)).pack(anchor="w", pady=6)
    novelty = tuple(_Passage(claim["text"], novelty_prefix(claim)) for claim in card["claims"] if claim["role"] == "novelty")
    if novelty:
        ttk.Label(frame, text=ROLE_NAMES["novelty"], style="Section.TLabel").pack(anchor="w", pady=(12, 4))
        _PassagePages(frame, panel, viewport, ROLE_NAMES["novelty"], novelty)
    for role in ("problem", "advantage", "case"):
        ttk.Label(frame, text=ROLE_NAMES[role], style="Section.TLabel").pack(anchor="w", pady=(12, 4))
        claims = [claim for claim in card["claims"] if claim["role"] == role]
        if not claims:
            ttk.Label(frame, text="Недостаточно доказательств в найденных публикациях.", wraplength=880).pack(anchor="w")
        if claims:
            _PassagePages(frame, panel, viewport, ROLE_NAMES[role], tuple(_Passage(claim["text"],
                "" if claim["support"] == "supported" else "[Интерпретация требует проверки] ") for claim in claims))
    assessment_artifact = next((item for item in payload.get("assessments", [])
                                if item["assessment"]["candidate_id"] == card["candidate"]["candidate_id"]), None)
    if assessment_artifact:
        assessment = assessment_artifact["assessment"]
        ttk.Label(frame, text="История найденных исследований", style="Section.TLabel").pack(anchor="w", pady=(18, 8))
        history = assessment_artifact["inputs"]["history"]
        observations = history["observations"]
        coverage = history.get("coverage", {})
        if coverage.get("state") != "complete" or not coverage.get("comparable"):
            _paragraph(frame, "История неполная или несопоставимая. На графике показаны только найденные работы; "
                       "ноль не доказывает отсутствие исследований в этом году.")
        graph = tk.Canvas(frame, height=190, background="#17222e", highlightthickness=0)
        graph.pack(fill="x", pady=6)
        def draw(_=None):
            graph.delete("all")
            width = max(graph.winfo_width(), 300)
            counts = [len(item["study_ids"]) for item in observations]
            maximum = max(counts, default=0) or 1
            step = (width - 60) / max(1, len(observations))
            for index, (observation, count) in enumerate(zip(observations, counts, strict=True)):
                x = 35 + step * index
                height = 120 * count / maximum
                graph.create_rectangle(x, 150 - height, x + max(4, step - 12), 150, fill="#26b8c4", outline="")
                graph.create_text(x + step / 2 - 6, 168, text=str(observation["year"]), fill="white")
                graph.create_text(x + step / 2 - 6, 140 - height, text=str(count), fill="white")
        graph.bind("<Configure>", draw)
        ttk.Label(frame, text=f"Последние 3 полных года: {assessment['recent_studies']} исследований; "
                  f"предыдущие 3 года: {assessment['baseline_studies']}. "
                  f"Сглаженное отношение: {assessment['smoothed_growth']:.2f}. "
                  f"Первое наблюдение в проверенной выборке: {assessment.get('first_observed_year') or 'не установлено'}. "
                  + ("Более ранние аналоги проверены в пределах охвата источников."
                     if history.get("earlier_search_complete") else "Поиск более ранних аналогов ещё не завершён."),
                  wraplength=880).pack(anchor="w", pady=6)
        if assessment.get("methodology_version") not in {"3.2.0", "3.3.0", "3.4.0"}:
            score = assessment.get("priority_score")
            score_text = (f"Приоритет: {score:.2f}/100" if score is not None else
                          f"Приоритет не полностью определён: {assessment['priority_lower_bound']:.2f}–{assessment['priority_upper_bound']:.2f}/100")
            ttk.Label(frame, text=score_text + ". Эвристический приоритет ещё не откалиброван на независимой разметке; это не вероятность слабого сигнала или успеха технологии. "
                      "Границы отражают неизвестные компоненты, а не статистический доверительный интервал.", wraplength=880).pack(anchor="w")
        if "single_primary_result_is_observation_not_confirmed_weak_signal" in assessment.get("limitations", []):
            _paragraph(frame, "Раннее наблюдение: найдён конкретный первичный результат, заслуживающий проверки. "
                "Приоритет 10 — одинаковая отметка для таких кандидатов. Рост публикаций и новизна ещё не подтверждены; "
                "неполная история, отсутствие поискового интереса или инвестиций не препятствуют наблюдению. "
                "Результат текущего года не добавлен к статистике завершённых лет.", pady=6)
        for text in signal_metric_text(assessment):
            _paragraph(frame, text, pady=6)
        if assessment["gate_failures"]:
            ttk.Label(frame, text="Подтверждение ограничено: " + "; ".join(GATE_NAMES.get(code, code) for code in assessment["gate_failures"]), wraplength=880).pack(anchor="w", pady=6)
    ttk.Label(frame, text="Первичные доказательства", style="Section.TLabel").pack(anchor="w", pady=(18, 8))
    patent_signal = next((item for item in payload.get("patent_signals", []) if item["candidate_id"] == identifier), None)
    if patent_signal:
        ttk.Label(frame, text="Отдельный патентный сигнал", style="Section.TLabel").pack(anchor="w", pady=(12, 6))
        available = patent_signal["coverage"]["state"] != "unavailable"
        text = (f"Найдено публикаций: {len(patent_signal['documents'])}; семейств с подтверждённым EPO ID: {len(patent_signal['families'])}. "
                f"Без известного семейства: {patent_signal['unresolved_family_publications']}." if available else
                "Патентный источник недоступен. Это не означает отсутствие патентов.")
        ttk.Label(frame, text=text + " Патенты не включены в график научных публикаций и итоговый балл.", wraplength=880).pack(anchor="w", pady=4)
    if card["evidence"]:
        _PassagePages(frame, panel, viewport, "Доказательства", tuple(_Passage(evidence["quote"], f"{number}. ",
            evidence["source_url"]) for number, evidence in enumerate(card["evidence"], 1)))
    ttk.Label(frame, text="Ограничения", style="Section.TLabel").pack(anchor="w", pady=(18, 8))
    if card["limitations"]:
        _PassagePages(frame, panel, viewport, "Ограничения", tuple(_Passage(item, "• ") for item in card["limitations"]))
    ttk.Label(frame, text="Оригинальные цитаты сохранены на языке источника. Дата первого наблюдения не означает дату изобретения.",
              wraplength=880, style="Muted.TLabel").pack(anchor="w", pady=12)


def export_payload(panel, payload):
    identifier = payload.get("view_id", payload["result"]["run_id"])
    owner = panel.generation
    path = filedialog.asksaveasfilename(parent=panel.app.root, title="Экспорт результата и доказательств",
                                       defaultextension=".trendresult", filetypes=[("Результат анализа", "*.trendresult")])
    if path:
        panel._call("export", "export_result", lambda _: panel.message.set("Результат и проверяемые доказательства экспортированы."),
                    identifier, path, owner=owner)
