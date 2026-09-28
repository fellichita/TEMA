"""Радар технологий: кандидаты, паспорт, рейтинг и сборка ТОПа без сети."""

from datetime import date

from app.radar import pipeline
from app.radar.evidence import FirstMention, language, pool_evidence
from app.radar.passport import build_passport, excluded_by_rule, exclusion_reasons, is_weak_signal, plural
from app.radar.phrases import Candidate, mine_candidates, variants
from app.signal_model.model import Prediction, SignalModel
from app.trend_confidence import Material, assess_curve, month_window
from app.trend_history import SourceCoverage, TechnologyHistory
from app.ui.radar_client import _technology

AS_OF = date(2026, 9, 26)


def item(number: int, title: str, source: str = "crossref", summary: str = "", kind: str = "journal-article") -> dict:
    return {"publication_id": f"p{number}", "title": title, "summary": summary, "source_id": source,
            "source_ids": [source], "kind": kind, "url": f"https://example.org/{number}",
            "published_at": "2026-08-01"}


POOL = [
    item(1, "Sulfide solid electrolytes for all-solid-state batteries", "crossref"),
    item(2, "Halide solid electrolytes with high conductivity", "arxiv"),
    item(3, "Sulfide solid electrolyte interfaces in practice", "europe_pmc"),
    item(4, "A sulfide solid electrolyte prototype cell", "arxiv", "We demonstrate a prototype. A startup raised seed funding."),
    item(5, "Ionic conductivity of garnets", "crossref"),
    item(6, "Ionic conductivity of sulfides", "arxiv"),
    item(7, "Ionic conductivity in polymers", "europe_pmc"),
    item(8, "Solid-state batteries review", "crossref"),
    item(9, "Solid-state batteries outlook", "arxiv"),
    item(10, "Solid-state batteries roadmap", "europe_pmc"),
    item(11, 'Review for "Sulfide solid electrolytes for all-solid-state batteries"', "crossref", kind="peer-review"),
    item(12, 'Decision letter for "Sulfide solid electrolytes for all-solid-state batteries"', "crossref",
         kind="peer-review"),
]


def test_candidates_are_specific_technologies_not_properties_or_the_query_itself():
    phrases = [candidate.phrase for candidate in mine_candidates(POOL, query_terms=["solid-state batteries"])]
    assert "sulfide solid electrolyte" in phrases
    assert not any("conductivity" in phrase for phrase in phrases)
    assert "solid-state battery" not in phrases
    sulfide = next(c for c in mine_candidates(POOL, query_terms=["solid-state batteries"])
                   if c.phrase == "sulfide solid electrolyte")
    # Рецензии Crossref не добавляют материалов.
    assert sulfide.documents == ("p1", "p3", "p4") and sulfide.sources == ("arxiv", "crossref", "europe_pmc")


def test_sibling_mechanisms_survive_shared_reviews_and_abstracts():
    pool = [item(100 + index, f"Sulfide solid electrolyte design {index}", source,
                 "Halide solid electrolyte and sulfide solid electrolyte are compared.")
            for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    pool += [item(110 + index, f"Halide solid electrolyte design {index}", source,
                  "Sulfide solid electrolyte and halide solid electrolyte are compared.")
             for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    candidates = mine_candidates(pool, query_terms=["solid-state batteries"])
    phrases = {candidate.phrase for candidate in candidates}
    assert {"sulfide solid electrolyte", "halide solid electrolyte"} <= phrases
    assert "solid electrolyte" not in phrases


def test_sibling_mechanisms_can_both_enter_the_top(monkeypatch):
    pool = [item(120 + index, f"Sulfide solid electrolyte design {index}", source,
                 "Halide solid electrolyte and sulfide solid electrolyte are compared.")
            for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    pool += [item(125 + index, f"Halide solid electrolyte design {index}", source,
                  "Sulfide solid electrolyte and halide solid electrolyte are compared.")
             for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    monkeypatch.setattr(pipeline, "collect_history", fake_history)
    monkeypatch.setattr(pipeline, "openalex_years", lambda *_args: ((2024, 5), (2025, 20)))
    radar = pipeline.build_radar(pool, query="Батареи", query_terms=["solid-state batteries"], as_of=AS_OF)
    assert {technology["phrase"] for technology in radar["technologies"]} == {
        "sulfide solid electrolyte", "halide solid electrolyte"}
    assert all(technology["probability"] > 0.6 for technology in radar["technologies"])
    assert all(technology["phrase"] in source["title"].casefold()
               for technology in radar["technologies"] for source in technology["sources"])


def test_top_uses_evidence_backed_candidates_without_inventing_model_confidence(monkeypatch):
    from app.signal_model.model import Prediction

    pool = [item(140 + index, f"Sulfide solid electrolyte design {index}", source)
            for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    pool += [item(145 + index, f"Halide solid electrolyte design {index}", source)
             for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    monkeypatch.setattr(pipeline, "collect_history", fake_history)
    monkeypatch.setattr(pipeline, "openalex_years", lambda *_args: ((2024, 5), (2025, 20)))
    monkeypatch.setattr(pipeline, "decide", lambda _model, passport: Prediction(
        "x", 0.42 if passport.technology.title.startswith("halide") else 0.93, True, ()))
    radar = pipeline.build_radar(pool, query="Батареи", query_terms=["solid-state batteries"], as_of=AS_OF)
    assert len(radar["technologies"]) == 2
    assert [entry["is_signal"] for entry in radar["technologies"]] == [True, False]
    assert radar["technologies"][1]["probability"] == 0.42
    assert radar["technologies"][1]["rule_excluded"] is False
    assert "42%" in radar["technologies"][1]["reasons"][0]


def test_lower_confidence_candidates_beyond_top_limit_keep_their_explanation(monkeypatch):
    from app.signal_model.model import Prediction

    pool = [item(160 + index, f"Sulfide solid electrolyte design {index}", source)
            for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    pool += [item(165 + index, f"Halide solid electrolyte design {index}", source)
             for index, source in enumerate(("crossref", "arxiv", "europe_pmc"))]
    monkeypatch.setattr(pipeline, "collect_history", fake_history)
    monkeypatch.setattr(pipeline, "openalex_years", lambda *_args: ((2024, 5), (2025, 20)))
    monkeypatch.setattr(pipeline, "decide", lambda _model, passport: Prediction(
        "x", 0.55 if passport.technology.title.startswith("sulfide") else 0.42, True, ()))
    monkeypatch.setattr(pipeline, "TOP_SIZE", 1)
    radar = pipeline.build_radar(pool, query="Батареи", query_terms=["solid-state batteries"], as_of=AS_OF)
    assert [(entry["phrase"], entry["is_signal"]) for entry in radar["technologies"]] == [
        ("sulfide solid electrolyte", False)]
    assert radar["excluded"][0]["reasons"][0] == pipeline.BELOW_TOP
    assert "42%" in radar["excluded"][0]["reasons"][1]


def test_two_independent_publications_can_propose_a_narrow_direction():
    pool = [item(130, "Sulfide anode prototype", "crossref"),
            item(131, "Sulfide anode interface", "arxiv")]
    candidates = mine_candidates(pool, query_terms=["sulfide cathodes", "anode interfaces"])
    assert "sulfide anode" in {candidate.phrase for candidate in candidates}


def test_mixed_lightning_network_senses_do_not_become_an_energy_technology():
    titles = (
        "Lightning Network bitcoin routing", "Lightning Network payment channels",
        "Lightning Network transaction fees", "Lightning Network wallet attacks",
        "Lightning Network blockchain scalability", "Lightning Network cryptocurrency topology",
        "Lightning Network storm observations", "Lightning Network thunderstorm detection",
        "Lightning Network for low-light imaging", "Lightning Network image enhancement",
    )
    pool = [item(500 + index, title, ("crossref", "openreview")[index % 2],
                 "A study of the stated network and its application.")
            for index, title in enumerate(titles)]
    assert "lightning network" in {candidate.phrase for candidate in mine_candidates(pool)}
    assert "lightning network" not in {candidate.phrase for candidate in mine_candidates(
        pool, query_terms=["electromagnetic energy from lightning"])}


def test_frequent_specific_technology_can_use_a_different_name_than_the_query():
    endings = ("electrodes", "coatings", "interfaces", "particles", "films",
               "crystals", "powders", "composites", "cells", "prototypes")
    pool = [item(600 + index, f"Lithium iron phosphate {ending}", ("crossref", "arxiv")[index % 2],
                 "Cycle life and composition are measured in this study.")
            for index, ending in enumerate(endings)]
    assert "lithium iron phosphate" in {candidate.phrase for candidate in mine_candidates(
        pool, query_terms=["electric vehicle battery"])}


def test_variants_cover_singular_and_plural():
    assert variants("sulfide solid electrolytes") == ("sulfide solid electrolytes", "sulfide solid electrolyte")
    assert variants("sodium battery") == ("sodium battery", "sodium batteries")


def test_pool_evidence_reads_stage_funding_and_source_cards():
    evidence = pool_evidence("sulfide solid electrolyte", POOL)
    assert evidence.stage == "Прототип/PoC" and evidence.funding_mentions == 3
    assert {view.trust for view in evidence.items} == {"высокий"}
    assert evidence.items[0].source_type in {"научная публикация", "препринт"}
    assert language("Статья на русском") == "русский" and language("English") == "английский"


def curve_of(counts: list[int]):
    months = month_window(AS_OF)
    materials = [Material(f"{index}-{number}", ("crossref", "arxiv")[number % 2], months[index])
                 for index, count in enumerate(counts) for number in range(count)]
    return assess_curve(materials, AS_OF)


def test_passport_marks_mature_topics_and_explains_exclusion():
    growing = curve_of([round(0.04 * index * index) + 1 for index in range(24)])
    evidence = pool_evidence("sulfide solid electrolyte", POOL)
    niche = build_passport("sulfide solid electrolyte", "Батареи", AS_OF, growing,
                           FirstMention(date(2023, 1, 5), 20, 10), evidence, None)
    assert not niche.mature and niche.first_year == 2023 and niche.technology.stage == "Прототип/PoC"
    assert "Тема нишевая: 30 работ за всё время." in niche.technology.rationale
    assert niche.technology.trend == "Растёт быстро"
    mature = build_passport("lithium-ion battery", "Батареи", AS_OF, growing,
                            FirstMention(date(2003, 1, 1), 5000, 3218), evidence, None)
    assert mature.mature and mature.technology.stage == "Массовое внедрение"
    model = SignalModel.load(pipeline.MODEL_PATH)
    niche_prediction, mature_prediction = model.predict([niche.technology, mature.technology])
    assert is_weak_signal(niche, niche_prediction)
    # Растущая, но массовая тема исключается правилом зрелости, даже если модель её пропустила.
    assert not is_weak_signal(mature, mature_prediction)
    reasons = exclusion_reasons(mature, growing, mature_prediction)
    assert reasons[0] == "Зрелая тема: 8218 работ за всё время, первое упоминание в 2003 году"


def test_openalex_years_decide_volume_first_year_and_old_concepts():
    flat = curve_of([2] * 24)
    evidence = pool_evidence("sulfide solid electrolyte", POOL)
    # Одиночная статья 1982 года не делает тему старой: год появления — первый с тремя работами.
    emerging = build_passport("prompt injection", "ИИ", AS_OF, curve_of([round(0.04 * i * i) + 1 for i in range(24)]),
                              FirstMention(None, 50, 0, ((1982, 1), (2023, 16), (2024, 80), (2025, 244))),
                              evidence, None)
    assert emerging.first_year == 2023 and emerging.all_time == 341 and not emerging.mature
    assert emerging.volume_basis == "openalex"
    policy = build_passport("security policy", "ИИ", AS_OF, flat,
                            FirstMention(None, 10, 5, ((1998, 40), (2025, 11_000))), evidence, None)
    assert policy.mature and not policy.old_concept
    concept = build_passport("intelligent agents", "ИИ", AS_OF, flat,
                             FirstMention(None, 10, 5, ((2000, 30), (2025, 200))), evidence, None)
    assert concept.old_concept and concept.mature
    model = SignalModel.load(pipeline.MODEL_PATH)
    reasons = exclusion_reasons(concept, flat, model.predict([concept.technology])[0])
    assert reasons[0] == "Давно известное понятие (с 2000 года) без ускорения роста"


def test_russian_plural_forms():
    assert [plural(number, "работа", "работы", "работ") for number in (1, 3, 5, 11, 21, 44)] == [
        "работа", "работы", "работ", "работ", "работа", "работы"]


def test_rank_score_prefers_confirmed_curves_but_keeps_unknown_neutral():
    assert pipeline.rank_score(1.0, 100) == 1.0
    assert pipeline.rank_score(1.0, 0) == 0.6
    assert pipeline.rank_score(1.0, None) == 0.8


def fake_history(forms, as_of, **_):
    from app.trend_history import HistoryRecord

    months = month_window(as_of)
    records = [HistoryRecord(f"{forms[0]}-{index}-{number}", ("crossref", "arxiv")[number % 2], months[index],
                             forms[0], "https://example.org", f"{months[index]}-15")
               for index in range(24) for number in range(round(0.04 * index * index) + 1)]
    coverage = tuple(SourceCoverage(source, "complete", 10, 10) for source in ("crossref", "arxiv"))
    return TechnologyHistory(tuple(forms), as_of, months, tuple(records), coverage)


def fake_years(_client, phrase, _cancel, _key=None):
    # Галогенидные электролиты — зрелая тема во всей науке, остальное — молодые фразы.
    if phrase.lower().startswith("halide"):
        return ((2001, 40), (2025, 9000))
    return ((2024, 5), (2025, 20))


def test_build_radar_splits_top_and_exclusions_without_network(monkeypatch):
    monkeypatch.setattr(pipeline, "collect_history", fake_history)
    monkeypatch.setattr(pipeline, "openalex_years", fake_years)
    extra = [item(20, "Halide solid electrolytes in cells", "crossref"),
             item(21, "Halide solid electrolytes at scale", "europe_pmc")]
    radar = pipeline.build_radar(POOL + extra, query="твердотельные батареи", query_terms=["solid-state batteries"],
                                 as_of=AS_OF)
    titles = [entry["title"].lower() for entry in radar["technologies"]]
    assert "sulfide solid electrolyte" in titles
    halide = next(entry for entry in radar["excluded"] if entry["title"].lower().startswith("halide"))
    # Зрелая тема отсеяна по годам OpenAlex, без помесячной истории.
    assert halide["reasons"][0].startswith("Зрелая тема: 9040 работ во всей науке")
    assert halide["curve"]["materials"] == 0
    assert radar["evaluated"] == radar["candidates_total"] and radar["failed"] == []
    assert all(len(entry["sources"]) <= pipeline.EXCLUDED_SOURCE_LIMIT for entry in radar["excluded"])
    top = radar["technologies"][0]
    assert top["curve"]["confidence"] == 100 and len(top["curve"]["months"]) == 24
    assert top["passport"]["first_year"] == 2024 and top["passport"]["volume_basis"] == "openalex"
    assert {"label", "value", "weight"} <= set(top["predictors"][0])


def test_technology_sources_are_linked_deduplicated_and_bounded(monkeypatch):
    monkeypatch.setattr(pipeline, "collect_history", fake_history)
    monkeypatch.setattr(pipeline, "first_mention", lambda *_args, **_kwargs: FirstMention(
        None, None, None, ((2024, 5), (2025, 20))))
    pool = [item(300 + index, f"Sulfide solid electrolyte device {index}",
                 ("crossref", "arxiv")[index % 2]) for index in range(35)]
    pool += [item(400, "Sulfide solid electrolyte device 0", "europe_pmc"),
             item(401, "Unrelated photonic chip study", "crossref")]
    candidate = Candidate("sulfide solid electrolyte", "Sulfide solid electrolyte", (), ())
    entry = pipeline._evaluate(candidate, pool, "батареи", AS_OF,
                               SignalModel.load(pipeline.MODEL_PATH), None, pipeline.Event(), 24,
                               years=((2024, 5), (2025, 20)))
    assert len(entry["sources"]) == pipeline.TOP_SOURCE_LIMIT
    assert len({source["url"] for source in entry["sources"]}) == pipeline.TOP_SOURCE_LIMIT
    assert len({source["title"] for source in entry["sources"]}) == pipeline.TOP_SOURCE_LIMIT
    assert all("sulfide solid electrolyte" in source["title"].casefold() for source in entry["sources"])
    assert all(source["url"] != "https://example.org/401" for source in entry["sources"])
    assert len(_technology(entry, {}).sources) == pipeline.TOP_SOURCE_LIMIT


def test_only_the_fastest_growing_candidates_get_a_monthly_history(monkeypatch):
    checked = []

    def counted(forms, as_of, **options):
        checked.append(forms[0])
        return fake_history(forms, as_of, **options)

    monkeypatch.setattr(pipeline, "collect_history", counted)
    monkeypatch.setattr(pipeline, "openalex_years", fake_years)
    monkeypatch.setattr(pipeline, "MAX_DEEP", 1)
    extra = [item(30 + index, f"Garnet oxide electrolyte {name}", source)
             for index, (name, source) in enumerate((("films", "crossref"), ("coatings", "arxiv"),
                                                     ("interfaces", "europe_pmc")))]
    radar = pipeline.build_radar(POOL + extra, query="батареи", query_terms=["solid-state batteries"], as_of=AS_OF)
    assert radar["candidates_total"] == 2
    assert len(checked) == 1
    skipped = [entry for entry in radar["excluded"] if pipeline.NOT_CHECKED in entry["reasons"]]
    assert skipped and all(entry["curve"]["materials"] == 0 for entry in skipped)


def test_openalex_refusal_is_retried_instead_of_falling_back(monkeypatch):
    import httpx

    from app.radar import evidence
    from app.radar.evidence import openalex_years

    monkeypatch.setattr(evidence, "OPENALEX_INTERVAL_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(request.url.params["filter"])
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"group_by": [{"key": "2025", "count": 900}, {"key": "1998", "count": 5}]})

    from threading import Event

    years = openalex_years(httpx.Client(transport=httpx.MockTransport(handler)), "human security", Event())
    assert years == ((1998, 5), (2025, 900)) and len(calls) == 2
    assert calls[0] == 'title.search:"human security"'


def test_phrase_with_a_few_works_is_not_a_signal():
    sparse = curve_of([0] * 20 + [1, 1, 1, 1])
    evidence = pool_evidence("sulfide solid electrolyte", POOL)
    passport = build_passport("traditional security", "ИИ", AS_OF, sparse,
                              FirstMention(None, None, None, ((2025, 4),)), evidence, None)
    prediction = SignalModel.load(pipeline.MODEL_PATH).predict([passport.technology])[0]
    assert not is_weak_signal(passport, prediction, sparse)
    assert "Мало работ для вывода: 4 за 24 месяца" in exclusion_reasons(passport, sparse, prediction)


def test_refused_history_keeps_the_candidate_below_signals_instead_of_excluding_it():
    months = month_window(AS_OF)
    refused = assess_curve([Material("1", "arxiv", months[-1]), Material("2", "arxiv", months[-2])], AS_OF,
                           coverage_complete=False)
    assert refused.confidence is None
    passport = build_passport("3d convolutional neural networks", "ИИ", AS_OF, refused,
                              FirstMention(None, None, None), pool_evidence("sulfide solid electrolyte", POOL), None)
    confident = Prediction("x", 0.99, True, ())
    # The limit, not the topic, left the curve empty: not excluded, but never a signal without history.
    assert not excluded_by_rule(passport, refused)
    assert not is_weak_signal(passport, confident, refused)
    assert any(reason.endswith("история неполная: источник отказал")
               for reason in exclusion_reasons(passport, refused, confident))


def test_declining_topic_is_excluded_even_if_the_model_is_confident():
    falling = curve_of([10] * 12 + [0] * 12)
    assert falling.trend == "снижается"
    passport = build_passport("sulfide solid electrolyte", "Батареи", AS_OF, falling,
                              FirstMention(None, None, None, ((2024, 5), (2025, 20))),
                              pool_evidence("sulfide solid electrolyte", POOL), None)
    assert excluded_by_rule(passport, falling)


def test_quick_exclusion_reason_is_not_repeated_and_unchecked_history_is_not_blamed():
    assert pipeline._reasons("Зрелая тема: 9 работ во всей науке",
                             ["Зрелая тема: 9 работ за всё время", "Мало работ для вывода: 0 за 24 месяца",
                              "Стадия развития: массовое/зрелое"], deep=False) == [
        "Зрелая тема: 9 работ во всей науке", "Стадия развития: массовое/зрелое"]
    assert pipeline._reasons(None, ["Мало работ для вывода: 3 за 24 месяца"], deep=True) == [
        "Мало работ для вывода: 3 за 24 месяца"]


def test_modifier_fragments_are_not_candidates():
    pool = [item(40 + index, f"Large language model-based {tail}", source) for index, (tail, source) in
            enumerate((("planning", "crossref"), ("coding", "arxiv"), ("search", "europe_pmc")))]
    phrases = [candidate.surface for candidate in mine_candidates(pool, query_terms=["agents"])]
    assert "large language model-based" not in phrases and "Large language model-based" not in phrases


def test_top_takes_only_probabilities_above_sixty_percent():
    from app.radar.passport import TOP_PROBABILITY
    from app.signal_model.model import Prediction

    growing = curve_of([round(0.04 * index * index) + 1 for index in range(24)])
    passport = build_passport("sulfide solid electrolyte", "Батареи", AS_OF, growing,
                              FirstMention(None, None, None, ((2024, 5), (2025, 20))),
                              pool_evidence("sulfide solid electrolyte", POOL), None)
    assert TOP_PROBABILITY == 0.6
    assert not is_weak_signal(passport, Prediction("x", 0.6, True, ()), growing)
    assert is_weak_signal(passport, Prediction("x", 0.61, True, ()), growing)


def test_signals_beyond_fifteen_go_to_the_journal_with_a_reason(monkeypatch):
    monkeypatch.setattr(pipeline, "collect_history", fake_history)
    monkeypatch.setattr(pipeline, "openalex_years", fake_years)
    monkeypatch.setattr(pipeline, "TOP_SIZE", 1)
    extra = [item(50 + index, f"Garnet oxide electrolyte {name}", source)
             for index, (name, source) in enumerate((("films", "crossref"), ("coatings", "arxiv"),
                                                     ("interfaces", "europe_pmc")))]
    radar = pipeline.build_radar(POOL + extra, query="батареи", query_terms=["solid-state batteries"], as_of=AS_OF)
    assert len(radar["technologies"]) == 1
    beyond = [entry for entry in radar["excluded"] if entry["reasons"] == [pipeline.BELOW_TOP]]
    # Ranking overflow does not change the model's weak-signal verdict.
    assert beyond and beyond[0]["is_signal"] and not beyond[0]["rule_excluded"]


def test_a_repeated_analysis_reuses_complete_phrase_answers_only(monkeypatch):
    histories, years = [], []

    def history(forms, as_of, **options):
        histories.append(forms[0].lower())
        found = fake_history(forms, as_of, **options)
        if forms[0].lower().startswith("sulfide"):
            # A refused source is lost coverage, not a fact worth remembering.
            return TechnologyHistory(found.aliases, as_of, found.months, found.records,
                                     (SourceCoverage("arxiv", "unavailable", 0, 0, "rate_limited"),))
        return found

    def counted_years(client, phrase, cancel, key=None):
        years.append(phrase.lower())
        return None if phrase.lower().startswith("halide") else fake_years(client, phrase, cancel, key)

    monkeypatch.setattr(pipeline, "collect_history", history)
    monkeypatch.setattr(pipeline, "openalex_years", counted_years)
    monkeypatch.setattr(pipeline, "first_mention", lambda *_args, **_kwargs: FirstMention(None, None, None))
    pool = POOL + [item(20, "Halide solid electrolytes in cells", "crossref"),
                   item(21, "Halide solid electrolytes at scale", "europe_pmc")]
    cache = pipeline.PhraseCache()

    def run(as_of=AS_OF):
        return pipeline.build_radar(pool, query="батареи", query_terms=["solid-state batteries"], as_of=as_of,
                                    cache=cache)

    first = run()
    asked, checked = sorted(years), sorted(histories)
    assert {"halide", "sulfide"} <= {phrase.split()[0] for phrase in asked}
    again = run()
    assert [entry["title"] for entry in again["technologies"]] == [entry["title"] for entry in first["technologies"]]
    # Only refused answers are asked again: OpenAlex for one phrase, history for the other.
    assert sorted(years[len(asked):]) == [phrase for phrase in asked if phrase.startswith("halide")]
    assert sorted(histories[len(checked):]) == [form for form in checked if form.startswith("sulfide")]
    # Another cutoff date is another question.
    repeated = len(years)
    run(date(2026, 9, 27))
    assert sorted(years[repeated:]) == asked


def test_phrase_cache_forgets_old_and_overflowing_answers(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pipeline, "monotonic", lambda: clock[0])
    cache = pipeline.PhraseCache(seconds=10, limit=2)
    calls = []

    def compute(value):
        return lambda: calls.append(value) or value

    for key in ("a", "b", "a"):
        cache.get(key, compute(key), lambda _: True)
    assert calls == ["a", "b"]
    cache.get("c", compute("c"), lambda _: True)  # The oldest answer leaves.
    cache.get("a", compute("a"), lambda _: True)
    assert calls == ["a", "b", "c", "a"]
    clock[0] = 11.0
    cache.get("c", compute("c"), lambda _: True)
    assert calls[-1] == "c" and len(calls) == 5


def test_a_long_openalex_refusal_closes_it_instead_of_waiting_for_every_phrase(monkeypatch):
    import httpx
    from threading import Event

    from app.radar import evidence

    monkeypatch.setattr(evidence, "_OPENALEX_CLOSED_UNTIL", [0.0])
    monkeypatch.setattr(evidence, "OPENALEX_INTERVAL_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(request.url.params["filter"])
        # Дневной бюджет без ключа исчерпан: прийти через час.
        return httpx.Response(429, headers={"Retry-After": "3486"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert evidence.openalex_years(client, "green hydrogen", Event()) is None
    assert evidence.openalex_closed_seconds() > 3000
    # Остальные фразы не спрашиваются и не ждут.
    assert evidence.openalex_years(client, "proton exchange membrane", Event()) is None
    assert len(calls) == 1
