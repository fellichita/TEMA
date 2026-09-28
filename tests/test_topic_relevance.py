"""Точное попадание в тему: профиль темы, оценка пула, ТОП по релевантности, радар."""

from __future__ import annotations

from datetime import date
import hashlib

from app.pilot.publication_relevance import assess_pool, relevance_items
from app.relevance_learning import RelevanceModel
from app.topic_relevance import TopicProfile, combine, filter_pool, lexical_match, stem, topic_match
from app.web_api import radar_input, relevance_ranking, web_result
from tests.owner_fixture import FakeEncoder, make_payload

PROFILE = TopicProfile.build("твердотельные аккумуляторы", "solid-state batteries",
                             synonyms=["solid electrolyte battery"], subdirections=["sulfide solid electrolytes"],
                             exclusions=["lead-acid battery"])


def test_stems_join_word_forms_in_both_languages():
    assert stem("батареи") == stem("батарея")
    assert stem("аккумуляторы") == stem("аккумулятор")
    assert stem("batteries") == stem("battery") and stem("sensors") == stem("sensor")


def test_a_single_common_word_no_longer_admits_a_story():
    # Прежний фильтр принимал любой материал со словом «state».
    assert not topic_match(PROFILE, "State of the Union address")
    assert not topic_match(PROFILE, "Battery state of charge estimation with a Kalman filter")
    assert topic_match(PROFILE, "Solid-state battery plant opens in Texas")
    assert topic_match(PROFILE, "Samsung unveils prototype", "A new solid-state battery reaches 900 Wh/L")
    assert topic_match(PROFILE, "Твердотельный аккумулятор для электромобилей")
    assert topic_match(PROFILE, "Sulfide electrolytes for all-solid-state lithium batteries")
    assert not topic_match(PROFILE, "Новые аккумуляторы для смартфонов")
    # Исключённая область плана: совпадение слов не спасает материал.
    assert not topic_match(PROFILE, "Lead-acid battery recycling", "solid waste")
    # Тему из одних общих слов по словам не отличить: такую ленту фильтр не режет.
    assert topic_match(TopicProfile.build("technology"), "Robotic battery")


def test_lexical_match_reports_phrase_and_coverage():
    match = lexical_match(PROFILE, "A solid-state battery prototype", None)
    assert match.phrase_title and match.score == 1.0
    partial = lexical_match(PROFILE, "Battery chemistry update", None)
    assert 0 < partial.score < 0.67 and not partial.phrase_text


def test_semantics_and_words_decide_together():
    assert combine(0.0, 0.84)[1] == "relevant"
    assert combine(0.0, 0.74)[1] == "off_topic"
    assert combine(1.0, 0.78)[1] == "relevant"
    assert combine(0.6, None)[1] == "relevant" and combine(0.1, None)[1] == "off_topic"
    blended, _ = combine(0.5, None, learned=1.0, learned_weight=0.5)
    assert blended == 0.75


def _pool():
    payload = make_payload(1, "квантовые сенсоры")
    from app.pilot.approved_sources import SourceSnapshot
    from app.web_api import _publication_pool

    snapshot = SourceSnapshot.model_validate(payload["approved_sources"])
    return payload, _publication_pool(payload["result"], snapshot, None, keep_studies=True)


def test_pool_assessment_uses_discovery_scores_and_encodes_the_rest():
    payload, pool = _pool()
    plan = payload["result"]["query_plan"]
    encoder = FakeEncoder(["quantum sens", "nitrogen-vacancy"])
    assessed = assess_pool(pool, plan, encoder=encoder)
    items = relevance_items(assessed)
    assert items is not None and len(items) == len(pool)
    decisions = {item["title"]: items[item["publication_id"]][1] for item in pool}
    assert all(decision == "relevant" for title, decision in decisions.items() if "advance" in title)
    assert all(decision == "off_topic" for title, decision in decisions.items() if "advance" not in title)
    assert assessed["counts"]["off_topic"] == sum(1 for title in decisions if "advance" not in title)
    # Научный документ получает оценку этапа обнаружения, не новый расчёт.
    study_item = dict(pool[0], study_ids=["study-1"], publication_id="f" * 64)
    graded = assess_pool([study_item], plan, discovery={"study-1": (0.9, "retained")}, encoder=None)
    assert graded["items"]["f" * 64][3] == 0.9 and graded["items"]["f" * 64][1] == "r"
    excluded = assess_pool([study_item], plan, discovery={"study-1": (0.9, "closer_to_excluded_scope")})
    assert excluded["items"]["f" * 64][1] == "o"


def test_web_top_is_chosen_by_relevance_and_hides_off_topic():
    payload, pool = _pool()
    before = web_result(payload)
    # Без оценки темы (старые анализы) ТОП — прежний: самые свежие записи.
    assert [item["title"] for item in before["top_publications"]][:1] == [pool[0]["title"]]
    assert any("advance" not in item["title"] for item in before["top_publications"])
    payload["publication_relevance"] = assess_pool(pool, payload["result"]["query_plan"],
                                                   encoder=FakeEncoder(["quantum sens", "nitrogen-vacancy"]))
    after = web_result(payload)
    assert after["off_topic_total"] > 0
    assert after["publication_total"] == before["publication_total"] - after["off_topic_total"]
    assert all("advance" in item["title"] for item in after["top_publications"])
    assert all(item["relevance"]["decision"] == "relevant" for item in after["top_publications"])
    sources = [item["source_id"] for item in after["top_publications"]]
    assert max(sources.count(source) for source in set(sources)) <= 6
    radar_pool, _, _ = radar_input(payload)
    assert all("advance" in item["title"] for item in radar_pool)


def test_ranking_prefers_relevance_then_freshness():
    def publication(number, days, source="arxiv"):
        return {"publication_id": hashlib.sha256(str(number).encode()).hexdigest(), "title": str(number),
                "url": f"https://example.org/{number}", "source_id": source,
                "published_at": date.fromordinal(date(2026, 9, 1).toordinal() - days).isoformat()}

    items = [publication(1, 1), publication(2, 900), publication(3, 5)]
    relevance = {items[0]["publication_id"]: (0.5, "weak", 0.5, None),
                 items[1]["publication_id"]: (0.9, "relevant", 1.0, 0.9),
                 items[2]["publication_id"]: (0.2, "off_topic", 0.0, 0.7)}
    ranked, removed = relevance_ranking(items, relevance, date(2026, 9, 1))
    assert removed == 1 and [item["title"] for item in ranked] == ["2", "1"]


def test_radar_filter_keeps_topic_and_falls_back_only_without_overlap():
    pool = [{"title": f"Quantum sensor {index}", "summary": None} for index in range(30)]
    pool += [{"title": f"Football news {index}", "summary": None} for index in range(30)]
    profile = TopicProfile.build("квантовые сенсоры", "quantum sensors")
    kept = filter_pool(pool, profile)
    assert len(kept) == 30 and all(item["title"].startswith("Quantum") for item in kept)
    assert filter_pool(pool[30:], profile) == pool[30:]  # Нет совпадений: выдачу не обнуляем.
    model = RelevanceModel(weights={"bias": -5.0}, blend=0.3)
    stricter = filter_pool(pool, profile, scorer=model.scorer(profile))
    assert all(item["title"].startswith("Quantum") for item in stricter)


def test_learned_scorer_is_off_until_the_model_is_trusted():
    assert RelevanceModel().scorer(PROFILE) is None
    assert RelevanceModel(weights={"bias": 1.0}, blend=0.0).scorer(PROFILE) is None
