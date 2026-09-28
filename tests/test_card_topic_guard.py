"""Card-level axes: short labels, source fallback and selection before TOP-K."""

from copy import deepcopy
import json

import pytest

from app.ml import engine, text
from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from tests.mvp_fixture import research_groups, snapshot


TOPIC = "photonic neuromorphic computing"


@pytest.mark.parametrize("terms,expected", [
    (["photonic", "neural", "integrated"], True),
    (["optical", "kerr", "bandwidth", "communications"], False),
    (["data", "transmission", "kerr", "microcombs"], False),
    (["reservoir", "computing", "photonic"], True),
    (["neural", "network", "training"], False),
    (["photonic synapses"], True),
    (["silicon microring", "microring resonator", "silicon photonics"], False),
    (["optoelectronic synaptic", "synaptic transistor"], True),
    (["photonic-electronic spiking"], True),
    (["micro-combs", "perceptrons"], True),
    (["MICROCOMBS", "SNNs"], True),
    (["VCSELs", "inference"], True),
    (["MZIs", "MAC"], True),
    (["waveguides", "convolutional"], True),
    (["interferometers", "learning"], True),
    (["lasers", "memristive"], True),
    (["photoelectric", "synapse"], True),
    (["nonneural", "optical"], False),
    (["neural", "geophotonics"], False),
    (["machining", "optical"], False),
    ([], False),
    (["", "  "], False),
])
def test_lexical_axes_are_explicit_and_match_whole_words(terms, expected):
    assert text.topic_axis_guard(terms) is expected


@pytest.mark.parametrize("separator", ["-", "‐", "‑", "–", "—", " ", "\n\t"])
def test_axis_hyphens_and_whitespace_are_equivalent(separator):
    assert text.topic_axis_guard([f"MICRO{separator}COMBS", f"PHOTONIC{separator}ELECTRONIC", "SYNAPSES"])


def study(title, abstract, identifier="source-1"):
    return {"id": identifier, "title": title, "abstract": abstract,
            "url": f"https://example.org/{identifier}"}


DIRECT = study("Device measurements", "Our optical chip performs neural inference. "
               "We report measurements under repeated operation.")
COMMUNICATIONS = study("Optical Kerr microcombs for bandwidth communications",
                      "They have enabled breakthroughs in spectroscopy, microwave photonics, "
                      "optical neuromorphic processing and more. "
                      "We demonstrate optical data transmission at high capacity.")


@pytest.mark.parametrize("terms,sources,expected", [
    (["photonic neural"], [DIRECT], "supported"),
    (["silicon microring"], [DIRECT], "partial"),
    (["silicon microring"], [], "partial"),
    (["optical communications"], [COMMUNICATIONS], "off_direction"),
])
def test_axis_check_has_its_own_vocabulary(terms, sources, expected):
    result = text.card_topic_guard(TOPIC, terms, sources)
    assert result["axis_check"] == expected
    assert "status" not in result


@pytest.mark.parametrize("terms", [
    ["silicon microring", "microring resonator", "silicon photonics"],
    ["term plasticity", "synaptic device", "neuromorphic computing"],
    ["phase change", "change materials"],
])
@pytest.mark.parametrize("source", [
    DIRECT,
    study("Silicon microring resonators for photonic neural networks", "The optical chip is characterized."),
    study("Device measurements", "Neural inference is implemented in an optical circuit."),
    study("Device measurements", "We fabricate an optical chip. This chip performs neural inference."),
    study("Device measurements", "We train parameters on a GPU while an optical chip performs neural inference."),
    study("Device measurements", "Neural weighted sums are computed in our processor. "
          "This processor uses optical pulses as signals."),
])
def test_missing_label_axis_uses_documents_without_excluding_real_technology(terms, source):
    original = deepcopy(source)
    result = text.card_topic_guard(TOPIC, terms, [source])
    assert result["axis_check"] == "partial"
    assert result["reason"] == "document_support_for_missing_cluster_axis"
    assert result["supporting_documents"][0]["study_id"] == source["id"]
    assert result["supporting_documents"][0]["url"] == source["url"]
    assert source == original


def test_background_application_list_does_not_rescue_an_unrelated_label():
    result = text.card_topic_guard(TOPIC, ["optical kerr", "bandwidth communications"], [COMMUNICATIONS])
    assert result["axis_check"] == "off_direction"
    assert result["reason"] == "no_document_support_for_missing_cluster_axis"
    assert result["cluster_matches"]["carrier"] == ["optical"]
    assert result["cluster_matches"]["compute"] == []
    assert result["supporting_documents"] == []


def test_an_action_in_the_previous_sentence_does_not_rescue_a_background_list():
    source = study("Device measurements", "We demonstrate an optical chip for data transmission. "
                   "This chip has applications in optical neuromorphic processing.")
    result = text.card_topic_guard(TOPIC, ["optical communications"], [source])
    assert result["axis_check"] == "off_direction"


def test_a_real_operation_in_the_next_sentence_still_rescues_the_label():
    source = study("Device measurements", "We demonstrate an optical chip for data transmission. "
                   "This chip performs neural inference.")
    assert text.card_topic_guard(TOPIC, ["optical communications"], [source])["axis_check"] == "partial"


def test_one_missing_title_axis_does_not_combine_unrelated_documents():
    sources = [study("Optical communication devices", "We measure optical data transmission."),
               study("Neural networks for classification", "A GPU performs neural inference.", "source-2")]
    result = text.card_topic_guard(TOPIC, ["bandwidth", "communications"], sources)
    assert result["axis_check"] == "off_direction"


def test_title_cannot_rescue_a_document_rejected_by_existing_scope_rules():
    source = study("Photonic neural networks", "Our optical device does not implement neural computation.")
    assert text.card_topic_guard(TOPIC, ["optical device"], [source])["axis_check"] == "off_direction"


def test_every_cluster_document_is_checked_including_after_the_first_twelve():
    sources = [study(COMMUNICATIONS["title"], COMMUNICATIONS["abstract"], f"other-{i}") for i in range(13)]
    sources.append(DIRECT)
    result = text.card_topic_guard(TOPIC, ["silicon microring"], sources)
    assert result["axis_check"] == "partial"
    assert result["checked_documents"] == 14
    assert [s["study_id"] for s in result["supporting_documents"]] == [DIRECT["id"]]


@pytest.mark.parametrize("sources", [[], [study("Unclassified material", "")]])
def test_missing_source_text_is_uncertainty_not_a_negative_finding(sources):
    result = text.card_topic_guard(TOPIC, ["silicon microring"], sources)
    assert result["axis_check"] == "partial"
    assert result["reason"] == "insufficient_document_text"


@pytest.mark.parametrize("topic", [TOPIC, "фотонные нейросети", "фотонные нейроморфные вычисления"])
def test_photonic_topic_aliases_activate_the_same_guard(topic):
    assert text.card_topic_guard(topic, ["photonic neural"], [DIRECT])["axis_check"] == "supported"


@pytest.mark.parametrize("topic", ["artificial intelligence", "ИИ", "технологии в ИИ",
                                  "quantum computing", "dna data storage", "optical communications"])
def test_other_directions_do_not_use_photonic_axes(topic):
    assert text.card_topic_guard(topic, ["neural networks"], [DIRECT]) is None


def test_real_pipeline_excludes_before_top_k_and_exports_the_original_candidate(monkeypatch):
    corpus = unpack_snapshot(snapshot())
    options = AnalysisOptions(topic=TOPIC, top_k=1)
    baseline = engine.analyze(corpus, options)
    rejected_id = research_groups(baseline)[0]["id"]
    rejected_ids = set(research_groups(baseline)[0]["study_ids"])
    real_guard = text.card_topic_guard

    def reject_one(topic, terms, members):
        result = real_guard(topic, terms, members)
        if {s["id"] for s in members} == rejected_ids:
            result.update(axis_check="off_direction", reason="labelled_test_negative", supporting_documents=[])
        return result

    monkeypatch.setattr(engine, "card_topic_guard", reject_one)
    result = engine.analyze(corpus, options)
    assert len(research_groups(result)) == len(research_groups(baseline)) - 1
    assert all(c["id"] != rejected_id for c in research_groups(result))
    assert len(result["excluded_off_direction"]) == 1
    excluded = result["excluded_off_direction"][0]
    assert excluded["id"] == rejected_id and excluded["status"] == "off_direction"
    assert excluded["direction_guard"]["reason"] == "labelled_test_negative"
    assert excluded["direction_guard"]["axis_check"] == "off_direction"
    for candidate in research_groups(result) + result["excluded_off_direction"]:
        assert "status" not in candidate["direction_guard"]
        assert candidate["stage"] == "requires_review"
    for key in ["metrics", "card", "sources", "study_ids", "keywords"]:
        assert excluded[key] == research_groups(baseline)[0][key]
    assert result["preparation"] == baseline["preparation"]
    assert result["direction_counts"] == baseline["direction_counts"]
    assert result == json.loads(json.dumps(result, allow_nan=False))


def test_excluded_list_is_present_when_there_are_no_candidates():
    corpus = unpack_snapshot(snapshot())
    corpus["entries"] = []
    result = engine.analyze(corpus, AnalysisOptions(topic=TOPIC))
    assert result["candidates"] == []
    assert result["excluded_off_direction"] == []


def test_real_nmf_keeps_photonic_groups_and_excludes_communications_without_a_stub():
    data = snapshot()
    bad_ids = set()
    for batch in data["batches"]:
        for entry in batch["documents"]:
            doc = entry["document"]
            if doc["source_id"].split("-")[1] == "0":
                bad_ids.add(entry["document_key"])
                doc["title"] = COMMUNICATIONS["title"] + ": experimental device " + doc["source_id"]
                doc["abstract"] = (COMMUNICATIONS["abstract"] + " Existing links face a bandwidth bottleneck. "
                                   "Our links improve transmission efficiency and reduce signal loss.")
    corpus = unpack_snapshot(data)
    original = deepcopy(corpus)
    options = AnalysisOptions(topic=TOPIC)
    result = engine.analyze(corpus, options)
    assert corpus == original
    assert result == engine.analyze(corpus, options)
    assert len(research_groups(result)) == 2
    assert all(not bad_ids.intersection(c["study_ids"]) for c in research_groups(result))
    assert len(result["excluded_off_direction"]) == 1
    excluded = result["excluded_off_direction"][0]
    assert set(excluded["study_ids"]) == bad_ids
    assert excluded["status"] == "off_direction"
    assert excluded["direction_guard"]["checked_documents"] == len(bad_ids)
    assert all(excluded["card"].values())
    # These publications remain in the unmodified corpus/model denominator.
    prepared, preparation = engine.prepare(corpus["entries"], options)
    assert result["preparation"] == preparation
    assert bad_ids <= {s["id"] for s in prepared}
    assert sum(row["documents"] for row in result["direction_counts"]) == len(prepared)


def test_all_excluded_is_a_valid_empty_result(monkeypatch):
    def reject(topic, terms, members):
        return {"axis_check": "off_direction", "reason": "labelled_test_negative"}
    monkeypatch.setattr(engine, "card_topic_guard", reject)
    result = engine.analyze(unpack_snapshot(snapshot()), AnalysisOptions(topic=TOPIC))
    assert result["status"] == "no_groups"
    assert result["candidates"] == []
    assert result["excluded_off_direction"]
    assert all(c["status"] == "off_direction" for c in result["excluded_off_direction"])


def test_off_direction_reason_is_retained_even_when_a_card_lacks_evidence(monkeypatch):
    data = snapshot()
    for batch in data["batches"]:
        for entry in batch["documents"]:
            entry["document"]["abstract"] = ("We demonstrate photonic neural computation in optical devices. "
                "The experiment uses repeated measurements of " + entry["document"]["title"] + ".")
    monkeypatch.setattr(engine, "card_topic_guard", lambda *args: {"axis_check": "off_direction", "reason": "test"})
    result = engine.analyze(unpack_snapshot(data), AnalysisOptions(topic=TOPIC))
    assert result["excluded_off_direction"]
    assert all(c["card"]["advantage"] is None for c in result["excluded_off_direction"])


def test_source_support_does_not_rewrite_passages_or_card_quotes():
    source = study("Silicon microring resonator", "Our optical chip performs neural inference. "
                   "Existing devices face a bandwidth bottleneck. "
                   "Our device reduces latency without additional cooling.")
    before = text.evidence_card([source])
    result = text.card_topic_guard(TOPIC, ["silicon microring"], [source])
    assert before == text.evidence_card([source])
    assert result["supporting_documents"][0]["text"] in text.sentences(source["abstract"])
    assert result["supporting_documents"][0]["mode"] == "source_excerpt"
