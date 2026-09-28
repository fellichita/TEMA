"""Annotated benefit assertions: negation scope, fallback and intact source quotes."""

from copy import deepcopy
import random

import pytest

from app.ml.text import clean, evidence_card


NEGATIVE_PASSAGES = [
    "Our device reduces neither latency nor power consumption under the tested conditions.",
    "Our device fails, however, to improve energy efficiency under repeated testing.",
    "An improvement in latency was ruled out by the measurements.",
    "Our device failed, despite the revised fabrication process, to improve energy efficiency.",
    "Our device fails (even after repeated calibration) to reduce the measured latency.",
    "Our device is unable, in these experiments, to outperform the baseline implementation.",
    "Our device does not, under the tested conditions, improve energy efficiency.",
    "Our device does not, even after repeated calibration across all of the operating conditions, improve latency.",
    "Our device never, even after calibration, achieved the expected processing speed.",
    "Our device doesn't improve energy efficiency under these conditions.",
    "Our device doesn’t, however, improve energy efficiency under these conditions.",
    "Neither architecture improves energy efficiency under the tested conditions.",
    "No statistically significant improvement in latency was found in our measurements.",
    "We found no evidence of improved energy efficiency in the proposed architecture.",
    "We observed a lack of improvement in energy efficiency during repeated testing.",
    "The failure to reduce latency was reproduced across all the tested devices.",
    "The measurements rule out an improvement in the energy efficiency of the device.",
    "The measurements ruled out any improvement in the energy efficiency of the device.",
    "An improvement in latency has been ruled out by the measurements.",
    "The improvement in latency, however, was ruled out by the measurements.",
    "The improvement in latency was, however, ruled out by the measurements.",
    "The improvement in latency could not be confirmed by the measurements.",
    "The improvement in latency was not, despite repeated testing, observed in our measurements.",
    "The improvement in latency was never experimentally confirmed by our measurements.",
    "The improvement in latency was neither observed nor reproduced in our measurements.",
    "The improvement in latency was absent from all of the measured devices.",
    "The improvement in latency remains unconfirmed after the repeated measurements.",
    "The improvement in latency was not statistically significant in these experiments.",
    "The improved efficiency could not be achieved in any of the tested devices.",
    "The improved efficiency was not achieved in any of the tested devices.",
    "The device is not efficient under the operating conditions used in our experiments.",
    "The device is efficient neither at room temperature nor at lower temperatures.",
    "The low-power operation was not demonstrated in any of the repeated experiments.",
    "The reported low-latency operation was ruled out by independent measurements.",
    "Our device reduces no measurable component of the latency under the tested conditions.",
    "Our device reduces, however, neither latency nor power consumption under the tested conditions.",
    "Our device improves neither energy efficiency nor classification accuracy during testing.",
    "We conducted the experiment without improving the efficiency of the optical device.",
    "The device improves latency, but does not reduce its energy consumption.",
    "The device improves latency; the improved efficiency was not observed experimentally.",
    "The device improves latency, whereas the improvement in accuracy was ruled out experimentally.",
    "Our device fails after repeated calibration to improve energy efficiency.",
    "Our device is unable even after repeated calibration to reduce latency.",
    "Our device failed to stabilize and reduce the latency under repeated operation.",
    "Our device cannot be said to improve the latency under repeated operation.",
    "Our device fails, however to improve energy efficiency during the repeated experiments.",
    "The improvement in latency, which was not statistically significant, was observed during testing.",
    "The device reduces latency by no measurable amount in the repeated experiments.",
    "The claimed improvement in latency is not supported by our measurements.",
    "The NOT gate does not improve energy efficiency under repeated operation.",
    "We require an improvement in latency before the device becomes practical.",
    "Low-power operation is necessary to make the optical device practical.",
    "Further research is essential to overcome the limitations, paving the way for more efficient devices.",
    "In this work, sensors must be portable, more energy-efficient, and more accurate than existing devices.",
    "Their widespread adoption is limited by the absence of efficient optical nonlinear activation units.",
    "Optical neural networks are promising; however, they are limited by the lack of energy-efficient nonlinear functions.",
]

POSITIVE_PASSAGES = [
    "Our device improves energy efficiency and reduces latency under the tested conditions.",
    "Our device not only improves efficiency but also reduces latency.",
    "Our device does not only improve efficiency but also reduces latency.",
    "Our device not just improves efficiency but also reduces latency.",
    "Our device not merely improves efficiency but also reduces latency.",
    "Our device not only, under the tested conditions, improves efficiency but also reduces latency.",
    "Our device reduces latency without requiring additional optical components.",
    "Our device reduces latency, without requiring any additional optical components.",
    "Our device improves energy efficiency without compromising the stability of the output.",
    "Without additional cooling, our device improves energy efficiency under repeated operation.",
    "No additional cooling is needed, and our device improves energy efficiency in repeated tests.",
    "The device reduces latency, but does not require additional optical components.",
    "The device reduces latency while no thermal drift was observed during testing.",
    "The device improves energy efficiency, and no thermal drift was observed during testing.",
    "The device improves energy efficiency, and the thermal drift was not observed during testing.",
    "The device reduces latency without additional cooling and improves classification accuracy.",
    "The device improves energy efficiency, however, its size remains unchanged during testing.",
    "An improvement in latency was confirmed by all of the repeated measurements.",
    "The improvement in latency was, however, experimentally confirmed by our measurements.",
    "The device improves efficiency by no less than twenty percent under the tested conditions.",
    "Our device achieves a latency of no more than five nanoseconds in repeated tests.",
    "The device is not only efficient but also stable under the tested operating conditions.",
    "The device is efficient and requires no additional cooling during repeated operation.",
    "An improvement in efficiency was observed after the calibration failure was corrected.",
    "The device improves efficiency after thermal instability was ruled out by our measurements.",
    "A power reduction was measured after the control circuit failed during repeated testing.",
    "The device achieves low-power operation without errors during the repeated measurements.",
    "Our device reduces latency without reducing the measured classification accuracy.",
    "Our device improves latency without failing to meet the stability requirement.",
    "Unlike devices that failed during testing, our device improves energy efficiency.",
    "After the baseline failed, our device improves the measured energy efficiency.",
    "The device reduces latency and does not require additional optical components.",
    "The device improves latency and is not affected by thermal noise.",
    "The device improves latency, not only in simulation but also in repeated measurements.",
    "The device not only improves latency, but also enhances accuracy without extra cooling.",
    "The laser operates without external perturbations, forming a simple and energy-efficient core.",
    "Without digital post-processing, the device reduces circuit complexity and power consumption.",
    "The circuit produces NOR and NOT operations and improves energy efficiency during repeated operation.",
    "The device improves accuracy after thermal drift was ruled out by independent measurements.",
    "The small energy barrier reduces the chance of element segregation-associated device failure.",
    "The approach is not restricted to a specific medium, thereby enhancing its applicability.",
    "Furthermore, the flexible synaptic transistors exhibit no apparent synaptic performance degradation "
    "even when the bending radius is reduced to 1 mm.",
    "Our approach does not require fine-tuning or refined knowledge of the setup, "
    "at the same time outperforming conventional approaches.",
]


def study(abstract, identifier="source"):
    return {"id": identifier, "title": "Measured photonic neural device " + identifier,
            "abstract": abstract, "url": "https://example.org/" + identifier}


@pytest.mark.parametrize("passage", NEGATIVE_PASSAGES)
def test_denied_benefit_is_not_an_advantage(passage):
    assert evidence_card([study(passage)])["advantage"] is None


@pytest.mark.parametrize("passage", POSITIVE_PASSAGES)
def test_supported_benefit_remains_an_exact_source_quote(passage):
    source = study(passage)
    assert evidence_card([source])["advantage"] == {
        "text": passage, "study_id": source["id"], "title": source["title"],
        "url": source["url"], "mode": "source_excerpt"}


@pytest.mark.parametrize("passage", NEGATIVE_PASSAGES[:3])
@pytest.mark.parametrize("another_source", [False, True])
def test_denied_benefit_falls_back_to_supported_quote(passage, another_source):
    positive = "Our device reduces latency without requiring additional optical components."
    sources = ([study(passage, "negative"), study(positive, "positive")] if another_source else
               [study(passage + " " + positive, "combined")])
    original = deepcopy(sources)
    card = evidence_card(sources)
    assert card["advantage"] == {
        "text": positive, "study_id": sources[-1]["id"], "title": sources[-1]["title"],
        "url": sources[-1]["url"], "mode": "source_excerpt"}
    assert sources == original


def test_all_denied_sources_leave_advantage_empty_without_changing_other_fields():
    problem = "The optical device suffers from a limited bandwidth during repeated operation."
    example = "We demonstrate an optical circuit with a stable response under repeated operation."
    sources = [study(problem + " " + example + " " + NEGATIVE_PASSAGES[0], "first"),
               study(NEGATIVE_PASSAGES[1], "second"), study(NEGATIVE_PASSAGES[2], "third")]
    original = deepcopy(sources)
    card = evidence_card(sources)
    assert card["advantage"] is None
    assert card["problem"]["text"] == problem
    assert card["example"]["text"] == example
    assert sources == original


@pytest.mark.parametrize("passage", NEGATIVE_PASSAGES[:3] + POSITIVE_PASSAGES[:3])
@pytest.mark.parametrize("transform", [str.upper, str.lower, lambda s: "\n\t".join(s.split())])
def test_case_and_whitespace_do_not_change_polarity_or_quote(passage, transform):
    abstract = transform(passage)
    advantage = evidence_card([study(abstract)])["advantage"]
    if passage in NEGATIVE_PASSAGES:
        assert advantage is None
    else:
        assert advantage["text"] == clean(abstract)


def test_negation_frames_with_different_benefits_and_interjections():
    benefits = ["improve efficiency", "reduce latency", "enhance accuracy",
                "outperform the baseline", "achieve low-power operation"]
    frames = ["Our device does not {aside} {benefit} under repeated testing.",
              "Our device fails {aside} to {benefit} under repeated testing.",
              "Our device is unable {aside} to {benefit} under repeated testing.",
              "Our device cannot {aside} {benefit} under repeated testing."]
    for benefit in benefits:
        for frame in frames:
            for aside in ["", ", however,", "(even with extra calibration)", "— despite the revised design —"]:
                passage = clean(frame.format(aside=aside, benefit=benefit))
                assert evidence_card([study(passage)])["advantage"] is None, passage


@pytest.mark.parametrize("capability", ["accuracy", "precision", "fidelity", "reliability",
                                       "throughput", "bandwidth", "performance"])
def test_preserving_a_capability_does_not_negate_a_separate_benefit(capability):
    positive = f"Our device reduces latency without reducing the measured {capability}."
    assert evidence_card([study(positive)])["advantage"]["text"] == positive
    negative = f"Our device fails to reduce latency without reducing the measured {capability}."
    assert evidence_card([study(negative)])["advantage"] is None


@pytest.mark.parametrize("have_positive", [False, True])
def test_real_pipeline_does_not_fill_advantage_from_denied_results(have_positive):
    from app.ml.contracts import AnalysisOptions
    from app.ml.corpus import unpack_snapshot
    from app.ml.engine import analyze, prepare
    from app.ml.text import sentences
    from tests.mvp_fixture import snapshot

    data = snapshot()
    positive = "The measured architecture reduces latency without requiring additional optical components."
    for batch in data["batches"]:
        for entry in batch["documents"]:
            document = entry["document"]
            parts = sentences(document["abstract"])
            parts[3] = " ".join(NEGATIVE_PASSAGES[:3] + ([positive] if have_positive else []))
            document["abstract"] = " ".join(parts)
    options = AnalysisOptions(topic="photonic neuromorphic computing")
    corpus = unpack_snapshot(data)
    result = analyze(corpus, options)
    from tests.mvp_fixture import research_groups
    groups = research_groups(result)
    if not have_positive:
        assert result["candidates"] == []
        assert groups and all(c["card"]["advantage"] is None for c in groups)
        return
    assert len(groups) >= 2
    sources = {s["id"]: s for s in prepare(corpus["entries"], options)[0]}
    for candidate in groups:
        quote = candidate["card"]["advantage"]
        assert quote["text"] == positive
        source = sources[quote["study_id"]]
        assert quote["url"] == source["url"]
        assert quote["text"] in source["abstract"]


@pytest.mark.parametrize("length,accepted", [(899, True), (900, True), (901, False)])
def test_negation_check_preserves_the_quote_length_limit(length, accepted):
    prefix = "Our device improves the measured latency under condition "
    passage = prefix + "x" * (length - len(prefix) - 1) + "."
    quote = evidence_card([study(passage)])["advantage"]
    assert (quote is not None) == accepted
    if quote:
        assert quote["text"] == passage


@pytest.mark.parametrize("positive_index,accepted", [(11, True), (12, False)])
def test_fallback_preserves_the_twelve_source_limit(positive_index, accepted):
    sources = [study(NEGATIVE_PASSAGES[i % 3], str(i)) for i in range(13)]
    sources[positive_index] = study(POSITIVE_PASSAGES[0], "positive")
    quote = evidence_card(sources)["advantage"]
    assert (quote is not None) == accepted
    if quote:
        assert quote["study_id"] == "positive"


def test_randomized_mixed_abstracts_never_select_a_labelled_negative_quote():
    rng = random.Random(20260908)
    pool = NEGATIVE_PASSAGES + POSITIVE_PASSAGES
    for _ in range(1000):
        parts = rng.choices(pool, k=rng.randint(1, 8))
        source = study(" ".join(parts))
        original = deepcopy(source)
        quote = evidence_card([source])["advantage"]
        expected = [p for p in parts if p in POSITIVE_PASSAGES]
        assert (quote is not None) == bool(expected), parts
        if quote:
            assert quote["text"] in expected
            assert quote["url"] == source["url"]
            assert quote["study_id"] == source["id"]
        assert source == original
