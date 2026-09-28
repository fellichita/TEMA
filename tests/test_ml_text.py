"""Sentence boundaries and focused photonic direction-filter regressions."""

import pytest

from app.ml.text import clean, evidence_card, scope_check, sentences


TOPIC = "photonic neuromorphic computing"


@pytest.mark.parametrize("parts", [
    ["Latency falls.", "Efficiency improves."],
    ["We measured the device response.", "the measured latency remained stable."],
    ["The delay is 1.5 ns at 2.4 GHz.", "The result was reproduced independently."],
    ["We use several devices, e.g. VCSEL arrays, for neural processing.", "They share a common input."],
    ["The response in Fig. 2 follows Eq. (3).", "The measurements agree with theory."],
    ["Smith et al. demonstrate stable operation in their paper.", "We reproduce their experiment."],
    ["The baseline was reported by Smith et al.", "We compare both configurations here."],
    ["Dr. A. Smith measured the response.", "The response was stable."],
    ["We used the U.S. laboratory setup.", "The same setup runs locally."],
    ['The authors report "stable operation."', "We confirm the result independently."],
    ["Does the device respond?", "Yes!", "The response remains stable."],
    ["Система работает.", "её задержка составляет 1.5 нс.", "Проверка завершена."],
])
def test_sentence_boundaries_preserve_complete_source_passages(parts):
    source = " ".join(parts)
    assert sentences(source) == parts
    assert " ".join(sentences(source)) == clean(source)


def test_sentence_cleaning_keeps_empty_input_empty_and_normalizes_only_whitespace():
    assert sentences(None) == []
    assert sentences("  \n ") == []
    assert sentences("<p>Latency falls.</p>\\n<p>Efficiency improves.</p>") == [
        "Latency falls.", "Efficiency improves."]


def test_evidence_keeps_the_existing_minimum_length_after_short_sentences_are_retained():
    study = {"id": "short", "title": "Measured photonic neural device",
             "abstract": "Efficiency improves. The measured device reduces latency under repeated operation.",
             "url": "https://example.org/short"}
    assert evidence_card([study])["advantage"]["text"] == (
        "The measured device reduces latency under repeated operation.")


@pytest.mark.parametrize("title", [
    "VCSEL-based spiking processors",
    "Laser-based reservoir computing",
    "Nanophotonic implementation of artificial neurons",
    "Light-induced artificial synapses",
    "Photomemristors implementing spike-timing-dependent plasticity",
    "Optical STDP devices",
    "Light-gated synaptic transistors",
    "Phototransistors for artificial synapses",
    "Photonic spike encoding circuits",
])
def test_direction_recognizes_explicit_optical_and_neural_terminology(title):
    assert scope_check(TOPIC, title, "We report repeated measurements of the device response.") == "direct_lexical_signal"


@pytest.mark.parametrize("title,abstract", [
    ("Electronic spiking processors", "We implement the circuit with CMOS transistors."),
    ("Lightweight spiking processor", "We demonstrate a digital implementation using electronic gates."),
    ("Laser characterization of electronic memristors", "We present measurements of material response."),
    ("Photographs for neural network training", "We demonstrate image classification on a GPU."),
    ("Intensity spikes in laser emission", "We demonstrate the origin of the observed fluctuations."),
    ("Laser control for water reservoirs", "We demonstrate an improved system for water level measurement."),
    ("Device optimization", "We use machine learning to optimize laser cavities."),
])
def test_direction_vocabulary_does_not_admit_unrelated_devices(title, abstract):
    assert scope_check(TOPIC, title, abstract) == "scope_not_established"


@pytest.mark.parametrize("abstract", [
    "Synaptic plasticity is demonstrated in an optical device, which is promising for integration.",
    "Neural weighted sums are computed by an optical interferometer, a promising building block.",
    "Spike encoding has been implemented in a VCSEL-SA, showing potential for integration.",
    "Synaptic weights are modulated by light pulses, suggesting possible future applications.",
    "A promising optical synapse is experimentally demonstrated under repeated stimulation.",
    "Neural computation is implemented in an optical circuit.",
])
def test_realized_passive_neural_operations_are_not_rejected_by_outlook(abstract):
    assert scope_check(TOPIC, "Device measurements", abstract) == "direct_lexical_signal"


@pytest.mark.parametrize("abstract", [
    "Neural processing could be implemented with optical devices in the future.",
    "Neural processing can be implemented with optical devices.",
    "Synaptic weights may be modulated by optical pulses in future applications.",
    "Synaptic plasticity is not demonstrated in this optical device.",
    "Optical transmission is demonstrated, while neural computing remains a potential application.",
    "Promising optical materials are synthesized for possible neural applications.",
    "Optical materials are demonstrated for potential neural computing applications.",
    "Synaptic plasticity is never observed in the optical device.",
])
def test_passive_wording_does_not_confirm_hypothetical_or_unrelated_functions(abstract):
    assert scope_check(TOPIC, "Device measurements", abstract) == "scope_not_established"


@pytest.mark.parametrize("abstract", [
    "An optical device was fabricated. This device implements neural weighted sums.",
    "This work introduces a laser cavity. It emulates a spiking neuron.",
    "A VCSEL-SA was fabricated. The device implements spike encoding.",
    "Neural weighted sums are computed in our processor. This processor uses optical pulses as signals.",
    "Photonic chips were fabricated. These chips implement neural computation.",
])
def test_adjacent_sentences_can_establish_a_link_for_the_same_device(abstract):
    assert scope_check(TOPIC, "Device measurements", abstract) == "direct_lexical_signal"


@pytest.mark.parametrize("abstract", [
    "An optical link is characterized. A separate electronic processor implements neural weighted sums.",
    "We fabricate an optical resonator. In a separate experiment, a neural model predicts material properties.",
    "An optical device was fabricated. This device could implement neural processing in the future.",
    "An optical device was fabricated. This electronic controller implements neural computation.",
    "An optical device was fabricated. A neural processor was characterized. This processor implements weighted sums.",
    "An optical link is characterized. This study implements a neural model on a GPU.",
    "An optical device was characterized. It implements a neural model on a GPU.",
    "We built an optical detector and an electronic processor. It implements neural computation.",
    "An optical device was fabricated. This device does not implement neural computation.",
])
def test_adjacent_sentences_do_not_join_unrelated_or_unsupported_operations(abstract):
    assert scope_check(TOPIC, "Device measurements", abstract) == "scope_not_established"


def test_title_is_not_an_antecedent_for_an_unrelated_abstract_sentence():
    assert scope_check(TOPIC, "An optical device", "This device implements a neural model on a GPU.") == "scope_not_established"


def test_scope_pairing_does_not_merge_or_rewrite_evidence_quotes():
    study = {"id": "adjacent", "title": "Device measurements",
             "abstract": "An optical device was fabricated. This device implements neural weighted sums. "
                         "The same device reduces latency in repeated experiments.",
             "url": "https://example.org/adjacent"}
    assert scope_check(TOPIC, study["title"], study["abstract"]) == "direct_lexical_signal"
    for field in evidence_card([study]).values():
        if field is not None and field["mode"] == "source_excerpt":
            assert field["text"] in sentences(study["abstract"])
            assert field["study_id"] == study["id"]


@pytest.mark.parametrize("title,abstract", [
    ("Device measurements", "Orientation selectivity and spatiotemporal processing are demonstrated using "
     "direct optical stimuli, illustrating an efficient approach for neuromorphic computing."),
    ("Integrated Neuromorphic Photonic Computing: Devices and Future Paradigms",
     "This review summarizes the existing physical implementations."),
    ("Optical neural circuit for inverse design",
     "We demonstrate synaptic computation in an optical chip. Its application is inverse design with machine learning."),
])
def test_scope_preserves_realized_operations_with_commas_and_review_titles(title, abstract):
    assert scope_check(TOPIC, title, abstract) == "direct_lexical_signal"


@pytest.mark.parametrize("title,abstract", [
    ("Metric learning for nanophotonics",
     "We present machine learning algorithms for inverse design of nanophotonic structures. "
     "The presented machine learning framework analyzes nanophotonic responses."),
    ("Lasers that learn: laser machining and machine learning",
     "We demonstrate neural networks for modelling laser machining processes."),
    ("Characterizing a chaotic laser using machine learning",
     "We present machine learning analysis of optical spikes and measurement statistics."),
    ("Nanophotonics inverse design with deep learning",
     "This work presents deep learning for inverse design of nanophotonic structures."),
])
def test_learning_used_to_study_optics_is_not_an_optical_neural_implementation(title, abstract):
    assert scope_check(TOPIC, title, abstract) == "scope_not_established"
