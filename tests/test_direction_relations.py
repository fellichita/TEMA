"""Labelled optical-executor relations, shared across titles and abstract passages."""

from copy import deepcopy

import pytest

from app.ml.text import evidence_card, scope_check, sentences


TOPIC = "photonic neuromorphic computing"
DIRECT = "direct_lexical_signal"
OUTSIDE = "scope_not_established"


@pytest.mark.parametrize("title", [
    "Optical devices for possible neural computing",
    "Photonic neural networks",
    "Integrated photonic neuromorphic computing",
    "Optical synaptic devices for neural processing",
])
@pytest.mark.parametrize("abstract", [
    "We characterize optical transmission only. Neural computation has not been implemented.",
    "No neural computation is demonstrated in the optical device.",
    "Our optical device does not implement neural computation.",
    "A neural network runs on a GPU to optimize an optical resonator.",
    "The neural network runs on an electronic computer and predicts optical device geometries.",
])
def test_title_cannot_override_denied_or_non_optical_implementation(title, abstract):
    assert scope_check(TOPIC, title, abstract) == OUTSIDE


NEGATIVE_RELATIONS = [
    "Neural network design of optical filters",
    "Neural networks for the inverse design of optical resonators",
    "Optical filters designed using a neural network",
    "Deep learning for nanophotonic device optimization",
    "A neural network running on a GPU optimizes optical resonators.",
    "We demonstrate a neural network running on a GPU to optimize an optical resonator.",
    "We use an artificial neural network to predict the geometry of optical filters.",
    "An optical resonator is designed by a neural network running on an electronic computer.",
    "We implement a neural network on a digital processor for optical filter design.",
    "We present optical transmission measurements and a separate neural network running on a GPU.",
    "We demonstrate optical transmission while a GPU performs neural inference.",
    "We demonstrate neural inference on a GPU while an optical link transmits the input data.",
    "Optical transmission and neural computation",
    "Optical devices are characterized for possible future neural computation.",
    "Neural computation could be performed by an optical chip in the future.",
    "An optical device fails to implement neural computation.",
    "Optical inputs are used by a GPU to perform neural inference.",
    "We present an optical neural network simulated and executed exclusively on a GPU.",
    "A GPU is used to perform neural inference with optical inputs.",
    "Photonic neural networks on a GPU",
    "A GPU processes optical inputs for neural inference.",
]

POSITIVE_RELATIONS = [
    "Photonic neural networks",
    "Optical synaptic transistors",
    "Laser-based reservoir computing",
    "Light-gated artificial synapses",
    "Integrated neuromorphic photonic computing",
    "Our optical chip implements neural computation.",
    "Neural computation is implemented in an optical circuit.",
    "Our optical processor performs neural inference.",
    "Neural inference is performed by an optical chip.",
    "We use an optical resonator to implement neural weighted sums.",
    "We demonstrate neural weighted sums using optical interferometers.",
    "An optical neural circuit performs inverse design of optical filters.",
    "GPU training of parameters for an optical neural network",
    "We train the parameters on a GPU while the optical chip performs neural inference.",
    "Neural inference is performed by an optical chip while a GPU trains the parameters.",
    "Our optical chip performs neural inference with weights trained on a GPU.",
    "Our optical chip does not require a GPU to perform neural inference.",
    "We demonstrate optical neural computation without a GPU.",
    "Neural inference is performed in an optical chip, not on a GPU.",
    "Photonic convolutional neural networks",
    "Design of photonic neural networks",
    "A photonic accelerator for convolutional neural networks",
    "A codesigned integrated photonic electronic neuron",
    "Phototransistor arrays for neuromorphic vision sensors",
]


@pytest.mark.parametrize("passage", NEGATIVE_RELATIONS)
@pytest.mark.parametrize("location", ["title", "abstract"])
def test_negative_relation_is_rejected_in_both_title_and_abstract(passage, location):
    title, abstract = ((passage, "We report repeated measurements of the response.") if location == "title"
                       else ("Device measurements", passage))
    assert scope_check(TOPIC, title, abstract) == OUTSIDE


@pytest.mark.parametrize("passage", POSITIVE_RELATIONS)
@pytest.mark.parametrize("location", ["title", "abstract"])
def test_positive_relation_is_accepted_in_both_title_and_abstract(passage, location):
    title, abstract = ((passage, "We report repeated measurements of the response.") if location == "title"
                       else ("Device measurements", passage))
    assert scope_check(TOPIC, title, abstract) == DIRECT


@pytest.mark.parametrize("abstract", [
    "An optical chip is fabricated and its parameters are trained on a GPU. This chip implements neural computation.",
    "An optical chip is fabricated and a GPU trains its parameters. This chip implements neural computation.",
    "We fabricate optical chips and train their weights on a GPU. These chips perform neural inference.",
    "We train parameters on a GPU and fabricate an optical chip. It performs neural inference.",
    "A GPU trains the neural network parameters. An optical chip performs inference with the trained weights.",
    "The optical chip performs neural inference. A GPU trains the parameters of the chip.",
    "Neural weighted sums are computed in our processor. This processor uses optical pulses as signals.",
    "We use neural networks to design optical resonators. Our optical chip implements neural computation.",
    "A baseline optical device does not implement neural computation. Our optical chip implements neural computation.",
    "Our optical chip performs neural inference and its weights are trained on a GPU.",
    "We train weights on a GPU and perform neural inference using an optical chip.",
    "A GPU is not used for inference. An optical chip performs neural inference.",
])
def test_mixed_training_and_optical_execution_remains_in_scope(abstract):
    assert scope_check(TOPIC, "Device measurements", abstract) == DIRECT


@pytest.mark.parametrize("abstract", [
    "We fabricate an optical resonator. A neural network on a GPU predicts its geometry.",
    "We fabricate an optical resonator. It is designed by a neural network on a GPU.",
    "We fabricate an optical resonator. This device is optimized using a neural network.",
    "An optical chip is characterized. It runs neural inference on a GPU.",
    "An optical chip is characterized. This GPU performs neural inference.",
    "We fabricate an optical detector and an electronic processor. It performs neural inference.",
    "We fabricate an optical chip and a GPU processor. This processor performs neural inference.",
    "We fabricate an optical chip. A digital processor is characterized. It performs neural inference.",
    "We fabricate an optical chip. This chip could perform neural inference in the future.",
    "We fabricate an optical chip. This chip does not perform neural inference.",
    "A neural network is implemented on a GPU. This processor communicates through an optical link.",
    "We discuss photonic neural networks. Neural computation has not been implemented.",
    "We discuss photonic neural networks. All neural inference runs on a GPU.",
])
def test_neighboring_sentences_cannot_assign_electronic_work_to_optics(abstract):
    assert scope_check(TOPIC, "Device measurements", abstract) == OUTSIDE


def test_scope_validation_preserves_original_sentences_and_card_sources():
    abstract = ("An optical chip is fabricated and its parameters are trained on a GPU. "
                "This chip implements neural computation. "
                "Existing devices suffer from a bandwidth bottleneck. "
                "Our device reduces latency without requiring additional optical components.")
    source = {"id": "hybrid", "title": "Device measurements", "abstract": abstract,
              "url": "https://example.org/hybrid"}
    original = deepcopy(source)
    parts = sentences(abstract)
    card = evidence_card([source])
    assert scope_check(TOPIC, source["title"], abstract) == DIRECT
    assert source == original
    assert sentences(abstract) == parts
    assert evidence_card([source]) == card
    for quote in card.values():
        if quote:
            assert quote["text"] in parts or quote["text"] == source["title"]
            assert quote["url"] == source["url"]


@pytest.mark.parametrize("topic", [TOPIC, "фотонные нейросети", "фотонные нейроморфные вычисления"])
@pytest.mark.parametrize("transform", [str.upper, str.lower, lambda s: "\n\t".join(s.split())])
def test_alias_case_and_whitespace_preserve_the_executor_decision(topic, transform):
    positive = transform("We train parameters on a GPU while an optical chip performs neural inference.")
    negative = transform("We implement a neural network on a GPU to design optical filters.")
    assert scope_check(topic, "Device measurements", positive) == DIRECT
    assert scope_check(topic, "Device measurements", negative) == OUTSIDE


def test_training_execution_and_neighboring_reference_combinations():
    for device in ["optical chip", "photonic circuit", "laser", "optoelectronic device"]:
        noun = device.split()[-1]
        for trainer in ["GPU", "CPU", "digital processor"]:
            for training in ["trains the parameters", "calibrates the weights", "tunes the weights"]:
                positive = f"A {trainer} {training} while a {device} performs neural inference."
                assert scope_check(TOPIC, "Device measurements", positive) == DIRECT, positive
                pair = (f"A {device} is fabricated and a {trainer} {training}. "
                        f"This {noun} implements neural computation.")
                assert scope_check(TOPIC, "Device measurements", pair) == DIRECT, pair
                negative = f"A {trainer} performs neural inference using inputs from a {device}."
                assert scope_check(TOPIC, "Device measurements", negative) == OUTSIDE, negative


@pytest.mark.parametrize("abstract,accepted", [
    ("We characterize optical transmission only. Neural computation has not been implemented.", False),
    ("We implement a neural network on a GPU to design optical filters.", False),
    ("An optical chip is characterized. This chip runs neural inference on a GPU.", False),
    ("An optical chip is fabricated and its parameters are trained on a GPU. "
     "This chip implements neural computation.", True),
])
def test_real_model_preparation_obeys_scope_even_with_matching_titles(abstract, accepted):
    from app.ml.contracts import AnalysisOptions
    from app.ml.corpus import unpack_snapshot
    from app.ml.engine import analyze
    from tests.mvp_fixture import snapshot

    data = snapshot()
    count = 0
    for batch in data["batches"]:
        for entry in batch["documents"]:
            count += 1
            # Keep years distinct so the existing version merger does not alter this scope assertion.
            entry["document"]["title"] += f" ({entry['document']['publication_year']})"
            entry["document"]["abstract"] = (abstract + " Existing devices face a bandwidth bottleneck. "
                                              "Our device reduces latency under repeated testing.")
    result = analyze(unpack_snapshot(data), AnalysisOptions(topic=TOPIC))
    if accepted:
        from tests.mvp_fixture import research_groups
        assert result["preparation"]["retained_studies"] == count
        assert research_groups(result)
        assert all(all(c["card"].values()) for c in research_groups(result))
    else:
        assert result["preparation"]["retained_studies"] == 0
        assert result["preparation"]["rejected"]["scope_not_established"] == count
        assert result["candidates"] == []


@pytest.mark.parametrize("title,abstract", [
    ("A new paradigm of reservoir computing exploiting hydrodynamics",
     "We created the Aqua-Photonic-Advantaged Computing Machine by Artificial Neural Networks. "
     "Wave propagation in shallow water performs the computation; a camera records the water surface."),
    ("AI-powered adaptive optical metamaterials",
     "We use generative AI to choose optimal optical material configurations. "
     "Some key applications include smart lenses and hardware configuration for neuromorphic photonic processors."),
])
def test_brand_names_and_application_lists_do_not_establish_an_optical_executor(title, abstract):
    assert scope_check(TOPIC, title, abstract) == OUTSIDE
