"""Executor evidence is separate from thematic admission and source modality."""

import json
from pathlib import Path

import pytest

from app.ml.text import enumeration_alternative, execution_evidence, scope_check, sentences


TOPIC = "photonic neuromorphic computing"
PROBES = json.loads((Path(__file__).resolve().parents[1] / "tests/fixtures/ml/executor-evidence.json").read_text(encoding="utf-8"))["executor_probes"]


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p["id"])
def test_pre_executor_audit_has_explicit_evidence_and_admission(probe):
    result = execution_evidence(probe["title"], probe["abstract"])
    expected = probe["expected_proposal"]
    assert result["evidence_level"] == ("none" if expected == "reject" else expected)
    admitted = expected in {"direct", "nominal"} and probe["id"] not in {"nominal_microcomb", "nominal_platform"}
    assert result["admitted"] is admitted
    assert scope_check(TOPIC, probe["title"], probe["abstract"]) == (
        "direct_lexical_signal" if admitted else "scope_not_established")
    assert result["reason"]
    assert isinstance(result["dominant_carrier"], dict)
    for support in result["evidence"]:
        assert support["mode"] in {"study_title", "source_excerpt", "neighboring_excerpts"}
        assert support["text"] == probe["title"] or all(
            part in sentences(probe["abstract"]) for part in support["text"].split("\n"))


@pytest.mark.parametrize("sentence", [
    "from optical to spintronic implementations",
    "Implementations range from magnetic to photonic devices.",
    "Carriers include optical and electronic devices.",
    "Systems such as optical, magnetic and electronic devices are used.",
    "Reservoirs can use either optical or mechanical substrates.",
])
def test_enumeration_requires_optical_and_one_nonoptical_alternative(sentence):
    assert enumeration_alternative(sentence)


@pytest.mark.parametrize("sentence", [
    "A photonic chip performs neural inference.",
    "The photonic and electronic components work together in a hybrid neuron.",
    "Electronic control and optical neural inference are combined.",
    "Implementations range from magnetic to electronic devices.",
    "Devices such as optical chips and photonic circuits perform inference.",
])
def test_hybrids_and_single_axis_lists_are_not_alternatives(sentence):
    assert not enumeration_alternative(sentence)


@pytest.mark.parametrize("passage", [
    "The optical chip performs the matrix multiplication.",
    "Matrix multiplication is performed by the photonic chip.",
    "Training was performed on a GPU while inference ran on the photonic chip.",
    "Our optical processor runs neural inference with weights trained on a GPU.",
    "Our phase-change optical chip performs neural inference.",
    "Our optical memristive device performs neural inference.",
])
@pytest.mark.parametrize("location", ["title", "abstract"])
def test_direct_relation_has_identical_meaning_in_title_and_abstract(passage, location):
    title, abstract = (passage, "We report repeated measurements.") if location == "title" else ("Device measurements", passage)
    result = execution_evidence(title, abstract)
    assert result["admitted"]
    assert result["evidence_level"] == "direct"


@pytest.mark.parametrize("abstract", [
    "Our optical chip performs neural inference. This chip does not perform neural inference.",
    "Our optical chip does not perform neural inference. Our optical chip performs neural inference.",
    "Our optical chip performs neural inference. Neural inference was performed exclusively on a GPU.",
])
def test_same_device_contradiction_prevents_direct_claim(abstract):
    result = execution_evidence("Photonic neural networks", abstract)
    assert not result["admitted"]
    assert result["evidence_level"] == "none"


@pytest.mark.parametrize("abstract", [
    "Our optical chip performs neural inference. A baseline optical device does not implement neural computation.",
    "An earlier optical chip does not perform neural inference. Our new optical chip performs neural inference.",
    "A GPU performs neural inference in the baseline. Our optical chip performs neural inference.",
])
def test_explicitly_different_baseline_does_not_veto_new_executor(abstract):
    result = execution_evidence("Device measurements", abstract)
    assert result["admitted"]
    assert result["evidence_level"] == "direct"


def test_carrier_frequency_is_only_a_diagnostic():
    result = execution_evidence(
        "Thin-film ferromagnetic devices", "Magnetic devices are compared. Our optical chip performs neural inference.")
    assert result["dominant_carrier"]["carrier"] == "magnetic"
    assert result["admitted"]
    assert result["evidence_level"] == "direct"


@pytest.mark.parametrize("title", ["Photonic neural networks", "Optical synaptic transistors", "Design of photonic neural networks"])
def test_named_architecture_is_admitted_but_never_invented_as_execution(title):
    result = execution_evidence(title, "We review recent progress and unresolved questions.")
    assert result["admitted"]
    assert result["evidence_level"] == "nominal"


def test_outlook_after_real_passive_operation_preserves_direct_evidence():
    abstract = "Synaptic plasticity is demonstrated in an optical device, which is promising for future integration."
    result = execution_evidence("Device measurements", abstract)
    assert result["admitted"]
    assert result["evidence_level"] == "direct"
    assert result["evidence"][0]["text"] == abstract


def test_an_application_list_does_not_turn_a_named_platform_into_direct_execution():
    abstract = "Our pixelated photonic circuit provides a platform for neuromorphic computing applications."
    result = execution_evidence("Programmable photonic devices", abstract)
    assert not result["admitted"]
    assert result["evidence_level"] == "nominal"


def test_empty_document_has_no_evidence():
    result = execution_evidence("", "")
    assert result["evidence_level"] == "none"
    assert not result["admitted"]
    assert result["evidence"] == []


@pytest.mark.parametrize("abstract", [
    "Our optical chip performs neural inference, but this chip does not perform neural inference.",
    "Our optical chip performs neural inference; however, neural inference has not been implemented in this chip.",
])
def test_same_sentence_contradiction_is_not_hidden_inside_positive_evidence(abstract):
    assert not execution_evidence("Photonic neural networks", abstract)["admitted"]


def test_enumeration_checks_all_markers_instead_of_stopping_at_unrelated_from():
    abstract = ("Reservoir computations are implemented with inputs from sensors in systems "
                "such as optical and electronic processors.")
    assert enumeration_alternative(abstract)
    assert not execution_evidence("Device measurements", abstract)["admitted"]


def test_optical_material_descriptors_do_not_become_nonoptical_alternatives():
    abstract = "Neural inference is implemented in devices such as optical memristive chips and phase-change photonic circuits."
    assert not enumeration_alternative(abstract)


def test_direct_executor_before_background_enumeration_keeps_its_own_evidence():
    abstract = ("Our optical chip performs neural inference unlike other implementations "
                "ranging from optical to spintronic systems.")
    result = execution_evidence("Device measurements", abstract)
    assert result["admitted"]
    assert result["evidence_level"] == "direct"


@pytest.mark.parametrize("carrier", ["ferromagnetic", "spintronic", "mechanical"])
def test_nominal_optical_title_does_not_override_explicit_nonoptical_computation(carrier):
    result = execution_evidence("Photonic neural networks", f"Our {carrier} device performs neural inference.")
    assert not result["admitted"]


def test_microcomb_operation_is_distinct_from_microcomb_application_outlook():
    direct = execution_evidence("Device measurements", "Our optical microcomb performs neural weighted sums.")
    nominal = execution_evidence("Device measurements", "Microcombs have applications in neuromorphic computing.")
    assert direct["admitted"] and direct["evidence_level"] == "direct"
    assert not nominal["admitted"] and nominal["evidence_level"] == "nominal"


def test_real_platform_outlook_remains_unadmitted_nominal():
    abstract = ("This scalable, energy-efficient, and nonvolatile photonic platform paves the way for "
                "large-scale optical computing, neuromorphic photonics, and next-generation reconfigurable photonic architectures.")
    result = execution_evidence("Programmable photonic integrated circuits", abstract)
    assert result["evidence_level"] == "nominal"
    assert not result["admitted"]


def test_real_optical_convolutional_accelerator_is_direct_even_in_mixed_application_review():
    abstract = ("We report applications to optical neural networks and optical data transmission. "
                "We demonstrate a universal optical vector convolutional accelerator operating at 11 Tera-OPS/s "
                "on 250,000 pixel images for 10 kernels simultaneously. "
                "We also report high data transmission over optical fiber.")
    result = execution_evidence("State-of-the-art Applications of Optical Microcombs", abstract)
    assert result["admitted"] and result["evidence_level"] == "direct"
    assert any("convolutional accelerator" in source["text"] for source in result["evidence"])


@pytest.mark.parametrize("passage", [
    "We experimentally demonstrate the building block of the ONN — a single neuron perceptron — "
    "by mapping synapses onto 49 wavelengths of a micro-comb to achieve a high single-unit "
    "throughput of 11.9 Giga-FLOPS at 8 bits per FLOP, corresponding to 95.2 Gbps.",
    "Our microcomb performs neural weighted sums.",
])
def test_real_microcomb_instrument_supports_direct_neural_execution(passage):
    result = execution_evidence("Device measurements", passage)
    assert result["admitted"] and result["evidence_level"] == "direct"
    assert result["evidence"][0]["text"] == passage


@pytest.mark.parametrize("passage", [
    "Microcombs have applications in neuromorphic computing.",
    "We experimentally demonstrate a micro-comb as a building block for possible future neural computation.",
    "Our microcomb could perform neural weighted sums in the future.",
])
def test_microcomb_outlook_never_becomes_execution_after_carrier_extension(passage):
    result = execution_evidence("Device measurements", passage)
    assert not result["admitted"]
    assert result["evidence_level"] == "nominal"


WDM_BACKGROUND = (
    "Concurrently, advances in neuromorphic photonic computing require novel devices that support "
    "both spectral multiplexing and optical weighting within a unified platform.",
    "This design not only supports efficient green-spectrum transmission but also lays the groundwork "
    "for integrated neuromorphic photonic networks.",
)


def test_control_wdm_synaptic_analogy_is_only_nominal_outlook():
    # Promoted from the frozen blind control after its original labels/predictions
    # were recorded. This is now a regression, not another independent control.
    title = "Design of a four channel green-wavelength multiplexer based on multicore polymer optical fiber"
    abstract = (
        "The growing demand for compact photonic systems in the green spectral range necessitates fully "
        "integrated wavelength division multiplexing (WDM) solutions. Conventional multiplexers often "
        "rely on bulky components, resulting in high insertion losses and limited integration potential. "
        + WDM_BACKGROUND[0] + " This study introduces a compact four-channel green-wavelength optical "
        "multiplexer based on a multi-core polymer optical fiber (MC-POF) embedded with polycarbonate "
        "(PC) cores. The device passively multiplexes light via engineered coupling between adjacent "
        "cores, operating across the 500–560 nm range without the need for external optics. Beam "
        "propagation method (BPM) simulations, combined with MATLAB-based optimization, confirm that "
        "a 20 mm fiber segment enables low insertion losses (0.13–0.55 dB), sharp channel isolation, "
        "and high thermal stability. A single optimized coupling region enables 20 nm channel spacing "
        "across four wavelengths and acts analogously to a synaptic junction, allowing simultaneous "
        "signal convergence. " + WDM_BACKGROUND[1] + " Experimental validation was performed using "
        "a two-channel PC-MC-POF with 500 nm and 540 nm green laser sources. Collimated beams were "
        "coupled into separate fiber cores, and the multiplexed output was directly imaged using "
        "a CMOS camera. The measured far-field intensity profile closely matches simulation results, "
        "confirming effective spatial multiplexing and validating the theoretical model. The proposed "
        "PC-MC-POF multiplexer offers a scalable, low-loss, and energy-efficient solution for "
        "green-wavelength WDM systems. It serves both as a functional optical multiplexer and "
        "a foundational building block for photonic neural architectures, contributing to the "
        "development of next-generation integrated optical communication and computing technologies."
    )
    result = execution_evidence(title, abstract)
    assert not result["admitted"]
    assert result["evidence_level"] == "nominal"
    assert result["reason"] == "potential_application_only"
    assert scope_check(TOPIC, title, abstract) == "scope_not_established"


@pytest.mark.parametrize("background", WDM_BACKGROUND)
def test_demand_or_groundwork_alone_is_not_a_nominal_architecture(background):
    result = execution_evidence("Optical wavelength multiplexer", background)
    assert not result["admitted"]
    assert result["evidence_level"] == "nominal"


@pytest.mark.parametrize("background", WDM_BACKGROUND)
@pytest.mark.parametrize("background_first", [False, True])
def test_demand_or_groundwork_keeps_separate_direct_execution(background, background_first):
    support = "Our optical chip performs neural inference."
    abstract = " ".join((background, support) if background_first else (support, background))
    result = execution_evidence("Optical chip measurements", abstract)
    assert result["admitted"] and result["evidence_level"] == "direct"
    assert any(item["text"] == support for item in result["evidence"])


@pytest.mark.parametrize("background", WDM_BACKGROUND)
@pytest.mark.parametrize("architecture_location", ["title", "abstract"])
def test_demand_or_groundwork_keeps_separate_thematic_architecture_review(background, architecture_location):
    title, abstract = (
        ("Photonic neural network architectures", background + " We review their recent development.")
        if architecture_location == "title"
        else ("Architectural review", background + " We review integrated optical synaptic transistors.")
    )
    result = execution_evidence(title, abstract)
    assert result["admitted"] and result["evidence_level"] == "nominal"
    assert all(item["text"] != background for item in result["evidence"])


@pytest.mark.parametrize("abstract", [
    "Advances in optical computing require novel devices. These devices for neural networks are reviewed.",
    "An optical chip lays the groundwork for integrated systems. This chip for neural networks is reviewed.",
])
def test_background_demand_or_groundwork_cannot_supply_a_neighboring_nominal_carrier(abstract):
    assert not execution_evidence("Device review", abstract)["admitted"]
