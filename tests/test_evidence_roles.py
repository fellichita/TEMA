import pytest
import json
from pathlib import Path

from app.ml.evidence import supported_card, quote_assessment
from app.ml.text import evidence_card, execution_evidence


def study(abstract, identifier="source", **changes):
    return {"id": identifier, "title": "An implemented photonic neural processor",
            "url": "https://example.org/" + identifier, "abstract": abstract, "type": "article", **changes}


@pytest.mark.parametrize("text", [
    "Then we introduce different optical neural networks achieved by free-space optics and integrated waveguides.",
    "Our photonic chip may improve latency in future neural processors.",
    "We propose a new approach that could reduce energy consumption in these systems.",
    "Our optical system is expected to improve energy efficiency considerably.",
])
def test_outlook_or_review_structure_is_not_a_reported_advantage(text):
    sources = [study(text)]
    card, annotations, _, _ = supported_card(sources, evidence_card(sources))
    assert card["advantage"] is None
    assert annotations["advantage"]["modality"] == "not_found"


@pytest.mark.parametrize("text,modality", [
    ("Our photonic device improves energy efficiency and reduces inference latency.", "reported_result"),
    ("Numerical simulations show that the photonic circuit reduces inference latency.", "simulation"),
    ("The circuit provides efficient optical neural computation with low power consumption.", "source_statement"),
])
def test_supported_qualitative_results_and_simulations_keep_original_quote(text, modality):
    sources = [study(text)]
    card, annotations, _, _ = supported_card(sources, evidence_card(sources))
    assert card["advantage"]["text"] == text
    assert annotations["advantage"]["modality"] == modality


def test_card_15_review_fragments_do_not_fill_wrong_roles():
    sources = [study("Then we introduce optical neural networks achieved by free-space optics. "
                     "We know that neural network execution on a digital computer cannot perform real-time processing.",
                     type="review")]
    card, annotations, _, _ = supported_card(sources, evidence_card(sources))
    assert card["advantage"] is None
    assert card["example"]["mode"] == "study_title"
    assert annotations["example"]["modality"] == "research_reference"
    assert card["problem"] is None  # Missing explicit extractable problem remains missing.


def test_rejected_choice_is_replaced_from_another_document_with_provenance():
    sources = [study("Our approach might improve energy efficiency in the future.", "future"),
               study("We demonstrate a photonic circuit that reduces measured energy consumption.", "measured")]
    card, _, _, _ = supported_card(sources, evidence_card(sources))
    assert card["advantage"]["study_id"] == "measured"
    assert card["advantage"]["text"] == sources[1]["abstract"]
    assert card["advantage"]["url"] == sources[1]["url"]


@pytest.mark.parametrize("text", [
    "Our circuit reduces neither energy consumption nor latency.",
    "Our chip fails, however, to improve energy efficiency.",
    "An improvement in latency was ruled out in our measurements.",
])
def test_postprocessing_cannot_bypass_protected_negation_guard(text):
    sources = [study(text)]
    assert supported_card(sources, evidence_card(sources))[0]["advantage"] is None


def test_result_of_review_is_labelled_as_secondary_evidence():
    sources = [study("These devices have achieved low-power operation at high inference speed.", type="review")]
    _, annotations, explanations, _ = supported_card(sources, evidence_card(sources))
    assert annotations["advantage"]["modality"] == "reviewed_result"
    assert "обзоре" in explanations["advantage"]


def test_all_fifteen_role_errors_from_manual_initial_audit_are_rejected():
    audit = json.loads((Path(__file__).resolve().parents[1] /
                       "tests/fixtures/ml/photonic-evidence-roles.json").read_text(encoding="utf-8"))
    checked = 0
    for row in audit["records"]:
        for field, review in row["field_reviews"].items():
            if review["verdict"] != "fail":
                continue
            source, quote = review["source_context"], review["quote"]
            document = study(source["abstract"], quote["study_id"], title=source["title"],
                             execution=execution_evidence(source["title"], source["abstract"]))
            assert not quote_assessment(field, quote, document)["accepted"], (row["key"], field, quote["text"])
            checked += 1
    assert checked == 15


def test_simulation_modality_is_preserved_from_the_original_adjacent_context():
    document = study("We numerically compare two photonic reservoir configurations. "
                     "The first configuration reduces the bit error rate by a factor of two.")
    card, annotations, explanations, _ = supported_card([document], evidence_card([document]))
    assert card["advantage"]["text"].startswith("The first")
    assert annotations["advantage"]["modality"] == "simulation"
    assert annotations["advantage"]["modality_uses_context"]
    assert annotations["advantage"]["context"] == ["We numerically compare two photonic reservoir configurations."]
    assert "симуляции" in explanations["advantage"]


def test_reported_fabrication_does_not_become_a_proposal():
    text = "Here we propose and fabricate an optical neural processor for inference."
    document = study(text)
    quote = evidence_card([document])["example"]
    assert quote_assessment("example", quote, document)["modality"] == "reported_result"


def assessment(field, text):
    document = study(text)
    quote = {"text": text, "study_id": document["id"], "url": document["url"],
             "title": document["title"], "mode": "source_excerpt"}
    return quote_assessment(field, quote, document)


@pytest.mark.parametrize("field,text", [
    ("advantage", "The neuromorphic system processes enormous information even with very low energy consumption, which practically can be achieved with photonic artificial synapse."),
    ("advantage", "Nanophotonic spiking neural networks are of key importance for realizing brain-inspired, power-efficient artificial intelligence systems."),
    ("advantage", "These findings contribute to the broader understanding of quantum reservoirs for high performance, efficient quantum machine learning and time-series forecasting."),
    ("advantage", "With the development of DNA synthesis and sequencing technologies and the reduction of cost, DNA digital storage has attracted more and more attention and achieved significant breakthroughs."),
    ("problem", "These approaches move towards addressing key challenges in molecular data retrieval by offering simplified, rapid isothermal protocols and new DNA data access capabilities."),
    ("problem", "These challenges are overcome by a proper choice of group homomorphisms."),
    ("problem", "DNA data storage systems have made significant strides toward addressing the limitations of traditional storage media."),
])
def test_remaining_bad_roles_from_final_three_direction_audit_are_rejected(field, text):
    assert not assessment(field, text)["accepted"]


@pytest.mark.parametrize("text", [
    "The optical processor achieves low energy consumption during inference.",
    "The photonic processor improves energy efficiency through parallel inference.",
    "Our quantum reservoir reduces prediction errors by twenty percent.",
])
def test_concrete_results_survive_rejection_of_conditional_or_generic_progress(text):
    assert assessment("advantage", text)["accepted"]


@pytest.mark.parametrize("text", [
    "Despite its potential, maintaining DNA integrity over extended periods is challenging.",
    "Limited by the current biochemical techniques, data might be corrupted during the processes of DNA data storage.",
    "Quantum computing has been moving from a theoretical phase to practical one, presenting daunting challenges in implementing physical qubits.",
    "Quantum gradient calculation poses challenges in current near-term quantum hardware and simulation software.",
    "Application is anticipated, but prevailing approaches suffer from the collapse of the quantum state upon measurement.",
    "Excitable nanophotonic devices remain challenging and mostly unexplored in experiments.",
    "High cost remains challenging; our design addresses these challenges with shorter strands.",
])
def test_problem_describes_a_constraint_not_a_simulation_outlook_or_experiment(text):
    verdict = assessment("problem", text)
    assert verdict["accepted"]
    assert verdict["modality"] == "source_statement"
    assert not verdict["modality_uses_context"]


@pytest.mark.parametrize("text", [
    "Here, an optical synaptic transistor is proposed to enhance memory stability.",
    "Here we propose an optical processor which can control neural inference.",
])
def test_proposed_benefit_does_not_turn_the_example_into_a_reported_result(text):
    assert assessment("example", text)["modality"] == "proposal"
    assert not assessment("advantage", text)["accepted"]


def test_simulation_and_wet_experiment_keep_both_modalities_and_an_honest_explanation():
    text = "The simulation and wet experiment results demonstrated that FDMC achieved handle-level random access in a lossless encrypted DNA storage system, which balanced security and robustness."
    document = study(text)
    card, annotations, explanations, _ = supported_card([document], evidence_card([document]))
    assert card["advantage"]["text"] == text
    assert annotations["advantage"]["modality"] == "simulation_and_experiment"
    assert "и о симуляции, и о физическом эксперименте" in explanations["advantage"]
    assert "а не" not in explanations["advantage"]


@pytest.mark.parametrize("text", [
    "Here we propose and demonstrate an optical neural processor for inference.",
    "Here a photonic neural processor has been proposed and demonstrated successfully.",
    "We also carried out large scale experiments to validate our proposed DNA-based data storage architecture.",
])
def test_reported_demonstration_is_not_lost_because_the_same_excerpt_also_mentions_proposal(text):
    assert assessment("example", text)["modality"] == "reported_result"


def test_unexplored_efficient_devices_describe_a_gap_not_an_established_advantage():
    text = "Despite significant advances in photonic computing, compact and efficient optical neural elements remain largely unexplored."
    assert not assessment("advantage", text)["accepted"]
    assert assessment("problem", text)["accepted"]


def test_research_interest_is_not_a_benefit_but_concrete_convergence_improvement_is():
    motivation = "This has revived research interest in new unconventional hardware for more efficient ANNs rather than emulating them on traditional machines."
    result = "Our optical processor reduces convergence time and improves classification accuracy."
    assert not assessment("advantage", motivation)["accepted"]
    assert assessment("advantage", result)["accepted"]
