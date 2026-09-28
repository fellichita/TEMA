"""Manually segmented scientific text and losslessness checks for the splitter."""

import random

import pytest

from app.ml.text import clean, evidence_card, sentences


@pytest.mark.parametrize("following", [
    "Results confirm stable operation of the device.",
    "Both devices operate under the same conditions.",
    "Measurements confirm the earlier findings.",
    "Further experiments use the same configuration.",
    "Independent measurements agree with the reference.",
    "Zero failures were observed during testing.",
    "Additional experiments confirm this result.",
    "Researchers repeated the measurements independently.",
    "Unexpected changes appeared after heating.",
    "Évaluation des résultats confirme la stabilité.",
    "Результаты подтверждают стабильность устройства.",
    "Ёмкость устройства остаётся постоянной.",
    "3 independent experiments confirm the result.",
    '"Results confirm stable operation," the authors report.',
    "“Both devices operate reliably,” the authors report.",
    "(Results were reproduced independently.)",
])
def test_citation_at_sentence_end_does_not_depend_on_a_starter_word_list(following):
    first = "We used the method of Smith et al."
    assert sentences(first + " " + following) == [first, following]


@pytest.mark.parametrize("parts", [
    ["Smith et al. demonstrate stable operation in their paper.", "Results were reproduced independently."],
    ["Smith et al. (2020) demonstrate stable operation.", "Both configurations were tested."],
    ["Smith et al. [12] demonstrate stable operation.", "Measurements agree with their findings."],
    ["We compared lasers, chips, etc.", "Results were consistent across trials."],
    ["Lasers, chips, etc. were tested under the same conditions.", "Both configurations were stable."],
    ["The output was measured in a.u.", "Measurements were repeated independently."],
    ["The output was measured in a.u. before normalization.", "Calibration remained stable."],
    ["We use several devices, e.g. VCSEL arrays, in the experiment.", "Results are reproducible."],
    ["We use several devices, e. g. VCSEL arrays, in the experiment.", "Results are reproducible."],
    ["We select one configuration, i.e. Device A, for the experiment.", "Both measurements agree."],
    ["We select one configuration, i. e. Device A, for the experiment.", "Both measurements agree."],
    ["Device A vs. Device B is the main comparison.", "Measurements use the same protocol."],
    ["The measurements agree, cf. Smith and colleagues.", "Further experiments are planned."],
    ["The response in Fig. 2 follows Eq. (3).", "Results agree with theory."],
    ["The response in Fig. S2 follows Eqs. (3) and (4).", "Both curves agree."],
    ["See Refs. [1–3] and Figs. 2–4 for details.", "Measurements remain consistent."],
    ["The circuit is described in Vol. II of the report.", "Results are reproduced here."],
    ["Device No. 3 was selected for testing.", "Both runs completed successfully."],
    ["Dr. A. Smith measured the response.", "Results were stable."],
    ["Prof. J. R. Smith measured the response.", "Both measurements agree."],
    ["J. R. Smith measured the response.", "Results were stable."],
    ["J.R. Smith measured the response.", "Results were stable."],
    ["The device was calibrated by A. Chen.", "Results agree with earlier measurements."],
    ["The device was calibrated by J.R. Smith.", "Results agree with earlier measurements."],
    ["The response was measured by A. van der Meer.", "Results agree with earlier measurements."],
    ["The theory follows the work of L. de Broglie.", "Both measurements agree."],
    ["É. Durand measured the response.", "Measurements were stable."],
    ["А. С. Иванов измерил задержку.", "Результаты совпали."],
    ["We measured sample A.", "Results confirm stable operation."],
    ["We selected configuration B.", "Both runs completed successfully."],
    ["The current is 1.5 A.", "Measurements agree with the specification."],
    ["The temperature is 300 K.", "Results agree with the specification."],
    ["The sample was annealed at 950°C.", "The measured response was stable."],
    ["The temperature was maintained at 77 °F.", "Both measurements agree."],
    ["The measurements constrain reflection R and absorption A.", "The results agree with theory."],
    ["The products depend on the value of Φ.", "MoO2 nanoparticles form under these conditions."],
    ["The model is imported into SPICE through Verilog-A.", "In this experiment the response is stable."],
    ["The chip supports high bandwidth optical I/O.", "This work demonstrates stable operation."],
    ["The effective temperature exceeds 2000~K.", "The emitted spectrum was measured."],
    ["The energy efficiency reaches 10^18 MAC/J.", "We compare the device with the baseline."],
    ["The amount begins to decrease above 14 at.% C.", "The atoms aggregate into clusters."],
    ["The response reaches 0.8 meV/K.", "This value exceeds the baseline."],
    ["The temperature remains below 60$^{\\circ}$C.", "A second device was tested independently."],
    ["We support detector R&D.", "These measurements are available to the team."],
    ["The transistors contain Cu$_2$O.", "The measured spectra agree with theory."],
    ["The constraints include energy E and coherence C.", "The capacity was measured."],
    ["The fluctuations increase at large U.", "This behavior was reproduced independently."],
    ["We use materials (silicon, germanium, etc.) with the same geometry.", "Both measurements agree."],
    ["We used the U.S. laboratory setup.", "Both devices operated reliably."],
    ["We used the U.S. Department of Energy setup.", "Results were reproduced independently."],
    ["The U.S. National Science Foundation supported the study.", "Measurements remain consistent."],
    ["The measurements were performed in the U.S.", "Results agree with the specification."],
    ["The delay is 1.5 ns at 2.4 GHz.", "Measurements were repeated."],
    ["Version 1.2.3 processes the data.", "Both outputs agree."],
    ["The source is https://example.org/fig.2.", "Results are available there."],
    ['The authors wrote "Smith et al."', "Results are discussed below."],
    ["The baseline follows the earlier study (Smith et al.).", "Both devices remain stable."],
    ["Stable?", "Yes!", "Both devices work."],
    ["Wait...", "Results are arriving."],
    ["It works.", "the result remains stable."],
    ["Готово.", "ещё один опыт завершён."],
    ["The measurements agree",],
])
def test_scientific_passages_match_manual_sentence_boundaries(parts):
    source = " ".join(parts)
    assert sentences(source) == parts
    assert " ".join(sentences(source)) == clean(source)


@pytest.mark.parametrize("separator", [" ", "  ", "\n", "\r\n", "\t", "\u00a0"])
def test_boundary_whitespace_does_not_change_text_or_segmentation(separator):
    parts = ["We used the method of Smith et al.", "Results were reproduced.", "Both devices work."]
    assert sentences(separator.join(parts)) == parts


def test_complete_sentence_endings_remain_boundaries_before_author_names():
    endings = ["We used the method of Smith et al.", "The temperature is 300 K.",
               "The current is 1.5 A.", "The sample was annealed at 950°C.",
               "We compared lasers, chips, etc.", "The measurements were performed in the U.S."]
    beginnings = ["A. Smith reproduced the measurements.", "J.R. Smith reproduced the measurements.",
                  "Dr. A. Smith reproduced the measurements.", "Results confirm stable operation."]
    for first in endings:
        for second in beginnings:
            assert sentences(first + " " + second) == [first, second], (first, second)


def test_fixed_boundary_is_used_by_evidence_extraction_without_rewriting_quotes():
    result_sentence = "Results show improved energy efficiency during repeated operation."
    study = {"id": "citation-boundary", "title": "Device measurements",
             "abstract": "We followed the protocol of Smith et al. " + result_sentence,
             "url": "https://example.org/citation-boundary"}
    advantage = evidence_card([study])["advantage"]
    assert advantage["text"] == result_sentence
    assert advantage["study_id"] == study["id"]
    assert advantage["url"] == study["url"]
    assert advantage["text"] in study["abstract"]


def test_empty_text_and_punctuation_are_not_fabricated_or_dropped():
    assert sentences(None) == []
    assert sentences("") == []
    assert sentences(" \t\n ") == []
    assert sentences("?!") == ["?!"]
    assert sentences("One") == ["One"]


def test_randomized_inputs_preserve_every_character_of_cleaned_text():
    rng = random.Random(20260908)
    tokens = ["device", "Results", "et al.", "Dr.", "A.", "1.5", "Fig. 2", "e.g.", "e. g.",
              "<b>test</b>", "&amp;", "Δ", "Ёмкость", "“quoted.”", "(measured.)", "[12]", "...", "?!", "end"]
    separators = [" ", "\n", "\t", "  ", "\u00a0"]
    for _ in range(1000):
        source = "".join(rng.choice(tokens) + rng.choice(separators) for _ in range(rng.randrange(1, 60)))
        parts = sentences(source)
        assert parts and all(part and part == part.strip() for part in parts)
        assert " ".join(parts) == clean(source)
        assert sentences(source) == parts
