"""Manually specified raw scientific sentences and exact archive offsets."""

import random

import pytest

from app.pilot.sentences import (
    MAX_TEXT_CHARACTERS, SEGMENTATION_VERSION, SentenceSegmentationError,
    SentenceSpan, sentence_spans, sentences,
)


@pytest.mark.parametrize("parts", [
    ("The sensitivity improved by 43.94% at 1.5 V.", "Measurements were repeated."),
    ("The value changed from −1.25 to 0.08 µT.", "The uncertainty was 0.01 µT."),
    ("Version 1.2.3 processed the data.", "Both outputs agree."),
    ("Devices, e.g. VCSEL arrays, were measured.", "Results agree."),
    ("Devices, e. g. VCSEL arrays, were measured.", "Results agree."),
    ("We select one configuration, i.e. Device A.", "Both runs finished."),
    ("We select one configuration, i. e. Device A.", "Both runs finished."),
    ("Device A vs. Device B is the comparison.", "Measurements agree."),
    ("The measurements agree, cf. Smith and colleagues.", "Further work follows."),
    ("Smith et al. demonstrate stable operation.", "Results were reproduced."),
    ("Smith et al. (2020) demonstrate stable operation.", "Both runs finished."),
    ("Smith et al. [12] demonstrate stable operation.", "Both runs finished."),
    ("We used the method of Smith et al.", "Results were reproduced."),
    ("We used the method of Smith et al.", "А. Иванов повторил измерения."),
    ("We compared chips, lasers, etc.", "Results were consistent."),
    ("Chips, lasers, etc. were tested.", "Both devices worked."),
    ("The output was measured in a.u.", "Measurements were repeated."),
    ("The output was measured in a.u. before normalization.", "Calibration was stable."),
    ("The response in Fig. 2 follows Eq. (3).", "Results agree."),
    ("The response in Fig. S2 follows Eqs. (3) and (4).", "Results agree."),
    ("See Refs. [1–3] and Figs. 2–4 for details.", "Measurements agree."),
    ("The circuit is described in Vol. II of the report.", "Results agree."),
    ("Device No. 3 was selected.", "Both runs finished."),
    ("Dr. A. Smith measured the response.", "Results were stable."),
    ("Prof. J. R. Smith measured the response.", "Results were stable."),
    ("J. R. Smith measured the response.", "Results were stable."),
    ("J.R. Smith measured the response.", "Results were stable."),
    ("The device was calibrated by A. Chen.", "Results agree."),
    ("The response was measured by A. van der Meer.", "Results agree."),
    ("The theory follows the work of L. de Broglie.", "Both runs finished."),
    ("É. Durand measured the response.", "Measurements were stable."),
    ("А. С. Иванов измерил задержку.", "Результаты совпали."),
    ("We measured sample A.", "Results confirm stable operation."),
    ("The current is 1.5 A.", "Measurements agree."),
    ("The temperature is 300 K.", "A. Smith repeated the measurement."),
    ("The sample was annealed at 950°C.", "The response was stable."),
    ("The temperature was maintained at 77 °F.", "Both measurements agree."),
    ("The products depend on the value of Φ.", "MoO2 nanoparticles form."),
    ("The model uses Verilog-A.", "In this experiment the response is stable."),
    ("The chip supports optical I/O.", "This work demonstrates stable operation."),
    ("The temperature exceeds 2000~K.", "The spectrum was measured."),
    ("The efficiency reaches 10^18 MAC/J.", "We compare the baseline."),
    ("The amount decreases above 14 at.% C.", "The atoms aggregate."),
    ("The response reaches 0.8 meV/K.", "This exceeds the baseline."),
    ("The temperature is below 60$^{\\circ}$C.", "A second device was tested."),
    ("We support detector R&D.", "These measurements are available."),
    ("The transistors contain Cu$_2$O.", "The spectra agree."),
    ("The constraints include energy E and coherence C.", "The capacity was measured."),
    ("We use materials (silicon, germanium, etc.) with the same geometry.", "Both runs finished."),
    ("We used the U.S. laboratory setup.", "Both devices operated reliably."),
    ("We used the U.S. Department of Energy setup.", "Results agree."),
    ("The U.S. National Science Foundation supported the study.", "Measurements agree."),
    ("Measurements were performed in the U.S.", "Results agree."),
    ("The source is https://example.org/fig.2.", "Results are available there."),
    ("See https://example.org/v1.2?q=yes!&x=3.14 for details.", "Results agree."),
    ("See https://en.wikipedia.org/wiki/Function_(mathematics).", "Results agree."),
    ("The source is www.example.org/fig.2.", "Results are available there."),
    ("The source is https://example.org/Dr.", "Results are available there."),
    ('The authors wrote "Smith et al."', "Results are discussed below."),
    ("The baseline follows the earlier study (Smith et al.).", "Both devices work."),
    ("Мы использовали лазеры, чипы и т. д. при измерениях.", "Результаты совпали."),
    ("Мы использовали лазеры, чипы и т.д.", "Результаты совпали."),
    ("Требуется настройка, т. е. Калибровка A, перед опытом.", "Проверка завершена."),
    ("Опыт повторён, т.к. датчик был заменён.", "Результат сохранился."),
    ("Иванов и др. измерили задержку.", "Результаты совпали."),
    ("См. рис. 2 и табл. 3 для сравнения.", "Результаты совпали."),
    ("Стоимость составила 1.5 млн. рублей.", "Проект завершён."),
    ("Stable?", "Yes!", "Both devices work."),
    ("Wait...", "Results are arriving."),
    ("It works.", "the result remains stable."),
    ("Готово.", "ещё один опыт завершён."),
    ("🙂 Работает.", "Ёмкость изменилась на 0,5 нФ."),
    ("The measurements agree",),
])
def test_manual_scientific_boundaries(parts):
    source = " ".join(parts)
    assert sentences(source) == parts
    expected = []
    start = 0
    for part in parts:
        expected.append(SentenceSpan(start, start + len(part)))
        start += len(part) + 1
    assert sentence_spans(source) == tuple(expected)


@pytest.mark.parametrize("separator", [" ", "  ", "\n", "\r\n", "\t", "\u00a0", "\u2003"])
def test_preserves_internal_whitespace_and_original_unicode_offsets(separator):
    first = "  " + "Δe\u0301  improved\tby 43.94%\r\nat 1.5\u00a0V."
    second = "Ёмкость\tосталась  постоянной."
    source = first + separator + second + "\n\t"
    expected = (SentenceSpan(2, len(first)), SentenceSpan(len(first) + len(separator), len(source) - 2))
    assert sentence_spans(source) == expected
    assert sentences(source) == (first[2:], second)
    assert source[expected[0].start:expected[0].end] == first[2:]


@pytest.mark.parametrize("source, expected", [
    ('<jats:p>We <italic>measured</italic> 43.94% at 1.5 V.</jats:p>\n<jats:p>Results agree.</jats:p>',
     ('<jats:p>We <italic>measured</italic> 43.94% at 1.5 V.</jats:p>', '<jats:p>Results agree.</jats:p>')),
    ('<p>A. Smith measured 1.5 V.</p><p>Results agree.</p>',
     ('<p>A. Smith measured 1.5 V.</p>', '<p>Results agree.</p>')),
    ('<p data-note="Results. More? > Still!">We measured &gt;43.94%.</p> Next result.',
     ('<p data-note="Results. More? > Still!">We measured &gt;43.94%.</p>', 'Next result.')),
    ('We used <xref ref-type="bibr">Smith et al.</xref> (2020) for calibration. Results agree.',
     ('We used <xref ref-type="bibr">Smith et al.</xref> (2020) for calibration.', 'Results agree.')),
    ('<!-- A comment. Still? -->We measured 1.5 V. Next result.',
     ('<!-- A comment. Still? -->We measured 1.5 V.', 'Next result.')),
    ('We measured &lt;2.5 µA &amp; &gt;1.0 V. Results agree.',
     ('We measured &lt;2.5 µA &amp; &gt;1.0 V.', 'Results agree.')),
    ('量子传感器测得1.5纳特。结果稳定！', ('量子传感器测得1.5纳特。', '结果稳定！')),
    ('هل يعمل؟نعم!', ('هل يعمل؟', 'نعم!')),
])
def test_raw_markup_entities_and_non_ascii_sentence_marks(source, expected):
    assert sentences(source) == expected
    spans = sentence_spans(source)
    assert tuple(source[start:end] for start, end in spans) == expected


@pytest.mark.parametrize("source, expected", [(None, ()), ("", ()), (" \t\n\u00a0", ()),
                                               ("?!", ("?!",)), ("One", ("One",)), ("...", ("...",))])
def test_empty_and_short_inputs(source, expected):
    assert sentences(source) == expected


def test_invalid_or_oversized_source_is_rejected_without_partial_output():
    assert SEGMENTATION_VERSION == "raw-sentence-spans/1.0.0"
    for value in (1, [], b"An abstract."):
        with pytest.raises(SentenceSegmentationError):
            sentence_spans(value)
    with pytest.raises(SentenceSegmentationError, match="220 000"):
        sentence_spans("a" * (MAX_TEXT_CHARACTERS + 1))
    assert sentence_spans("a" * MAX_TEXT_CHARACTERS) == (SentenceSpan(0, MAX_TEXT_CHARACTERS),)


@pytest.mark.parametrize("source", [
    "e.g. " * 40_000,
    "A. " * 65_000,
    "X. y. " * 30_000,
    "https://example.org/" + ")" * 100_000 + ". End.",
    '<p title="' + ". " * 90_000 + '">Measured 1.5 V.</p>',
    "<!--" * 50_000 + " This comment was never closed. Still raw!",
    "<p title='" * 15_000 + " No closing tag. Still raw!",
    "<malformed " * 15_000 + "First raw sentence. Second raw sentence.",
# Name the cases: a generated id repeats the whole input, and Windows refuses
# the PYTEST_CURRENT_TEST variable once it passes 32 767 characters.
], ids=["abbreviations", "initials", "initial_pairs", "closing_parentheses",
        "attribute_periods", "unclosed_comments", "unclosed_attributes", "malformed_tags"])
def test_long_adversarial_inputs_keep_ordered_lossless_spans(source):
    spans = sentence_spans(source)
    previous = 0
    for start, end in spans:
        assert previous <= start < end <= len(source)
        assert not source[start].isspace() and not source[end - 1].isspace()
        assert not source[previous:start].strip()
        previous = end
    assert not source[previous:].strip()


def test_generated_mixed_text_preserves_every_non_boundary_character():
    rng = random.Random(20260915)
    tokens = ["device", "Results", "et al.", "Dr.", "A.", "1.5", "Fig. 2", "e.g.", "e. g.",
              "<b>test</b>", "&amp;", "Δe\u0301", "Ёмкость", "“quoted.”", "(measured.)", "[12]",
              "https://example.org/1.5?q=yes", "...", "?!", "end", "т. д.", "🙂"]
    separators = [" ", "\n", "\t", "  ", "\u00a0"]
    for _ in range(300):
        source = "".join(rng.choice(tokens) + rng.choice(separators) for _ in range(rng.randrange(1, 60)))
        spans = sentence_spans(source)
        assert spans == sentence_spans(source)
        previous = 0
        for start, end in spans:
            assert previous <= start < end
            assert not source[previous:start].strip()
            assert source[start:end] == source[start:end].strip()
            previous = end
        assert not source[previous:].strip()
