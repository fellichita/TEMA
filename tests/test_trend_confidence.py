"""Уверенность по помесячной кривой технологии: логика эксперта."""

from datetime import date

from app.trend_confidence import (FULL_CONFIDENCE_R2, PARTIAL_COVERAGE_CAP, SINGLE_CLASS_CAP, Material,
                                  assess_curve, month_window, smooth, trust_level)

AS_OF = date(2026, 9, 26)
MONTHS = month_window(AS_OF)


def series(counts: list[int], sources: tuple[str, ...] = ("crossref", "arxiv"), prefix: str = "w") -> list[Material]:
    materials = []
    for index, count in enumerate(counts):
        for number in range(count):
            source = sources[number % len(sources)]
            materials.append(Material(f"{prefix}-{index}-{number}", source, MONTHS[index]))
    return materials


ACCELERATING = [round(0.04 * index * index) + 1 for index in range(24)]


def test_window_is_twenty_four_finished_months():
    assert len(MONTHS) == 24
    assert MONTHS[0] == "2024-09" and MONTHS[-1] == "2026-08"


def test_accelerating_growth_from_independent_sources_is_full_confidence():
    result = assess_curve(series(ACCELERATING), AS_OF)
    assert result.confidence == 100
    assert result.trend == "растёт быстро"
    assert result.quadratic_r2 >= FULL_CONFIDENCE_R2 and result.curvature > 0
    assert all(check.passed for check in result.checks)
    assert result.source_classes == ("preprint", "scientific_index")


def test_single_source_type_is_capped():
    result = assess_curve(series(ACCELERATING, ("crossref",)), AS_OF)
    assert result.confidence == SINGLE_CLASS_CAP
    assert not next(check for check in result.checks if check.name == "independent_sources").passed


def test_partial_coverage_is_capped():
    assert assess_curve(series(ACCELERATING), AS_OF, coverage_complete=False).confidence == PARTIAL_COVERAGE_CAP


def test_decline_gives_zero_not_unknown():
    result = assess_curve(series(list(reversed(ACCELERATING))), AS_OF)
    assert result.confidence == 0
    assert result.trend == "снижается"


def test_too_little_data_is_unknown():
    result = assess_curve(series([0] * 20 + [1, 1, 2, 3]), AS_OF)
    assert result.confidence is None


def test_same_work_in_two_sources_counts_once_with_highest_coefficient():
    materials = [Material("doi:1", "arxiv", MONTHS[5]), Material("doi:1", "crossref", MONTHS[5]),
                 Material("doi:2", "hacker_news", MONTHS[5])]
    point = assess_curve(materials, AS_OF).months[5]
    assert point.materials == 2
    assert point.weighted == 1.25
    assert assess_curve(materials, AS_OF).sources == {"arxiv": 1, "crossref": 1, "hacker_news": 1}


def test_materials_outside_the_period_and_unknown_sources_are_ignored():
    materials = [Material("a", "crossref", "2020-01"), Material("b", "unknown", MONTHS[3]),
                 Material("c", "crossref", "2026-09")]
    assert assess_curve(materials, AS_OF).materials == 0


def test_smoothing_is_a_centred_moving_average():
    assert smooth([0, 3, 0, 3]) == [1.5, 1.0, 2.0, 1.5]


def test_trust_levels_follow_source_coefficients():
    assert trust_level("crossref") == "высокий"
    assert trust_level("github") == "средний"
    assert trust_level("habr") == "пониженный"
    assert trust_level("blog") == "не определён"
