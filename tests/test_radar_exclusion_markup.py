"""The model's confidence remains visible when a hard rule excludes a topic."""

from types import SimpleNamespace

from app.ui.radar_view import exclusions_markup


def test_rule_exclusion_shows_model_score_and_rule_verdict():
    technology = SimpleNamespace(title="Mature cathode", probability=0.88,
                                 rule_excluded=True, reasons=("Зрелая тема",))
    result = SimpleNamespace(excluded=(technology,))

    markup = exclusions_markup(result)

    assert "Mature cathode" in markup
    assert "88% · исключено правилом" in markup
    assert "Зрелая тема" in markup
