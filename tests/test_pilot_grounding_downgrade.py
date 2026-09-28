"""Current cards cannot select weaker automatic support policies from an import."""
import pytest

from app.pilot.contracts import AnalysisResult, Claim, TrendCard
from app.pilot.evidence import EvidenceError, quote_evidence
from app.pilot.export import verify_result
from app.pilot.history import assess_snapshot
from app.runtime.backup import ArchiveError
from tests.test_pilot_methodology_v3_4 import prepared34


ADVERSE = "Our device reduced measurement accuracy in laboratory experiments using lithium selective membranes."
METHODS = (
    ("exact-archived-quotation/1.0.0", "advantage", "3.0.0"),
    ("exact-contextual-quotation/2.0.0", "advantage", "3.1.0"),
    ("exact-contextual-quotation/3.0.0", "advantage", "3.3.0"),
    ("verified-application/research", "application", "3.1.0"),
)


def imported_claim(tmp_path, method, role, card_version):
    archive, source, discovery, context, checked, result = prepared34(tmp_path, abstract=ADVERSE)
    evidence = quote_evidence(discovery.documents[0], source, text_field="abstract", quote=ADVERSE)
    claim = Claim(claim_id="imported-old-grounding", role=role, support="supported", text=ADVERSE,
                  evidence_ids=(evidence.evidence_id,), grounding_method=method)
    card = TrendCard(candidate=checked.card.candidate, methodology_version=card_version,
        category="early_signal" if card_version == "3.0.0" else "unassessed_cluster",
        quality="partial", claims=(claim,), evidence=(evidence,),
        limitations=("Synthetic import policy regression; no scientific judgment is asserted.",))
    package = AnalysisResult.model_validate(result.model_dump(mode="python") | dict(
        snapshots=(discovery,), cards=(card,), quality="partial", top_trend_ids=()))
    return archive, context, checked, package


@pytest.mark.parametrize("method,role,legacy_version", METHODS)
def test_new_card_rejects_old_automatic_grounding_during_export(tmp_path, method, role, legacy_version):
    archive, _, _, result = imported_claim(tmp_path, method, role, "3.4.0")
    with pytest.raises(ArchiveError):
        verify_result(result, archive)


@pytest.mark.parametrize("method,role,legacy_version", METHODS)
def test_new_assessment_rejects_old_automatic_grounding(tmp_path, method, role, legacy_version):
    archive, context, checked, result = imported_claim(tmp_path, method, role, "3.4.0")
    with pytest.raises(EvidenceError):
        assess_snapshot(result.cards[0].candidate, result.query_plan, checked.snapshot, archive, context,
                        passport=result.cards[0], methodology_version="3.4.0")


@pytest.mark.parametrize("method,role,legacy_version", METHODS)
def test_older_card_in_new_container_retains_its_frozen_grounding(tmp_path, method, role, legacy_version):
    archive, _, _, result = imported_claim(tmp_path, method, role, legacy_version)
    encoded = result.cards[0].model_dump_json()
    verified, _ = verify_result(result, archive)
    assert verified.methodology_version == "3.4.0"
    assert verified.cards[0].model_dump_json() == encoded
