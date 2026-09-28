"""Offline result transport verifies source versions, quotations and arithmetic."""

import json
from pathlib import Path
from threading import Event

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, Claim
from app.pilot.evidence import build_passport, quote_evidence
from app.pilot.export import export_result, read_result_package, verify_result
from app.pilot.methodology import AssessmentArtifact, AssessmentInput, evaluate_candidate
from app.pilot.history import assess_snapshot
from app.runtime.backup import ArchiveError, unpack_package, write_package
from app.runtime.jobs import TaskCancelled
from tests.test_pilot_evidence import NOW, Context, candidate, document, query_plan, snapshot
from tests.test_pilot_history import frozen


def make_result(tmp_path, *, historical=False, numeric_text=None, metric="recent_studies", source="openalex", raw_metadata=None):
    archive = DocumentArchive(tmp_path / "revisions")
    changes = {"source": source}
    if raw_metadata is not None:
        changes["raw_metadata"] = raw_metadata
    docs = tuple(document(1000 + year * 10 + index, year=year, **changes) for year, count in zip(
        range(2020, 2026), (1, 1, 1, 2, 4, 8), strict=True) for index in range(count))
    discovery = snapshot(docs[-4:], archive)
    discovered = frozen(candidate(discovery))
    context = Context()
    passport = build_passport(discovered, discovery, archive, context)
    if not historical:
        output = AnalysisResult(result_id="result-one", run_id="run-one", query_plan=query_plan(), created_at=NOW,
            quality="partial", snapshots=(discovery,), cards=(passport,), limitations=("History incomplete",))
        return output, archive, ()
    if numeric_text is not None:
        claim = Claim(claim_id="count", role="summary", kind="numeric", text=numeric_text,
            support="unverified", grounding_method="computed-v3", evidence_ids=(passport.evidence[0].evidence_id,), metric_refs=(metric,))
        passport = type(passport).model_validate(passport.model_dump(mode="python") | {"claims": (*passport.claims, claim)})
    historical_snapshot = snapshot(docs, archive, purpose="history")
    artifact, trend, _ = assess_snapshot(discovered, query_plan(), historical_snapshot, archive, context, passport=passport)
    output = AnalysisResult(result_id="result-one", run_id="run-one", query_plan=query_plan(), created_at=NOW,
        quality=trend.quality, snapshots=(discovery, historical_snapshot), cards=(trend,))
    return output, archive, (artifact,)


@pytest.mark.parametrize("historical", [False, True])
def test_result_package_roundtrip_needs_only_verified_files(tmp_path, historical, monkeypatch):
    output, archive, artifacts = make_result(tmp_path, historical=historical)
    saved = export_result(tmp_path / "result.zip", output, archive, artifacts)
    # Explicitly prove import performs no model/network/credentials work.
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("Offline import attempted network")

    monkeypatch.setattr(socket, "socket", no_network)
    with read_result_package(saved.path) as package:
        assert package.result == output
        assert package.assessments == artifacts
        revision = output.snapshots[0].documents[0].revision_id
        assert package.archive.get(revision) == archive.get(revision)
        temporary = package.archive.directory.parent
    assert not temporary.exists()


def test_historical_card_without_assessment_cannot_be_exported(tmp_path):
    output, archive, _ = make_result(tmp_path, historical=True)
    with pytest.raises(ArchiveError):
        export_result(tmp_path / "bad.zip", output, archive)


def test_forged_quote_with_valid_offsets_fails_against_exact_archived_field(tmp_path):
    output, archive, artifacts = make_result(tmp_path)
    changed = output.model_dump(mode="json")
    changed["cards"][0]["evidence"][0].update(quote="Invented", start=0, end=8)
    from app.pilot.contracts import AnalysisResult
    forged = AnalysisResult.model_validate(changed)
    with pytest.raises(ArchiveError):
        export_result(tmp_path / "bad.zip", forged, archive, artifacts)


def test_exact_quotation_claim_cannot_be_changed_into_an_invented_summary(tmp_path):
    output, archive, artifacts = make_result(tmp_path)
    claim = Claim(claim_id="invented", role="summary", text="Invented company built an impossible product.",
        support="supported", grounding_method="exact-archived-quotation/1.0.0", evidence_ids=(output.cards[0].evidence[0].evidence_id,))
    forged = output.model_copy(update={"cards": (output.cards[0].model_copy(update={"claims": (claim,)}),)})
    with pytest.raises(ArchiveError):
        verify_result(forged, archive, artifacts)


def claim_result(tmp_path, abstract, quote, *, role="advantage", method="exact-contextual-quotation/2.0.0"):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1, abstract=abstract), document(2, abstract=abstract))
    discovery = snapshot(docs, archive)
    passport = build_passport(frozen(candidate(discovery)), discovery, archive, Context())
    evidence = quote_evidence(discovery.documents[0], docs[0], text_field="abstract", quote=quote)
    claim = Claim(claim_id="claim-context", role=role, text=quote, support="supported",
                  evidence_ids=(evidence.evidence_id,), grounding_method=method)
    changed = passport.model_copy(update={"claims": (claim,), "evidence": (evidence,)})
    return AnalysisResult(result_id="source-claim", run_id="source-run", query_plan=query_plan(), created_at=NOW,
        quality="partial", snapshots=(discovery,), cards=(changed,), limitations=("History not evaluated",)), archive


@pytest.mark.parametrize("abstract,quote,role,method", [
    ("Previous research hypothesized that protein motion enhances magnetic sensing.",
     "Previous research hypothesized that protein motion enhances magnetic sensing.", "advantage", "exact-contextual-quotation/2.0.0"),
    ("Our membrane reduces energy consumption in experiments. However, this effect is likely negligible in realistic systems.",
     "Our membrane reduces energy consumption in experiments.", "advantage", "exact-contextual-quotation/2.0.0"),
    ("Our membrane does not reduce energy consumption in experiments.",
     "reduce energy consumption in experiments.", "advantage", "exact-contextual-quotation/2.0.0"),
    ("Our membrane reduces energy consumption.", "Our membrane reduces energy consumption.",
     "application", "verified-application/research"),
    ("Our membrane reduces energy consumption in experiments.", "Our membrane reduces energy consumption in experiments.",
     "application", "verified-application/deployment"),
    ("The system has fundamental limitations.", "The system has fundamental limitations.",
     "advantage", "exact-contextual-quotation/2.0.0"),
    ("Our membrane reduces energy consumption in experiments.", "Our membrane reduces energy consumption in experiments.",
     "summary", "verified-application/research"),
    ("The system has fundamental limitations.", "The system has fundamental limitations.",
     "advantage", "self-declared-verifier/1.0.0"),
])
def test_import_replays_context_and_application_role_not_just_exact_quote(tmp_path, abstract, quote, role, method):
    output, archive = claim_result(tmp_path, abstract, quote, role=role, method=method)
    with pytest.raises(ArchiveError):
        verify_result(output, archive)


def test_contextual_forgery_cannot_pass_by_rehashing_the_package(tmp_path):
    quote = "Previous research hypothesized that protein motion enhances magnetic sensing."
    output, archive = claim_result(tmp_path, quote, quote)
    claim = output.cards[0].claims[0].model_copy(update={"support": "unverified"})
    ordinary = output.model_copy(update={"cards": (output.cards[0].model_copy(update={"claims": (claim,)}),)})
    saved = export_result(tmp_path / "unverified.zip", ordinary, archive)
    forged = _rewrite_result_package(saved.path, tmp_path / "context-forged.zip",
        lambda data: data["cards"][0]["claims"][0].update(support="supported"))
    with pytest.raises(ArchiveError):
        read_result_package(forged.path)


@pytest.mark.parametrize("version", [None, "3.0.0"])
def test_legacy_exact_provenance_is_preserved_but_cannot_bypass_new_context_rules(tmp_path, version):
    quote = "Previous research hypothesized that protein motion enhances magnetic sensing."
    output, archive = claim_result(tmp_path, quote, quote, method="exact-archived-quotation/1.0.0")
    old_card = output.cards[0].model_copy(update={"methodology_version": version, "category": "early_signal"})
    old = output.model_copy(update={"methodology_version": "3.0.0", "cards": (old_card,)})
    verify_result(old, archive)
    with pytest.raises(ArchiveError):
        verify_result(output, archive)


@pytest.mark.parametrize("version", [None, "3.0.0"])
def test_new_assessment_cannot_be_attached_to_card_claiming_legacy_version(tmp_path, version):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    forged = output.model_copy(update={"cards": (output.cards[0].model_copy(update={"methodology_version": version}),)})
    with pytest.raises(ArchiveError):
        verify_result(forged, archive, artifacts)


def test_legacy_history_replay_accepts_absent_or_explicit_matching_card_version(tmp_path):
    output, archive, _ = make_result(tmp_path, historical=True)
    card = output.cards[0]
    historical = next(item for item in output.snapshots if item.purpose == "history")
    artifact, legacy_card, _ = assess_snapshot(card.candidate, output.query_plan, historical, archive,
        Context(), passport=card, methodology_version="3.0.0")
    legacy = output.model_copy(update={"methodology_version": "3.0.0", "cards": (legacy_card,)})
    verify_result(legacy, archive, (artifact,))
    explicit = legacy.model_copy(update={"cards": (legacy_card.model_copy(update={"methodology_version": "3.0.0"}),)})
    verify_result(explicit, archive, (artifact,))


def test_changed_document_bytes_are_rejected_even_when_quote_still_matches(tmp_path):
    output, archive, artifacts = make_result(tmp_path)
    revision = output.snapshots[0].documents[0].revision_id
    path = archive.path(revision)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authors"] = ["Invented author"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ArchiveError):
        export_result(tmp_path / "bad.zip", output, archive, artifacts)


def test_self_consistent_arithmetic_using_a_forged_publication_year_is_rejected(tmp_path):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    inputs = artifacts[0].inputs.model_dump(mode="json")
    moved = inputs["history"]["observations"][0]["study_ids"].pop()
    inputs["history"]["observations"][1]["study_ids"].append(moved)
    inputs["history"]["first_observed_year"] = 2021
    changed_input = AssessmentInput.model_validate(inputs)
    changed = AssessmentArtifact(inputs=changed_input, assessment=evaluate_candidate(changed_input))
    updated_card = output.cards[0].model_copy(update={"assessment_hash": changed.assessment.assessment_hash})
    forged = output.model_copy(update={"cards": (updated_card,)})
    with pytest.raises(ArchiveError):
        verify_result(forged, archive, (changed,))


def test_model_copy_cannot_bypass_assessment_recomputation(tmp_path):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    forged = artifacts[0].model_copy(update={"assessment": artifacts[0].assessment.model_copy(update={"recent_studies": 999})})
    with pytest.raises(ArchiveError):
        verify_result(output, archive, (forged,))


def test_unsupported_historical_admission_version_has_actionable_error(tmp_path):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    candidate = output.cards[0].candidate.model_copy(update={"admission_rule_version": "future-rule/99"})
    inputs = AssessmentInput.model_validate(artifacts[0].inputs.model_dump(mode="python") | {"candidate": candidate})
    changed = AssessmentArtifact(inputs=inputs, assessment=evaluate_candidate(inputs))
    changed_card = output.cards[0].model_copy(update={"candidate": candidate, "assessment_hash": changed.assessment.assessment_hash})
    with pytest.raises(ArchiveError, match="Версия исторического отбора"):
        verify_result(output.model_copy(update={"cards": (changed_card,)}), archive, (changed,))


@pytest.mark.parametrize("metric,text", [
    ("recent_studies", "За последние три года: 14 публикаций."),
    ("smoothed_growth", "Рост в 3,75 раза."),
    ("smoothed_growth", "Рост в 3,8 раза."),
])
def test_numeric_claims_resolve_known_metrics_and_honest_rounding(tmp_path, metric, text):
    output, archive, artifacts = make_result(tmp_path, historical=True, numeric_text=text, metric=metric)
    verify_result(output, archive, artifacts)


@pytest.mark.parametrize("metric,text", [
    ("recent_studies", "999 публикаций."), ("unknown_statistic", "14 публикаций."),
    ("smoothed_growth", "Рост в 99 раз."), ("category", "14 публикаций."),
])
def test_unreferenced_or_changed_numeric_claims_are_rejected(tmp_path, metric, text):
    output, archive, artifacts = make_result(tmp_path, historical=True, numeric_text=text, metric=metric)
    with pytest.raises(ArchiveError):
        verify_result(output, archive, artifacts)


@pytest.mark.parametrize("metadata", [{"api_key": "private-key"}, {"nested": {"access_token": "private"}}, {"full_text": "licensed report"}])
def test_keys_and_unlicensed_full_text_are_not_exported(tmp_path, metadata):
    output, archive, artifacts = make_result(tmp_path, raw_metadata=metadata)
    with pytest.raises(ArchiveError):
        export_result(tmp_path / "private.zip", output, archive, artifacts)
    assert not (tmp_path / "private.zip").exists()


@pytest.mark.parametrize("name", ["yandex_api_key", "yandex_iam_token", "openai_api_key", "YANDEX_CLOUD_API_KEY"])
def test_supported_provider_keys_in_nested_metadata_cannot_enter_shared_result(tmp_path, name):
    private = "synthetic-export-marker-not-a-real-key"
    output, archive, artifacts = make_result(tmp_path, raw_metadata={"provider": [{name: private}]})
    destination = tmp_path / "private.trendresult"
    with pytest.raises(ArchiveError) as caught:
        export_result(destination, output, archive, artifacts)
    assert private not in str(caught.value)
    assert not destination.exists()


def test_source_inverted_index_words_are_not_mistaken_for_stored_credentials(tmp_path):
    output, archive, artifacts = make_result(tmp_path, raw_metadata={"abstract_inverted_index": {"secret": [3], "full_text": [4], "api_key": [5]}})
    verify_result(output, archive, artifacts)


def test_credential_bearing_metadata_urls_are_not_shared(tmp_path):
    output, archive, artifacts = make_result(tmp_path, raw_metadata={"link": "https://example.com/document?api_key=private"})
    with pytest.raises(ArchiveError):
        export_result(tmp_path / "private.zip", output, archive, artifacts)


def test_public_kiss_article_identifier_does_not_block_result_export(tmp_path):
    metadata = {"resource": {"primary": {"URL": "https://kiss.kstudy.com/Detail/Ar?key=1234567"}}}
    output, archive, artifacts = make_result(tmp_path, source="crossref", raw_metadata=metadata)
    saved = export_result(tmp_path / "public-article.trendresult", output, archive, artifacts)
    with read_result_package(saved.path) as package:
        assert package.result == output


def test_report_records_require_explicit_rights_format(tmp_path):
    from app.pilot.reports import ReportPage, ReportRecord

    report = ReportRecord(source_id="a" * 64, title="Published report", publication_year=2025, date_precision="year",
        url="https://example.org/report.pdf", full_text="Licensed report text.",
        page_spans=(ReportPage(page=1, start=0, end=21),), pages_total=1, pages_with_text=1,
        extraction_status="text", pdf_sha256="a" * 64, public_license_allowed=True, license_note="CC BY 4.0")
    archive = DocumentArchive(tmp_path / "revisions")
    output = AnalysisResult(result_id="result-report", run_id="run-report", query_plan=query_plan(), created_at=NOW,
        quality="insufficient_data", snapshots=(snapshot((report,), archive),), cards=(), limitations=("History not evaluated",))
    with pytest.raises(ArchiveError):
        export_result(tmp_path / "report.zip", output, archive)


def test_public_report_bibliographic_genre_is_not_confused_with_report_full_text(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    record = document(1, source="crossref", document_type="report",
        title="Quantum Enhanced Tracker (Final Report)", doi="10.2172/1963593")
    output = AnalysisResult(result_id="report-metadata", run_id="report-metadata", query_plan=query_plan(), created_at=NOW,
        quality="insufficient_data", snapshots=(snapshot((record,), archive),), cards=(), limitations=("Not assessed",))
    exported = export_result(tmp_path / "metadata-report.trendresult", output, archive)
    with read_result_package(exported.path) as package:
        assert package.archive.get(output.snapshots[0].documents[0].revision_id) == record


def test_unrelated_evidence_url_cannot_pass_with_a_correct_quote(tmp_path):
    output, archive, artifacts = make_result(tmp_path)
    changed_evidence = output.cards[0].evidence[0].model_copy(update={"source_url": "https://example.com/unrelated"})
    forged = output.model_copy(update={"cards": (output.cards[0].model_copy(update={"evidence": (changed_evidence,)}),)})
    with pytest.raises(ArchiveError):
        verify_result(forged, archive, artifacts)


def _rewrite_result_package(original: Path, target: Path, change):
    with unpack_package(original, expected_kind="trendanalizer-result") as (directory, _):
        path = directory / "result.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data))
        files = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*") if path.is_file()}
        return write_package(target, files, kind="trendanalizer-result")


def test_forgery_is_rejected_even_after_all_zip_and_manifest_hashes_are_recomputed(tmp_path):
    output, archive, artifacts = make_result(tmp_path)
    saved = export_result(tmp_path / "valid.zip", output, archive, artifacts)

    def forge(data):
        data["cards"][0]["evidence"][0].update(quote="Invented", start=0, end=8)

    forged = _rewrite_result_package(saved.path, tmp_path / "forged.zip", forge)
    with pytest.raises(ArchiveError):
        read_result_package(forged.path)


def test_result_package_rejects_added_budget_database(tmp_path):
    output, archive, artifacts = make_result(tmp_path)
    saved = export_result(tmp_path / "valid.zip", output, archive, artifacts)
    with unpack_package(saved.path, expected_kind="trendanalizer-result") as (directory, _):
        (directory / "pilot.sqlite3").write_bytes(b"must not share budget")
        files = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*") if path.is_file()}
        extra = write_package(tmp_path / "extra.zip", files, kind="trendanalizer-result")
    with pytest.raises(ArchiveError):
        read_result_package(extra.path)


def make_reviewed_result(tmp_path, *, field_reviews=False):
    from app.pilot.antecedents import collect_antecedents
    from app.pilot.review import apply_novelty_review, record_review
    from app.runtime.credentials import CredentialStore
    from tests.test_pilot_antecedents import Provider
    from tests.test_pilot_review import decision_for, field_review_for

    output, archive, _ = make_result(tmp_path, historical=True)
    source_card = output.cards[0]
    old = document(81, year=2010)
    bundle = collect_antecedents(source_card.candidate, output.query_plan, archive, CredentialStore(), Context(),
        provider_factory=lambda _: Provider((old,)))
    decision = decision_for(bundle, source_card, field_reviews=(
        field_review_for(source_card), field_review_for(source_card, "application", application_kind="research"))
        if field_reviews else ())
    expanded, novelty = apply_novelty_review(decision, bundle, source_card, archive)
    historical = next(item for item in output.snapshots if item.purpose == "history")
    artifact, card, _ = assess_snapshot(source_card.candidate, output.query_plan, historical, archive,
        Context(), passport=expanded, verified_novelty=novelty, antecedents=bundle)
    record = record_review(tmp_path / "reviews", decision, bundle, source_card=source_card, reviewed_card=card,
        artifact=artifact, historical_snapshot=historical, archive=archive)
    result = AnalysisResult.model_validate(output.model_dump(mode="python") | {
        "cards": (card,), "snapshots": (*output.snapshots, bundle.snapshot), "quality": card.quality})
    return result, archive, (artifact,), (record,)


def test_reviewed_confirmed_result_requires_full_expert_record_not_a_claim_prefix(tmp_path):
    output, archive, artifacts, reviews = make_reviewed_result(tmp_path)
    assert output.cards[0].category == "confirmed_trend"
    with pytest.raises(ArchiveError, match="ReviewRecord"):
        export_result(tmp_path / "missing-review.zip", output, archive, artifacts)
    verify_result(output, archive, artifacts, reviews=reviews)


def test_reviewed_result_roundtrip_keeps_all_early_sources_and_requires_no_network(tmp_path, monkeypatch):
    output, archive, artifacts, reviews = make_reviewed_result(tmp_path)
    saved = export_result(tmp_path / "reviewed.zip", output, archive, artifacts, reviews=reviews)
    import socket

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("Review import must be offline"))
    with read_result_package(saved.path) as package:
        assert package.reviews == reviews and package.result == output and package.assessments == artifacts
        for reference in reviews[0].bundle.snapshot.documents:
            assert package.archive.get(reference.revision_id) == archive.get(reference.revision_id)


def test_manual_field_reviews_roundtrip_and_cannot_be_replaced_without_replaying_claims(tmp_path):
    output, archive, artifacts, reviews = make_reviewed_result(tmp_path, field_reviews=True)
    saved = export_result(tmp_path / "field-review.zip", output, archive, artifacts, reviews=reviews)
    with read_result_package(saved.path) as package:
        assert package.reviews == reviews
        assert package.assessments[0].inputs.application.kind == "research"
        assert any(claim.grounding_method == "reviewed-evidence/manual-v1" for claim in package.result.cards[0].claims)
    changed_decision = reviews[0].decision.model_copy(update={"field_reviews": (
        reviews[0].decision.field_reviews[0].model_copy(update={"verdict": "contradicted"}),)})
    forged = reviews[0].model_copy(update={"decision": changed_decision})
    with pytest.raises(ArchiveError):
        verify_result(output, archive, artifacts, reviews=(forged,))


@pytest.mark.parametrize("verdict", ["supported", "unverified", "contradicted"])
def test_any_attributed_field_verdict_requires_its_review_record(tmp_path, verdict):
    output, archive, _ = make_result(tmp_path)
    claim = output.cards[0].claims[0].model_copy(update={"grounding_method": "reviewed-evidence/manual-v1",
                                                       "support": verdict})
    forged = output.model_copy(update={"cards": (output.cards[0].model_copy(update={"claims": (claim,)}),)})
    with pytest.raises(ArchiveError, match="ReviewRecord"):
        verify_result(forged, archive)


def test_real_quote_cannot_borrow_another_candidates_study_identity(tmp_path):
    output, archive, _ = make_result(tmp_path)
    original = output.cards[0]
    mismatched = original.evidence[0].model_copy(update={"study_id": original.candidate.discovery_study_ids[-1]})
    forged = output.model_copy(update={"cards": (original.model_copy(update={"evidence": (mismatched, *original.evidence[1:])}),)})
    with pytest.raises(ArchiveError):
        verify_result(forged, archive)


def test_review_with_changed_authored_reason_or_unused_review_is_rejected(tmp_path):
    from app.pilot.review import ReviewRecord

    output, archive, artifacts, reviews = make_reviewed_result(tmp_path)
    changed = reviews[0].model_dump(mode="json")
    changed["decision"]["rationale"] = "This altered expert rationale is not the one used by the published assessment."
    forged = ReviewRecord.model_validate(changed)
    with pytest.raises(ArchiveError):
        verify_result(output, archive, artifacts, reviews=(forged,))
    ordinary, ordinary_archive, ordinary_artifacts = make_result(tmp_path / "ordinary", historical=True)
    with pytest.raises(ArchiveError):
        verify_result(ordinary, ordinary_archive, ordinary_artifacts, reviews=reviews)


def test_missing_review_file_is_rejected_even_with_recomputed_zip_manifest(tmp_path):
    output, archive, artifacts, reviews = make_reviewed_result(tmp_path)
    saved = export_result(tmp_path / "valid-reviewed.zip", output, archive, artifacts, reviews=reviews)
    with unpack_package(saved.path, expected_kind="trendanalizer-result") as (directory, _):
        (directory / "reviews.json").unlink()
        files = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*") if path.is_file()}
        forged = write_package(tmp_path / "removed-review.zip", files, kind="trendanalizer-result")
    with pytest.raises(ArchiveError):
        read_result_package(forged.path)


def test_duplicate_candidate_reviews_are_rejected(tmp_path):
    output, archive, artifacts, reviews = make_reviewed_result(tmp_path)
    with pytest.raises(ArchiveError):
        verify_result(output, archive, artifacts, reviews=reviews * 2)


def test_cancelled_package_read_stops_before_opening_input(tmp_path):
    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        read_result_package(tmp_path / "does-not-exist.zip", cancel=cancel)


def test_verification_checks_cancel_between_document_revisions(tmp_path, monkeypatch):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    cancel = Event()
    original = archive.get
    reads = []

    def read_and_cancel(digest):
        reads.append(digest)
        document = original(digest)
        cancel.set()
        return document

    monkeypatch.setattr(archive, "get", read_and_cancel)
    with pytest.raises(TaskCancelled):
        verify_result(output, archive, artifacts, cancel=cancel)
    assert len(reads) == 1


def test_cancelled_evidence_replay_cleans_unpacked_quarantine(tmp_path, monkeypatch):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    saved = export_result(tmp_path / "valid.zip", output, archive, artifacts)
    cancel = Event()
    original = DocumentArchive.get
    temporary = []

    def read_and_cancel(self, digest):
        document = original(self, digest)
        temporary.append(self.directory.parent)
        cancel.set()
        return document

    monkeypatch.setattr(DocumentArchive, "get", read_and_cancel)
    with pytest.raises(TaskCancelled):
        read_result_package(saved.path, cancel=cancel)
    assert len(temporary) == 1
    assert not temporary[0].exists()
    assert saved.path.exists()


def test_export_cancelled_after_verification_does_not_publish_file(tmp_path, monkeypatch):
    output, archive, artifacts = make_result(tmp_path, historical=True)
    original = archive.path
    cancel = Event()

    def path_and_cancel(digest):
        cancel.set()
        return original(digest)

    # get() accesses paths internally too: arm only at packaging, after replay.
    from app.pilot import export
    original_verify = export.verify_result

    def verify_then_arm(*args, **kwargs):
        verified = original_verify(*args, **kwargs)
        monkeypatch.setattr(archive, "path", path_and_cancel)
        return verified

    monkeypatch.setattr(export, "verify_result", verify_then_arm)
    with pytest.raises(TaskCancelled):
        export_result(tmp_path / "cancelled.zip", output, archive, artifacts, cancel=cancel)
    assert not (tmp_path / "cancelled.zip").exists()


def test_external_context_cancellation_remains_cancellation(tmp_path):
    output, archive, artifacts = make_result(tmp_path)

    class CancelledContext:
        def check_cancelled(self):
            raise TaskCancelled("The window closed")

    with pytest.raises(TaskCancelled, match="window closed"):
        verify_result(output, archive, artifacts, context=CancelledContext())


def test_one_deadline_includes_unpacking_and_replay(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from app.pilot import export

    output, archive, artifacts = make_result(tmp_path)
    saved = export_result(tmp_path / "valid.zip", output, archive, artifacts)
    clock = [0.0]
    original = export.unpack_package
    temporary = []

    @contextmanager
    def slow_unpack(*args, **kwargs):
        with original(*args, **kwargs) as (directory, manifest):
            temporary.append(directory)
            clock[0] = 601.0  # Past the verification deadline, whatever its budget.
            yield directory, manifest

    monkeypatch.setattr(export.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(export, "unpack_package", slow_unpack)
    with pytest.raises(ArchiveError, match="допустимое время"):
        read_result_package(saved.path)
    assert len(temporary) == 1 and not temporary[0].exists()
