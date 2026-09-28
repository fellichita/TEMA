"""Replay automatic claim support consistently in histories and portable results."""

from app.backend.contracts import DocumentRecord
from app.pilot.contracts import Claim, Evidence, MethodologyVersion
from app.pilot.evidence import QuoteSelection, _selection_supported, _EXPERIMENT

_REVIEWED_METHODS = ("reviewed-novelty/", "reviewed-evidence/", "reviewed-application/")


def verify_supported_source_claim(claim: Claim, evidence: dict[str, Evidence],
                                   documents: dict[str, DocumentRecord], *, legacy: bool,
                                   candidate_studies: set[str],
                                   methodology_version: MethodologyVersion | None = None) -> None:
    """Repeat the producer's role/context checks; a method string is not proof."""
    if claim.support != "supported" or (claim.grounding_method or "").startswith(_REVIEWED_METHODS):
        return  # Attributed methods require a full independently replayed ReviewRecord above.
    method = claim.grounding_method
    if methodology_version == "3.4.0" and method not in {
            "exact-contextual-quotation/4.0.0", "verified-application/research/2.0.0"}:
        raise ValueError("Automatic claim grounding must match methodology 3.4")
    if method not in {"exact-archived-quotation/1.0.0", "exact-contextual-quotation/2.0.0",
                       "exact-contextual-quotation/3.0.0", "exact-contextual-quotation/4.0.0",
                       "verified-application/research", "verified-application/research/2.0.0"}:
        raise ValueError("Unsupported automatic claim grounding method")
    if len(claim.evidence_ids) != 1:
        raise ValueError("An extractive claim must refer to exactly one quotation")
    source = evidence[claim.evidence_ids[0]]
    if source.study_id not in candidate_studies:
        raise ValueError("Candidate claim borrows a source from a different technological group")
    if claim.text != source.quote:
        raise ValueError("An exact quotation claim was changed into an unverified paraphrase")
    if legacy and method == "exact-archived-quotation/1.0.0":
        return  # Archived v3.0 asserted provenance, not the newer semantic screen.
    document = documents[source.revision_id]
    application = method in {"verified-application/research", "verified-application/research/2.0.0"}
    role = "advantage" if application else claim.role
    if ((application and claim.role != "application") or role not in {"problem", "advantage", "case"}
            or source.text_field not in {"title", "abstract"}):
        raise ValueError("Source grounding method does not support this claim role or text field")
    selection = QuoteSelection.model_validate({"role": role, "revision_id": source.revision_id,
                                               "field": source.text_field, "quote": source.quote})
    grounding = (method if method in {"exact-contextual-quotation/3.0.0", "exact-contextual-quotation/4.0.0"}
                 else "exact-contextual-quotation/4.0.0" if method == "verified-application/research/2.0.0"
                 else "exact-contextual-quotation/2.0.0")
    if grounding == "exact-contextual-quotation/4.0.0":
        from app.pilot.sentences import sentence_spans

        if source.text_field != "abstract" or (source.start, source.end) not in sentence_spans(document.abstract or ""):
            raise ValueError("The quotation must reference a complete original sentence at its exact source offsets")
    if not _selection_supported(selection, document, method_version=grounding):
        raise ValueError("Source quotation/context does not support the asserted field or research application")
    if application:
        if method == "verified-application/research/2.0.0":
            from app.pilot.evidence import automatic_research_application

            supported = automatic_research_application(source.quote, document, method_version=grounding)
        else:
            supported = bool(_EXPERIMENT.search(source.quote))
        if not supported:
            raise ValueError("Source does not support the claimed experimental research application")
