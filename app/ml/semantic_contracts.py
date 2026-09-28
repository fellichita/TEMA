"""Optional semantic result extension; no inference dependencies are imported."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class SemanticContract(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)


class SemanticDecision(SemanticContract):
    study_id: str
    admitted: bool
    route: Literal["lexical", "semantic", "rejected"]
    lexical_admitted: bool
    semantic_only: bool
    similarity: float | None = Field(ge=-1, le=1)
    scored: bool
    requires_review: bool
    unscored_reason: str | None

    @model_validator(mode="after")
    def consistent_decision(self):
        if (self.admitted != (self.route != "rejected")
                or self.lexical_admitted != (self.route == "lexical")
                or self.semantic_only != (self.route == "semantic")
                or self.requires_review != self.semantic_only
                or self.scored != (self.similarity is not None)
                or self.scored != (self.unscored_reason is None)
                or (self.semantic_only and not self.scored)):
            raise ValueError("Несогласованное решение смысловой проверки.")
        return self


class SemanticModel(SemanticContract):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class SemanticSummary(SemanticContract):
    mode: Literal["semantic_assisted"]
    policy: Literal["lexical_or_semantic_generic"]
    policy_version: str
    threshold: float = Field(ge=0, le=1)
    calibrated: Literal[False]
    similarity_kind: Literal["cosine"]
    guarded_direction: bool
    model: SemanticModel
    text_policy: dict[str, JsonValue]
    language_policy: dict[str, JsonValue]
    publication_year_bounds: dict[str, int]
    skipped_occurrences_by_reason: dict[str, int]
    blocked_texts_by_reason: dict[str, int]
    input_occurrences: int = Field(ge=0)
    eligible_occurrences: int = Field(ge=0)
    scored_texts: int = Field(ge=0)
    retained_studies: int = Field(ge=0)
    lexical_studies: int = Field(ge=0)
    semantic_only_studies: int = Field(ge=0)
    unscored_studies: int = Field(ge=0)
    study_decisions: list[SemanticDecision]

    @model_validator(mode="after")
    def consistent_counts(self):
        rows = self.study_decisions
        ids = [row.study_id for row in rows]
        if (ids != sorted(set(ids)) or self.retained_studies != len(rows)
                or any(not row.admitted for row in rows)
                or self.lexical_studies != sum(row.lexical_admitted for row in rows)
                or self.semantic_only_studies != sum(row.semantic_only for row in rows)
                or self.unscored_studies != sum(not row.scored for row in rows)
                or self.scored_texts > self.eligible_occurrences
                or self.eligible_occurrences > self.input_occurrences
                or any(value < 0 for value in self.skipped_occurrences_by_reason.values())
                or any(value < 0 for value in self.blocked_texts_by_reason.values())
                or sum(self.skipped_occurrences_by_reason.values()) + self.eligible_occurrences
                   != self.input_occurrences
                or sum(self.blocked_texts_by_reason.values()) > sum(self.skipped_occurrences_by_reason.values())
                or (self.guarded_direction and self.semantic_only_studies)
                or any(row.semantic_only and (row.similarity is None or row.similarity < self.threshold)
                       for row in rows)):
            raise ValueError("Несогласованные счётчики смысловой проверки.")
        return self


class SemanticGroup(SemanticContract):
    semantic_only_documents: int = Field(ge=0)
    corpus_requires_review: bool
    requires_review: bool
    study_decisions: list[SemanticDecision]

    @model_validator(mode="after")
    def consistent_group(self):
        count = sum(row.semantic_only for row in self.study_decisions)
        if (self.semantic_only_documents != count
                or self.requires_review != bool(count or self.corpus_requires_review)):
            raise ValueError("Несогласованная смысловая проверка группы.")
        return self


def validate_semantic_result(result):
    """Validate new cross-field invariants without tightening historical lexical v2."""
    extension = result.get("semantic_relevance")
    selected = result.get("options", {}).get("relevance_mode", "lexical") == "semantic"
    if not selected:
        if extension is not None:
            raise ValueError("Смысловая проверка не соответствует выбранному режиму.")
        for bucket in ("candidates", "preliminary_signals", "established", "excluded_off_direction"):
            if any(group.get("semantic_relevance") is not None for group in result.get(bucket, [])):
                raise ValueError("Смысловая проверка группы не соответствует выбранному режиму.")
        return
    summary = SemanticSummary.model_validate(extension)
    by_id = {row.study_id: row for row in summary.study_decisions}
    if summary.retained_studies != result["preparation"]["retained_studies"]:
        raise ValueError("Смысловая проверка не соответствует подготовленному корпусу.")
    for bucket in ("candidates", "preliminary_signals", "established", "excluded_off_direction"):
        for candidate in result[bucket]:
            group = SemanticGroup.model_validate(candidate.get("semantic_relevance"))
            ids = [row.study_id for row in group.study_decisions]
            if (len(ids) != len(set(ids)) or sorted(ids) != candidate["study_ids"]
                    or group.corpus_requires_review != (summary.semantic_only_studies > 0)
                    or any(by_id.get(row.study_id) != row for row in group.study_decisions)
                    or (group.requires_review and bucket not in {
                        "preliminary_signals", "excluded_off_direction"})):
                raise ValueError("Смысловая проверка не соответствует документам или разделу группы.")
