"""Publish and validate the JSON v2 integration contract, without ML dependencies."""

import argparse
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from app.ml.semantic_contracts import SemanticGroup, SemanticSummary, validate_semantic_result


class Contract(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def finite_json_numbers(cls, value):
        """Pydantic's float constraint alone does not inspect untyped extensions."""
        pending, visited = [value], set()
        while pending:
            node = pending.pop()
            if isinstance(node, float) and not math.isfinite(node):
                raise ValueError("JSON результата не допускает NaN или Infinity, включая расширения схемы.")
            if isinstance(node, (dict, list, tuple)) and id(node) not in visited:
                visited.add(id(node))
                pending.extend(node.values() if isinstance(node, dict) else node)
        return value


class Quote(Contract):
    text: str = Field(min_length=1)
    study_id: str
    url: str = Field(pattern=r"^https?://")
    title: str
    mode: Literal["source_excerpt", "study_title"]


class Card(Contract):
    problem: Quote | None
    advantage: Quote | None
    example: Quote | None


class YearCount(Contract):
    year: int
    documents: int = Field(ge=0)
    direction_documents: int = Field(ge=0)
    share: float | None = Field(ge=0, le=1)


class Metrics(Contract):
    years: list[YearCount]
    first_observed_year_in_corpus: int
    first_observed_year_in_window: int | None
    growth_ratio: float | None = Field(ge=0)
    score: float = Field(ge=0, le=100)
    score_components: dict[str, float]
    score_weights: dict[str, int]
    growth_pattern: bool


class Guard(Contract):
    axis_check: Literal["supported", "partial", "off_direction"]
    reason: str


class Selection(Contract):
    bucket: Literal["candidates", "preliminary_signals", "established", "excluded_off_direction"]
    reasons: list[str]
    window_document_share: float | None = Field(ge=0, le=1)
    established_threshold: float = Field(strict=True, json_schema_extra={"const": 0.08})

    @field_validator("established_threshold", mode="before")
    @classmethod
    def fixed_established_threshold(cls, value: object) -> float:
        # Float values are not valid typing.Literal parameters. Retain the
        # exact legacy JSON constant and reject coercion or approximate matches.
        if type(value) is not float or value != 0.08:
            raise ValueError("established_threshold must be exactly the JSON number 0.08")
        return value


class Candidate(Contract):
    id: str
    title: str
    keywords: list[str]
    study_count: int = Field(ge=1)
    study_ids: list[str]
    metrics: Metrics
    card: Card
    sources: list[dict[str, JsonValue]]
    status: Literal["growth_candidate", "exploratory_candidate", "established", "off_direction"]
    stage: Literal["requires_review"]
    direction_guard: Guard | None = Field(default=None, exclude_if=lambda value: value is None)
    selection: Selection
    execution: dict[str, JsonValue]
    document_evidence: list[dict[str, JsonValue]]
    evidence_annotations: dict[str, JsonValue]
    explanations: dict[str, str]
    limitations: list[str]
    semantic_relevance: SemanticGroup | None = Field(default=None, exclude_if=lambda value: value is None)


class DirectionDenominator(Contract):
    year: int
    documents: int | None
    state: Literal["positive", "zero", "unknown", "invalid"]


class GrowthComparability(Contract):
    scope: Literal["retained_studies_in_saved_corpus"]
    calendar_complete: bool
    blocking_issues: dict[str, list[str]]
    denominators: list[DirectionDenominator]


class ResultContract(Contract):
    schema_version: Literal[2]
    pipeline_version: str
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    implementation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    options: dict[str, JsonValue]
    source: Literal["openalex", "crossref"]
    provenance: dict[str, JsonValue]
    preparation: dict[str, JsonValue]
    temporal_selection: dict[str, JsonValue]
    coverage: dict[str, list[str]]
    growth_data_comparable: bool
    growth_comparability: GrowthComparability | None = Field(default=None, exclude_if=lambda value: value is None)
    data_quality_notes: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    warnings: list[str]
    candidates: list[Candidate] = Field(max_length=15)
    preliminary_signals: list[Candidate]
    established: list[Candidate]
    excluded_off_direction: list[Candidate]
    selection_summary: dict[str, JsonValue]
    status: Literal["ranked", "no_groups", "insufficient_data"]
    semantic_relevance: SemanticSummary | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_semantic_extension(self):
        validate_semantic_result(self.model_dump())
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-schema", type=Path)
    group.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.write_schema:
        schema = ResultContract.model_json_schema()
        schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                       "$id": "urn:trendanalizer:result:2"})
        with args.write_schema.open("x", encoding="utf-8") as output:
            json.dump(schema, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    else:
        ResultContract.model_validate(json.loads(args.result.read_text(encoding="utf-8")))
        print("JSON соответствует контракту результата v2.")


if __name__ == "__main__":
    main()
