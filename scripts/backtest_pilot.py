"""Prepare and replay an auditable as-of corpus without claiming predictive success.

This is a local, network-free backtest boundary. A caller must supply a frozen
candidate-discovery function and independently labeled later outcomes to measure
lead time or prediction quality. Filtering publication years alone is explicitly
reported as reconstructed history, never a strict causal experiment.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.backend.contracts import DocumentRecord
from scripts.evaluate_pilot import EvaluationError, digest, read_json, write_json


class FrozenAsOfConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cutoff: date
    mode: Literal["strict", "reconstructed"]
    query_known_at: date
    aliases_known_at: date
    rules_frozen_at: date
    encoder_training_cutoff: date | None = None
    encoder_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    rules_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def limitations(self) -> list[str]:
        if self.query_known_at > self.cutoff or self.aliases_known_at > self.cutoff:
            raise EvaluationError("Historical discovery cannot use a future query or alias registry")
        concerns = []
        if self.rules_frozen_at > self.cutoff:
            concerns.append("Retrospectively designed rules may encode later technological knowledge.")
        if self.encoder_training_cutoff is None or self.encoder_training_cutoff > self.cutoff:
            concerns.append("Encoder training data are not proven to predate the cutoff.")
        if self.mode == "strict" and concerns:
            raise EvaluationError("Strict replay requires dated pre-cutoff rules and encoder training provenance")
        if self.mode == "reconstructed":
            concerns.append("Modern metadata revisions are allowed: reconstructed history, not a strict causal backtest.")
        return concerns


_LATER_AGGREGATES = {"cited_by_count", "counts_by_year", "citation_count", "is_retracted", "updated_date", "referenced_works"}


def prepare_asof(documents: list[DocumentRecord], configuration: FrozenAsOfConfiguration) -> tuple[tuple[DocumentRecord, ...], dict[str, Any]]:
    """Exclude unknown availability and future revisions before discovery can see them."""
    limitations = configuration.limitations()
    cutoff = configuration.cutoff
    rejected = {"unknown_availability": 0, "future_availability": 0, "late_observation": 0}
    accepted = []
    for document in documents:
        available = document.publication_date
        if available is None:
            rejected["unknown_availability"] += 1
            continue
        if available > cutoff:
            rejected["future_availability"] += 1
            continue
        if document.fetched_at.date() > cutoff and configuration.mode == "strict":
            rejected["late_observation"] += 1
            continue
        values = document.model_dump(mode="python")
        if configuration.mode == "reconstructed":
            # Current cumulative citations and retrospective retraction labels
            # cannot be fed into a historical prediction. This does not make
            # the remaining modern metadata causally clean; retain limitation.
            values["raw_metadata"] = {key: value for key, value in document.raw_metadata.items() if key not in _LATER_AGGREGATES}
            if "cited_by_count" in values:
                values["cited_by_count"] = None
            if "citation_count" in values:
                values["citation_count"] = None
        accepted.append(DocumentRecord.model_validate(values))
    # All version choice occurs after the temporal filter, so a later revision
    # cannot replace a genuinely historical record of the same source work.
    selected: dict[tuple[str, str], DocumentRecord] = {}
    for document in accepted:
        key = (document.source, document.source_id)
        previous = selected.get(key)
        if previous is None or (document.fetched_at, digest(document.model_dump(mode="json"))) > (
                previous.fetched_at, digest(previous.model_dump(mode="json"))):
            selected[key] = document
    prepared = tuple(selected[key] for key in sorted(selected))
    corpus_digest = digest([document.model_dump(mode="json") for document in prepared])
    return prepared, {"schema_version": 1, "mode": configuration.mode, "cutoff": cutoff.isoformat(),
        "configuration_hash": digest(configuration.model_dump(mode="json")), "corpus_hash": corpus_digest,
        "accepted_revisions": len(prepared), "excluded": rejected,
        "status": "corpus_prepared_not_scientifically_evaluated", "limitations": limitations,
        "precision": None, "recall": None, "lead_time": None}


def replay_at_cutoff(documents: list[DocumentRecord], configuration: FrozenAsOfConfiguration,
                     discover: Callable[[tuple[DocumentRecord, ...]], Any]) -> dict[str, Any]:
    prepared, report = prepare_asof(documents, configuration)
    # Only prepared records reach the supplied frozen candidate generator.
    predictions = discover(prepared)
    return report | {"predictions": predictions, "prediction_hash": digest(predictions),
                     "status": "predictions_replayed_outcomes_not_evaluated"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON containing configuration and archived DocumentRecord documents")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = read_json(args.input)
        if not isinstance(raw, dict) or set(raw) != {"configuration", "documents"} or len(raw["documents"]) > 20000:
            raise EvaluationError("Expected a bounded configuration/documents input")
        configuration = FrozenAsOfConfiguration.model_validate(raw["configuration"])
        documents = [DocumentRecord.model_validate(item) for item in raw["documents"]]
        _, report = prepare_asof(documents, configuration)
        write_json(args.output, report | {"created_at": datetime.now(UTC).isoformat()})
        return 0
    except (EvaluationError, ValueError, OSError):
        print("As-of preparation failed; no scientific result is claimed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
