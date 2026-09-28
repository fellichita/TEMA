"""The local AI client must reach eligible candidates and leave decoys unassessed."""

import json

from app.pilot.archive import DocumentArchive
from app.pilot.evidence import label_candidates
from app.pilot.local_client import LocalLlmClient, LocalProviderConfig
from app.pilot.local_llm import LocalCompletion
from app.runtime.budget import BudgetLimits, BudgetService
from app.sqlite_runtime import sqlite3
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot


class AnsweringModel:
    spec = {"max_new_tokens": 2048}

    def __init__(self):
        self.calls = []

    def generate(self, *, system, user, max_new_tokens, cancel, stop_at_json, answer_prefix):
        assert stop_at_json and answer_prefix == '{"candidates":'
        assert not cancel.is_set() and max_new_tokens <= 1024
        material = json.loads(user.split("\n", 1)[0])
        self.calls.append(material)
        assert len(material["clusters"]) == 1
        cluster = material["clusters"][0]
        answer = {"candidates": [{
            "candidate_id": cluster["candidate_id"],
            "label": "Литий-селективные мембраны",
            "definition": "Мембраны для избирательного извлечения лития из рассолов.",
            "phrase_number": 1,
            "specificity": "specific_technology",
            "in_scope": True,
            "scope_support": [
                {"revision_id": source["revision_id"], "field": "title", "quote": source["title"]}
                for source in cluster["documents"]
            ],
            "scope_reason": "Каждая работа исследует этот механизм извлечения лития.",
        }]}
        return LocalCompletion(json.dumps(answer, ensure_ascii=False), 200, 100, False)


def test_local_model_names_only_source_grounded_candidate_and_records_receipt(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    papers = (document(1), document(2), document(3,
        title="Optical fiber communication networks",
        abstract="Photonic interconnects carry data between communication nodes."))
    found = snapshot(papers, archive)
    supported = candidate(found, candidate_id="supported",
        discovery_study_ids=tuple(sorted(ref.study_id for ref in found.documents[:2])))
    off_topic = candidate(found, candidate_id="off-topic",
        discovery_study_ids=(found.documents[2].study_id,))
    connection = sqlite3.connect(tmp_path / "ledger.sqlite3", isolation_level=None)
    try:
        budget = BudgetService(connection)
        budget.create_scope("run", BudgetLimits(5, 100_000, 10_000, 0), currency="USD")
        model = AnsweringModel()
        client = LocalLlmClient(LocalProviderConfig(), budget, model=model)
        context = Context()

        named = label_candidates((off_topic, supported), found, archive, context,
                                 client=client, scope_ids=("run",), query_plan=query_plan())

        by_id = {item.candidate_id: item for item in named}
        assert by_id["supported"].specificity == "specific_technology"
        assert by_id["off-topic"].specificity == "uncertain"
        assert len(model.calls) == 1
        assert [item["candidate_id"] for item in model.calls[0]["clusters"]] == ["c0"]
        rejected = context.checkpoints["labels_0"]
        admitted = context.checkpoints["labels_1"]
        assert rejected["label_status"] == "unverified_lexical_label"
        assert rejected["failure_code"] == "evidence_rejected"
        assert rejected["receipt"] is None and rejected["fallback_candidate_ids"] == ["off-topic"]
        assert admitted["label_status"] == "model_proposal"
        assert admitted["failure_code"] is None and admitted["fallback_candidate_ids"] == []
        assert admitted["receipt"]["provider"] == "local"
        assert label_candidates((off_topic, supported), found, archive, context,
                                client=client, scope_ids=("run",), query_plan=query_plan()) == named
        assert len(model.calls) == 1
        assert budget.snapshot("run").used.calls == 1
    finally:
        connection.close()
