"""The browser must present bounded source snapshots without inventing coverage."""

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("streamlit")

from app.ui.api_client import ApiError, parse_result


WEB = Path(__file__).resolve().parents[1] / "app/ui/web.py"


def _arxiv() -> dict:
    return {
        "coverage_state": "partial", "scanned": 2,
        "months": [
            {"month": "2026-08-01", "domain": "cs", "primary_category": "cs.LG",
             "article_count": 1},
            {"month": "2026-09-01", "domain": "cs", "primary_category": "cs.LG",
             "article_count": 1},
        ],
    }


def _funding(count: int = 1) -> dict:
    return {
        "source_id": "nih_reporter", "from_date": "2024-10-01", "to_date": "2026-09-25",
        "total_available": 20, "partial_coverage": True, "coverage_reason": "page_cap",
        "awards": [
            {"application_id": index, "title": "<script>alert(1)</script>" if index == 1 else f"Project {index}",
             "award_notice_date": "2026-09-01", "award_amount_usd": "100000.25",
             "funding_type": "grant_or_cooperative",
             "detail_url": f"https://reporter.nih.gov/project-details/{index}"}
            for index in range(1, count + 1)
        ],
    }


@pytest.mark.parametrize("change", [
    lambda summary: summary["months"].append(deepcopy(summary["months"][0])),
    lambda summary: summary["months"][0].update(month="2026-W01-4"),
    lambda summary: summary["months"][0].update(primary_category="cs.<img>"),
])
def test_arxiv_parser_rejects_duplicate_or_invalid_categories(change) -> None:
    summary = _arxiv()
    change(summary)
    with pytest.raises(ApiError, match="arXiv"):
        parse_result({"signals": [], "arxiv_domains": summary})


@pytest.mark.parametrize("change", [
    lambda summary: summary["awards"][0].update(detail_url="javascript:alert(1)"),
    lambda summary: summary["awards"][0].update(award_amount_usd="1000000000001"),
    lambda summary: summary.update(total_available=0),
    lambda summary: summary.update(from_date="20241001"),
    lambda summary: summary["awards"][0].update(award_notice_date="20260901"),
])
def test_nih_parser_rejects_unsafe_or_inconsistent_values(change) -> None:
    summary = _funding()
    change(summary)
    with pytest.raises(ApiError, match="финансирования"):
        parse_result({"signals": [], "funding_evidence": summary})
