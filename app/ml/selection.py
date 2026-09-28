"""Transparent display gates; never modify topic assignments or score weights."""

ESTABLISHED_SHARE = 0.08


def selection_for(candidate, comparable):
    rows = candidate["metrics"]["years"]
    total = sum(row["direction_documents"] for row in rows)
    share = sum(row["documents"] for row in rows) / total if total else None
    reasons = []
    if candidate.get("direction_guard", {}).get("axis_check") == "off_direction":
        bucket = "excluded_off_direction"
        reasons.append("off_direction")
    elif candidate.get("semantic_relevance", {}).get("requires_review"):
        bucket = "preliminary_signals"
        reasons.append("semantic_scope_requires_review")
    elif share is not None and share > ESTABLISHED_SHARE:
        bucket = "established"
        reasons.append("large_share_in_available_window")
    else:
        if not comparable:
            reasons.append("incomparable_growth_coverage")
        if not candidate["metrics"]["growth_pattern"]:
            reasons.append("sustained_growth_not_established")
        if candidate.get("execution", {}).get("evidence_level") == "nominal":
            reasons.append("nominal_execution_only")
        if candidate.get("direction_guard", {}).get("reason") == "insufficient_document_text":
            reasons.append("insufficient_direction_evidence")
        if not all(candidate["card"].get(field) for field in ("problem", "advantage", "example")):
            reasons.append("incomplete_supported_card")
        if candidate.get("evidence_annotations", {}).get("example", {}).get("modality") == "research_reference":
            reasons.append("example_is_bibliographic_reference")
        bucket = "preliminary_signals" if reasons else "candidates"
    return {"bucket": bucket, "reasons": reasons, "window_document_share": share,
            "established_threshold": ESTABLISHED_SHARE}


def ranking_key(candidate):
    return (candidate.get("execution", {}).get("evidence_level") == "nominal",
            -candidate["metrics"]["score"], -candidate["metrics"]["recent_documents"], candidate["id"])


def partition_candidates(groups, comparable, top_k):
    buckets = {key: [] for key in ("candidates", "preliminary_signals", "established", "excluded_off_direction")}
    seen = set()
    duplicates = []
    for group in sorted(groups, key=ranking_key):
        group["selection"] = selection_for(group, comparable)
        bucket = group["selection"]["bucket"]
        if bucket == "candidates":
            if group["title"] in seen:
                duplicates.append(group["id"])
                continue
            seen.add(group["title"])
        group["status"] = {"candidates": "growth_candidate", "preliminary_signals": "exploratory_candidate",
                           "established": "established", "excluded_off_direction": "off_direction"}[bucket]
        buckets[bucket].append(group)
    before_limit = len(buckets["candidates"])
    below_top = [c["id"] for c in buckets["candidates"][top_k:]]
    buckets["candidates"] = buckets["candidates"][:top_k]
    summary = {key: len(value) for key, value in buckets.items()}
    summary.update(requested_top_k=top_k, eligible_before_limit=before_limit,
                   below_top_ids=below_top, duplicate_label_ids=duplicates,
                   shortfall=max(0, top_k - len(buckets["candidates"])))
    return buckets, summary
