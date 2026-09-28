"""Bounded semantic refinement and size-neutral discovery scheduling.

Branches are hypotheses for subsequent technology/scope review. A split, a small
cluster or a singleton does not establish novelty or a trend lifecycle.
"""
from __future__ import annotations

from collections import deque
import re
from typing import Any


def study_review_priority(record):
    """Schedule primary-work hypotheses before explicit literature syntheses.

    This is a review-cost heuristic, not a research-quality or lifecycle label.
    Reviews and books remain available, and an unmarked article remains unknown.
    No topic-specific terms, years, citation counts or author names participate.
    """
    kinds = (record.document_type, record.raw_metadata.get("type"))
    synthesis = (any(isinstance(kind, str) and kind.casefold() in {
                        "review", "book", "book-chapter", "reference-entry", "edited-book"} for kind in kinds)
                 or re.search(r"\b(?:review|overview|roadmap|survey|обзор\w*)\b", record.title.casefold()))
    return int(bool(synthesis))


def study_hypotheses(leaves, count):
    """Keep a paper-level path even when a density cluster cannot be split.

    Every retained study can propose a mechanism for subsequent strict review.
    Assignment to a broad cluster must never close that path. A one-study leaf
    already represents that hypothesis and must not create a duplicate ID.
    """
    parents = {index: leaf for leaf in leaves for index in leaf["members"]}
    result = []
    for index in range(count):
        parent = parents.get(index)
        if parent and len(parent["members"]) == 1:
            continue
        result.append({"members": [index], "parent_node_id": parent["node_id"] if parent else None,
                       "node_id": None, "depth": parent["depth"] + 1 if parent else 0,
                       "origin": "cluster_member_study" if parent else "unclustered_study"})
    return result


def refine_groups(groups, vectors, *, minimum_size=2, cancel=None):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits

    from app.pilot.encoder import checkpoint

    leaves: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    def descend(members, parent, depth):
        checkpoint(cancel)
        identifier = len(nodes)
        node = {"node_id": identifier, "parent_node_id": parent, "depth": depth, "members": members}
        nodes.append(node)
        # Keep small groups intact; larger, genuinely separated branches may be
        # refined. The threshold bounds work and does not admit a technology.
        if len(members) >= max(24, minimum_size * 4) and depth < 3:
            local = vectors[members]
            centered = local - local.mean(axis=0)
            if float(np.linalg.norm(centered)) > 1e-6:
                with threadpool_limits(limits=1):
                    labels = KMeans(n_clusters=2, random_state=42, n_init=5).fit_predict(centered)
                checkpoint(cancel)
                children = [[member for member, label in zip(members, labels, strict=True) if label == group]
                            for group in (0, 1)]
                if min(map(len, children)) >= minimum_size:
                    # A uniform sample can omit a rare child entirely (e.g.
                    # 4998 + 2), causing silhouette to crash with one label.
                    # Preserve both children in a bounded, reproducible sample.
                    sample_count = min(512, len(members))
                    sides = [np.flatnonzero(labels == label) for label in (0, 1)]
                    first_count = min(len(sides[0]), sample_count - 1,
                                      max(sample_count - len(sides[1]), 1,
                                          int(sample_count * len(sides[0]) / len(members))))
                    rng = np.random.default_rng(42)
                    sample = np.sort(np.concatenate((rng.choice(sides[0], first_count, replace=False),
                        rng.choice(sides[1], sample_count - first_count, replace=False))))
                    checkpoint(cancel)
                    with threadpool_limits(limits=1):
                        quality = float(silhouette_score(centered[sample], labels[sample]))
                    checkpoint(cancel)
                    node["split_silhouette"] = quality
                    if quality >= 0.15:
                        node["split"] = True
                        for child in sorted(children, key=lambda child: tuple(child)):
                            descend(child, identifier, depth + 1)
                        return
        node["split"] = False
        leaves.append({"members": members, "parent_node_id": parent, "node_id": identifier, "depth": depth})

    for members in sorted(groups, key=tuple):
        descend(sorted(members), None, 0)
    return leaves, nodes


# Publication rhetoric that frames a title rather than naming a mechanism. Like
# the generic tokens of admission, these are domain-neutral words, not a topic
# dictionary: «challenges and prospects» reads the same in every field.
_RHETORIC = frozenset(
    "challenge challenges prospect prospects perspective perspectives progress recent advance advances "
    "advancement advancements future next generation status overview trend trends opportunity "
    "opportunities issue issues toward towards current art all".split())
_TITLE_SEGMENT = re.compile(r"[:;,.!?()\[\]{}|/]+")
TITLE_GROUP_MINIMUM = 4
TITLE_GROUP_SHARE = 0.15
TITLE_GROUP_LIMIT = 24


def title_phrase_groups(titles, eligible, *, scope_names, existing=(), minimum_size=TITLE_GROUP_MINIMUM,
                        maximum_share=TITLE_GROUP_SHARE, limit=TITLE_GROUP_LIMIT, cancel=None):
    """Groups of in-scope studies whose titles share one multi-word phrase.

    Embeddings of one field stay too close for a silhouette-approved split:
    measured on «solid-state batteries», density clustering returned leaves of
    181 and 253 of 1036 studies (1313 and 1117 of 4194 at the deep profile)
    with split silhouettes of 0.04–0.09 against the 0.15 threshold. Such a leaf
    is the whole direction: no phrase occurs in every one of its titles, so its
    naming is always refused, and every other candidate was a single paper that
    history can never confirm. The TOP stayed empty or nearly so.

    A group here is the set of studies that ground the requested scope on their
    own (the scope rule of admission) and whose titles contain one shared phrase
    that is neither the direction's own name nor publication rhetoric. That
    phrase anchors every member by construction — the admission rule computed,
    not relaxed — and naming, history and every gate still decide what it is.
    """
    from collections import Counter

    from app.pilot.encoder import checkpoint
    from app.pilot.evidence import (_CLAUSE_WORDS, _GENERIC_TOKENS, _SCOPE_FUNCTION_WORDS, _concept_tokens,
                                    normalize_title)

    def plural(words):
        # The admission matcher's own normalization, so a group and its
        # phrase agree with the rule that later checks them.
        return [word[:-1] if len(word) > 4 and word.endswith("s") else word for word in words]

    scope = [" ".join(plural(normalize_title(name).split())) for name in scope_names if name]
    scope_prefixes = {word[:6] for name in scope for word in name.split()}
    edges = _SCOPE_FUNCTION_WORDS | _GENERIC_TOKENS | _RHETORIC
    counts: Counter[str] = Counter()
    members: dict[str, list[int]] = {}
    surface: dict[str, Counter[str]] = {}
    total = 0
    for index, title in enumerate(titles):
        if not eligible[index]:
            continue
        checkpoint(cancel)
        total += 1
        seen: set[str] = set()
        for segment in _TITLE_SEGMENT.split(title):
            words = normalize_title(segment).split()
            base = plural(words)
            for size in (2, 3, 4):
                for start in range(len(base) - size + 1):
                    chosen = base[start:start + size]
                    phrase = " ".join(chosen)
                    if (phrase in seen or chosen[0] in edges or chosen[-1] in edges
                            or chosen[-1][:6] in scope_prefixes
                            or any(word in _RHETORIC or word in _CLAUSE_WORDS for word in chosen)
                            or any(f" {phrase} " in f" {name} " for name in scope)
                            or len(_concept_tokens(phrase)) < 2):
                        continue
                    seen.add(phrase)
                    surface.setdefault(phrase, Counter())[" ".join(words[start:start + size])] += 1
        for phrase in seen:
            members.setdefault(phrase, []).append(index)
        counts.update(seen)
    ceiling = maximum_share * total
    existing_sets = [set(group) for group in existing]
    selected: list[tuple[str, set[int]]] = []

    def order(item: tuple[str, int]) -> tuple:
        # Among phrases that select the same studies, the one shaped like a
        # name wins the group: no function word inside, led by its own
        # modifier rather than the direction's name, the most words beyond
        # that name, then the fuller phrase.
        words = item[0].split()
        own = sum(word[:6] in scope_prefixes for word in words)
        return (-item[1], any(word in _SCOPE_FUNCTION_WORDS for word in words), words[0][:6] in scope_prefixes,
                -(len(words) - own), -len(words), item[0])

    for phrase, count in sorted(counts.items(), key=order):
        if len(selected) >= limit:
            break
        if count < minimum_size or count > ceiling:
            continue
        group = set(members[phrase])
        if any(len(group & other) / len(group | other) >= 0.6
               for other in [*existing_sets, *(chosen for _, chosen in selected)]):
            continue
        selected.append((surface[phrase].most_common(1)[0][0], group))
    return [{"members": sorted(group), "phrase": phrase} for phrase, group in selected]


def balanced_groups(groups, *, score, key):
    """Round-robin support strata; volume does not decide access to review.

    This only schedules review. It is not the final scientific ranking.
    """
    buckets: list[list[dict[str, Any]]] = [[], [], [], []]
    for group in groups:
        count = len(group["members"])
        bucket = 0 if count == 1 else 1 if count < 8 else 2 if count < 32 else 3
        buckets[bucket].append(group)
    queues = [deque(sorted(bucket, key=lambda group: (-score(group), key(group)))) for bucket in buckets]
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    return result
