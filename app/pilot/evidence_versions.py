"""Explicit extraction policies for immutable scientific result versions.

Empty evidence collections also require a policy: inferring it from their first
item would let a removed observation silently switch the replay algorithm.
"""

from typing import Literal

from app.pilot.contracts import MethodologyVersion

NoveltyMethod = Literal["archived-author-novelty/1.0.0", "archived-author-novelty/2.0.0",
                        "archived-author-novelty/3.0.0"]
PrimaryMethod = Literal["archived-primary-result/1.0.0", "archived-primary-result/2.0.0"]
PublicationRules = Literal["publication-status/1.0.0", "publication-status/2.0.0"]


def novelty_method(version: MethodologyVersion) -> NoveltyMethod:
    if version == "3.4.0":
        return "archived-author-novelty/3.0.0"
    if version == "3.3.0":
        return "archived-author-novelty/2.0.0"
    return "archived-author-novelty/1.0.0"


def primary_method(version: MethodologyVersion) -> PrimaryMethod:
    return "archived-primary-result/2.0.0" if version == "3.4.0" else "archived-primary-result/1.0.0"


def publication_rules(version: MethodologyVersion) -> PublicationRules:
    return "publication-status/2.0.0" if version == "3.4.0" else "publication-status/1.0.0"
