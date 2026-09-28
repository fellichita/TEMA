"""Versioned sentence boundaries in unmodified archived source text.

Offsets are Python string (Unicode character) offsets, with an exclusive end.
Only whitespace *between* sentences is left outside spans; no cleaning, entity
decoding, Unicode normalization or joining of distant passages takes place.
Scientific abbreviation rules are conservative heuristics, not linguistic proof.
Legacy evidence extraction must continue to use its original segmentation.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import NamedTuple

SEGMENTATION_VERSION = "raw-sentence-spans/1.0.0"
MAX_TEXT_CHARACTERS = 220_000
_CONTEXT_CHARACTERS = 512
_LOOKAHEAD_CHARACTERS = 256
_ENDINGS = re.compile(r"[.!?…。！？؟۔]+")
_CLOSERS = frozenset("\"'”’»)]}」』】")
_OPENERS = frozenset("\"'“‘«([{「『【")
_STRONG_ENDINGS = frozenset("!?…。！？؟۔")
# Only used inside a bounded 512-character decision window. Whole-source
# markup scanning below is linear even with repeated unterminated comments.
_MARKUP = re.compile(r"<!--.*?-->|<(?=[/!?A-Za-z])(?:[^<>\"']|\"[^\"]*\"|'[^']*')*>", re.S)
_URL = re.compile(r"\b(?:https?://|ftp://|www\.)[^\s<>\"'“”‘’«»]+", re.I)
_CONTINUATION = re.compile(r"\b(?:e\.\s*g|i\.\s*e|vs|cf|dr|prof|mr|mrs|ms|т\.\s*е|т\.\s*к)\.$", re.I)
_REFERENCE = re.compile(r"\b(?:figs?|eqs?|refs?|no|vol|рис|табл|стр|пп?)\.$", re.I)
_REFERENCE_LABEL = re.compile(r"(?:[\[(]?\d|[A-Z]\d|[IVXLCDM]+\b|[A-Z](?=[\s.,;:)\]]))")
_CITATION = re.compile(r"\bet\s+al\.$", re.I)
_CITATION_LABEL = re.compile(r"(?:\(\d{4}[a-z]?\)|\[[\d,\s–-]+\])", re.I)
_INITIALS = re.compile(r"\b((?:[^\W\d_]\.\s*)+)$")
_NEXT_INITIAL = re.compile(r"[^\W\d_]\.")
_QUANTITY_OR_COMPOUND = re.compile(r"(?:\d\s*°?\s*|[/$°~&_-]\s*)$")
_NAME_CONTEXT = re.compile(r"(?:\b(?:by|of|from|with|to)|\b(?:dr|prof|mr|mrs|ms)\.|:)$", re.I)
_SURNAME = re.compile(r"(?:(?:van|von|de|del|da|di|du|der|den)\s+){0,3}([^\W\d_][^\W\d_’'-]*)\b")
_PROPER_NAME = re.compile(r"([^\W\d_][\w’'-]*)\s+(?:(?:of|for|and)\s+(?:the\s+)?)?([^\W\d_][\w’'-]*)")
_CONDITIONAL_ABBREVIATION = re.compile(r"\b(?:et\s+al\.|etc\.|и\s+др\.|(?:[^\W\d_]\.\s*){2,})$", re.I)
_NUMERIC_ABBREVIATION = re.compile(r"\b(?:млн|млрд|тыс)\.$", re.I)
_RUSSIAN_SEE = re.compile(r"\b(?:см|ср)\.$", re.I)


class SentenceSpan(NamedTuple):
    start: int
    end: int


class SentenceSegmentationError(ValueError):
    """An invalid or oversized source cannot be silently truncated."""


def _markup_spans(text: str) -> Iterator[tuple[int, int, bool]]:
    """Scan opaque tags/comments once, including punctuation in quoted attributes.

    Unterminated markup is conservatively opaque through the end of the source.
    It is neither repaired nor removed. Repeated malformed opening tags outside
    quotes restart at the next '<' without rescanning their previous contents.
    """
    cursor = 0
    while cursor < len(text):
        start = text.find("<", cursor)
        if start < 0 or start + 1 == len(text):
            return
        if text.startswith("<!--", start):
            closing = text.find("-->", start + 4)
            end = len(text) if closing < 0 else closing + 3
            yield start, end, False
            cursor = end
            continue
        if text[start + 1] not in "/!?" and not text[start + 1].isalpha():
            cursor = start + 1
            continue
        cursor = start + 1
        quote = ""
        while cursor < len(text):
            char = text[cursor]
            if quote:
                if char == quote:
                    quote = ""
            elif char in "\"'":
                quote = char
            elif char == ">":
                cursor += 1
                break
            elif char == "<":
                break
            cursor += 1
        # A malformed nested opener is opaque only up to the next opener;
        # processing resumes there, rather than scanning that suffix twice.
        yield start, cursor, text.startswith("</", start)


def _protected(text: str) -> tuple[bytearray, dict[int, tuple[int, bool]]]:
    protected = bytearray(len(text))
    tags: dict[int, tuple[int, bool]] = {}
    for start, end, closing in _markup_spans(text):
        protected[start:end] = b"\x01" * (end - start)
        tags[start] = end, closing
    for match in _URL.finditer(text):
        start, end = match.span()
        # Adjacent prose punctuation is outside the URL. A balanced URL path
        # parenthesis stays protected (e.g. /Function_(mathematics)).
        token = match.group()
        balance = {closing: token.count(closing) - token.count(opening)
                   for closing, opening in ((")", "("), ("]", "["), ("}", "{"))}
        while end > start and text[end - 1] in ".!?…。,;:)]}":
            last = text[end - 1]
            if last in ")]}":
                if balance[last] <= 0:
                    break
                balance[last] -= 1
            end -= 1
        protected[start:end] = b"\x02" * (end - start)
    return protected, tags


def _following(text: str, start: int, tags: dict[int, tuple[int, bool]]) -> int:
    """Find the next word without copying the unbounded remaining source."""
    while start < len(text):
        if text[start].isspace():
            start += 1
        elif start in tags:
            start = tags[start][0]
        else:
            break
    return start


def _continues(bare: str, following: str, *, at_sentence_start: bool, closed: bool) -> bool:
    """Resolve a single full stop from bounded context on either side."""
    lead = following.lstrip("".join(_OPENERS))
    starts_sentence = bool(lead) and (lead[0].isupper() or lead[0].isdigit())
    if closed:
        # An abbreviation can remain mid-clause inside parentheses, while a
        # quotation followed by a fresh sentence keeps its closing delimiter.
        return bool(_CONDITIONAL_ABBREVIATION.search(bare)) and not starts_sentence
    if _CONTINUATION.search(bare):
        return True
    see = _RUSSIAN_SEE.search(bare)
    if see and not re.search(r"\d\s*$", bare[:see.start()]):
        return True
    if _REFERENCE.search(bare) and _REFERENCE_LABEL.match(following):
        return True
    if _CITATION.search(bare) and _CITATION_LABEL.match(following):
        return True
    if _NUMERIC_ABBREVIATION.search(bare) and (following[:1].isdigit() or following[:1].islower()):
        return True
    initials = _INITIALS.search(bare)
    if initials and _QUANTITY_OR_COMPOUND.search(bare[:initials.start()]):
        initials = None
    if initials:
        sequence = initials[1]
        if sequence.count(".") == 1 and _NEXT_INITIAL.match(following):
            return True
        if sequence.isupper():
            prefix = bare[:initials.start()].rstrip()
            name_context = ((at_sentence_start and not prefix) or any(char.isspace() for char in sequence)
                            or _NAME_CONTEXT.search(prefix))
            surname = _SURNAME.match(following)
            if name_context and surname and surname[1][0].isupper():
                return True
            proper_name = _PROPER_NAME.match(following)
            if (sequence.count(".") >= 2 and proper_name
                    and proper_name[1][0].isupper() and proper_name[2][0].isupper()):
                return True
    return bool(_CONDITIONAL_ABBREVIATION.search(bare)) and not starts_sentence


def sentence_spans(text: str | None) -> tuple[SentenceSpan, ...]:
    """Return exact non-overlapping sentence spans, excluding boundary whitespace.

    Work and storage are bounded by the 220,000-character source limit. Regexes
    identifying initials/names only inspect fixed-size context windows, never
    successively copy or rescan the whole prefix/suffix of a long abstract.
    None and blank sources yield no spans. Oversized sources raise rather than
    silently turning a truncated passage into an apparently complete sentence.
    """
    if text is None:
        return ()
    if not isinstance(text, str):
        raise SentenceSegmentationError("Для разбиения нужен исходный текст.")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise SentenceSegmentationError("Исходный текст превышает лимит 220 000 символов.")
    protected, tags = _protected(text)
    spans: list[SentenceSpan] = []
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    for mark in _ENDINGS.finditer(text):
        if mark.start() < start or protected[mark.start()]:
            continue
        end = mark.end()
        closed = False
        while end < len(text):
            if text[end] in _CLOSERS:
                closed = True
                end += 1
            elif end in tags and tags[end][1]:
                end = tags[end][0]
            else:
                break
        next_start = end
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        strong = any(char in _STRONG_ENDINGS for char in mark.group())
        # A closing block followed immediately by another tag is a raw XML
        # boundary. Interior dots in decimals, initialisms and URLs are not.
        markup_boundary = end > mark.end() and next_start in tags
        if end < len(text) and next_start == end and not strong and not markup_boundary:
            continue
        lead = _following(text, next_start, tags)
        url_ending = mark.start() > 0 and protected[mark.start() - 1] == 2
        if mark.group() == "." and lead < len(text) and not url_ending:
            context_start = max(start, mark.end() - _CONTEXT_CHARACTERS)
            # Markup is ignored only in this bounded boundary decision; the
            # returned span still contains every original tag and character.
            bare = _MARKUP.sub("", text[context_start:mark.end()])
            following = text[lead:lead + _LOOKAHEAD_CHARACTERS]
            if _continues(bare, following, at_sentence_start=context_start == start, closed=closed):
                continue
        if end > start:
            spans.append(SentenceSpan(start, end))
        start = next_start
    end = len(text)
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append(SentenceSpan(start, end))
    return tuple(spans)


def sentences(text: str | None) -> tuple[str, ...]:
    """Convenience view of the same exact raw spans; short sentences are kept."""
    spans = sentence_spans(text)
    return tuple(text[start:end] for start, end in spans) if text is not None else ()
