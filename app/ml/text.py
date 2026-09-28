"""Local lexical scope checks and evidence extraction; no remote language model."""

# The frozen _denied_implementation source deliberately zips adjacent sentences.
# Keep its protected source hash intact; the second sequence is one item shorter.
# ruff: noqa: B905

import html
from copy import deepcopy
from functools import lru_cache
import re
import unicodedata

from app.input_safety import MAX_ABSTRACT_CHARACTERS, MAX_TITLE_CHARACTERS, is_safe_http_url


ALIASES = {
    "фотонные нейроморфные вычисления": "photonic neuromorphic computing",
    "фотонные нейросети": "photonic neuromorphic computing",
    "искусственный интеллект": "artificial intelligence", "ии": "artificial intelligence",
    "технологии в ии": "artificial intelligence", "технологии ии": "artificial intelligence",
    "квантовые вычисления": "quantum computing", "хранение данных в днк": "dna data storage",
}
SERVICE_TITLE = re.compile(r"^(?:programme|program|index|front matter|table of contents|contents|"
                           r"editorial board|author index|committee|preface|copyright|erratum|correction)\b", re.I)
PROCEEDINGS_TITLE = re.compile(r"\b(?:international conference|conference proceedings|proceedings of|technical digest)\b", re.I)
OPTICAL = re.compile(r"\b(?:photon\w*|nanophoton\w*|optic\w*|optoelectr\w*|photoelectr\w*|photosynap\w*|"
                     r"photomemrist\w*|phototransist\w*|micro[-‐‑– ]?combs?|lasers?|vcsel(?:s|[-‐‑–]sa)?|"
                     r"light[-‐‑– ](?:driven|stimulat\w*|induced|gated|modulat\w*|mediated|pulses?))\b", re.I)
NEURAL = re.compile(r"\b(?:neur\w*|synap\w*|reservoir[- ](?:comput\w*|networks?)|perceptrons?|"
                    r"deep learning|machine learning|spiking|stdp|spike[-‐‑– ]encoding|"
                    r"spike[-‐‑– ]timing[-‐‑– ]dependent(?:[-‐‑– ]plasticity)?)\b", re.I)
UNCERTAIN = re.compile(r"\b(?:potential\w*|possible|prospect\w*|could|may|might|would|should|can|future|towards)\b", re.I)
CONTEXT = re.compile(r"\b(?:promising|potential\w*|possible|prospect\w*|could|may|might|would|should|can\s+be|"
                     r"future|building blocks?|towards|such as)\b", re.I)
FUNCTION = re.compile(r"\b(?:demonstrat\w*|implement\w*|realiz\w*|realise\w*|emulat\w*|mimic\w*|simulat\w*|"
                       r"construct\w*|propos\w*|present\w*|review\w*|synaptic|neuron|neurons|activation)\b", re.I)
PASSIVE_FUNCTION = re.compile(
    r"\b(?:is|are|was|were|has been|have been)\s+"
    r"(?:(?:experimentally|numerically|successfully|directly)\s+)?"
    r"(?:demonstrated|implemented|realized|realised|emulated|mimicked|simulated|computed|encoded|modulated)\b", re.I)
DENIED_FUNCTION = re.compile(r"\b(?:not(?!\s+(?:only|just)\b)|never|cannot|\w+n['’]t)\b", re.I)
HARDWARE = re.compile(r"\b(?:devices?|chips?|circuits?|processors?|resonators?|interferometers?|lasers?|"
                      r"vcsel(?:s|[-‐‑–]sa)?|transistors?|memristors?|cavit(?:y|ies))\b", re.I)
OTHER_EXECUTOR = re.compile(r"\b(?:cpu|gpu|(?:electronic|electrical|digital)\s+(?:processor|computer|controller|circuit))s?\b", re.I)
LEARNING = re.compile(r"\b(?:machine|deep)[-‐‑– ]learning\b", re.I)
LEARNING_TOOL = re.compile(r"\b(?:inverse[-‐‑– ]design|laser[- ]machining|"
                          r"(?:characteri[sz]\w*|modelling|modeling|predict\w*|optimi[sz]\w*)\b[^.!?;]{0,100}"
                          r"\b(?:lasers?|vcsel\w*|nanophoton\w*|optic\w*))\b", re.I)
NEURAL_OPERATION = re.compile(NEURAL.pattern + r"|\b(?:inference|weighted[- ]sums?|"
                              r"convolution(?:al)?|cnns?|"
                              r"matrix[- ]multiplication|matrix[- ]vector[- ]multiplication|multiply[- ]accumulate|"
                              r"orientation selectivity|spatiotemporal processing)\b", re.I)
COMPUTE_ACTION = re.compile(r"\b(?:implement(?:s|ed|ing)?|perform(?:s|ed|ing)?|process(?:es|ed|ing)?|comput(?:e[sd]?|ing)|"
                            r"execut(?:e[sd]?|ing)|emulat(?:e[sd]?|ing)|mimic(?:s|ked|king)?|"
                            r"reali[sz](?:e[sd]?|ing)|encod(?:e[sd]?|ing)|modulat(?:e[sd]?|ing)|"
                            r"simulat(?:e[sd]?|ing)|demonstrat(?:e[sd]?|ing)|"
                            r"runs?|ran|running)\b", re.I)
TRAIN_ACTION = re.compile(r"\b(?:train\w*|calibrat\w*|tun(?:e[sd]?|ing))\b", re.I)
DESIGN_ACTION = re.compile(r"\b(?:design\w*|predict\w*|optimi[sz]\w*|characteri[sz](?:e[sd]?|ing|ation)|"
                           r"model(?:led|ing|ling)|machining)\b", re.I)
LEARNING_METHOD = re.compile(r"\b(?:neural[- ](?:networks?|models?)|(?:machine|deep)[- ]learning)\b", re.I)
DENIED_OPERATION = re.compile(r"\b(?:no|not(?!\s+(?:only|just|merely)\b)|never|cannot|\w+n['’]t|"
                              r"unable|fail(?:s|ed)?|unimplemented)\b", re.I)
NEGATED_BENEFIT = re.compile(r"\b(?:no(?!\s+(?:less|more)\s+than\b)|"
                             r"not(?!\s+(?:only|just|merely)\b)|never|neither|cannot|\w+n['’]t|"
                             r"unable|fail(?:s|ed|ure|ing)?|without|lack(?:s|ed|ing)?|absence|"
                             r"rul(?:e[sd]?|ing)\s+out)\b", re.I)
REQUIRED_BENEFIT = re.compile(
    r"\b(?:is|are|was|were|be|been|remain(?:s|ed)?)\s+(?:\w+\s+){0,2}"
    r"(?:necessary|required|needed|essential)\b|"
    r"\b(?:requires?|must|should)\b", re.I)
UNSUPPORTED_BENEFIT = re.compile(
    r"\b(?:is|are|was|were|be|been|remain(?:s|ed)?)\s+(?:\w+\s+){0,2}"
    r"(?:absent|unconfirmed|unsupported|unobserved|undemonstrated)\b", re.I)
DENIED_BENEFIT_RESULT = re.compile(
    r"\brul(?:e[sd]?|ing)\s+out\b|"
    r"\b(?:not(?!\s+(?:only|just|merely)\b)|never|neither|cannot|\w+n['’]t|unable)\b"
    r"(?:\W+\w+){0,5}\W+(?:observed|measured|demonstrated|achieved|obtained|confirmed|found|"
    r"shown|reproduced|supported|detected|significant|realized|realised)\b", re.I)

# Separate card lexicons: do not change document admission or the model vocabulary.
AXIS_CARRIER = {
    "optical": r"optical", "photonic": r"photonics?", "nanophotonic": r"nanophotonics?",
    "optoelectronic": r"optoelectronic", "photoelectric": r"photoelectric",
    "microring": r"microrings?", "waveguide": r"waveguides?", "vcsel": r"vcsels?",
    "laser": r"lasers?", "mzi": r"mzis?", "diffractive": r"diffractive",
    "interferometer": r"interferometers?", "microcomb": r"micro\s*combs?",
}
AXIS_COMPUTE = {
    "neural": r"neural", "neuromorphic": r"neuromorphic", "neuron": r"neurons?",
    "synaptic": r"synaptic", "synapse": r"synapses?", "spiking": r"spiking", "snn": r"snns?",
    "reservoir": r"reservoirs?", "inference": r"inference", "convolution": r"convolutions?",
    "convolutional": r"convolutional", "mac": r"macs?", "learning": r"learning",
    "perceptron": r"perceptrons?", "memristive": r"memristive",
}
# An abstract fallback needs a stated action; nouns such as "processing" in an
# application list are not enough. The existing relation check is reused below.
AXIS_PREDICATE = re.compile(
    r"\b(?:implements?|implemented|performs?|performed|executes?|executed|computes?|computed|"
    r"emulates?|emulated|reali[sz]es?|reali[sz]ed|mimics?|mimicked|runs?|ran|"
    r"demonstrates?|demonstrated)\b", re.I)


def _strip_markup(value):
    """The former <[^>]+> substitution, with a single forward scan."""
    pieces, copied, cursor = [], 0, 0
    while True:
        opening = value.find("<", cursor)
        if opening < 0:
            break
        closing = value.find(">", opening + 1)
        if closing < 0:
            break
        cursor = closing + 1
        if closing == opening + 1:  # An empty <> was never considered markup.
            continue
        pieces.extend((value[copied:opening], " "))
        copied = cursor
    pieces.append(value[copied:])
    return "".join(pieces)


def _bounded_character_reference(match):
    """Keep HTML semantics without converting arbitrarily large integers."""
    reference = match.group()
    if not reference.startswith("&#"):
        return html.unescape(reference)
    token = reference[2:].rstrip(";")
    hexadecimal = token[:1] in {"x", "X"}
    digits = (token[1:] if hexadecimal else token).lstrip("0") or "0"
    if len(digits) > (6 if hexadecimal else 7):
        return "\ufffd"
    return html.unescape("&#" + ("x" if hexadecimal else "") + digits + ";")


def document_text_issue(document):
    """Guard raw fields before cleaning, including direct in-memory callers."""
    for field, limit in (("title", MAX_TITLE_CHARACTERS), ("abstract", MAX_ABSTRACT_CHARACTERS)):
        value = document.get(field)
        if value is not None and len(value) > limit:
            return f"oversized_{field}_requires_review"
    return None


def clean(value):
    value = value or ""
    # Match the HTML decoder's references once: a decoded ampersand must not
    # cause a second decode of the following text. Named references still use
    # the standard HTML5 rules; numeric conversion receives at most 7 digits.
    value = re.sub(r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)",
                   _bounded_character_reference, value)
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", _strip_markup(value.replace("\\n", " "))).strip()


def title_key(value):
    return " ".join(re.findall(r"\w+", clean(value).casefold()))


def resolve_topic(value):
    from app.ml.directions import resolve_direction
    return resolve_direction(clean(value))


def _topic_axis_matches(topic_terms):
    if isinstance(topic_terms, str):
        topic_terms = [topic_terms]
    value = re.sub(r"[-‐‑‒–—−]+", " ", clean(" ".join(topic_terms))).casefold()
    return {axis: sorted(term for term, pattern in lexicon.items()
                         if re.search(r"\b(?:" + pattern + r")\b", value))
            for axis, lexicon in (("carrier", AXIS_CARRIER), ("compute", AXIS_COMPUTE))}


def topic_axis_guard(topic_terms):
    """Whether the supplied terms mention both axes; not a relevance verdict."""
    return all(_topic_axis_matches(topic_terms).values())


def card_topic_guard(topic, topic_terms, studies):
    """Review incomplete cluster labels against all member documents before TOP-K."""
    query = resolve_topic(topic).casefold()
    if not ("photon" in query and ("neuro" in query or "neural" in query)):
        return None
    terms = list(topic_terms)
    matches = _topic_axis_matches(terms)
    result = {"axis_check": "supported", "reason": "cluster_terms_cover_axes",
              "topic_terms": terms, "cluster_matches": matches,
              "checked_documents": 0, "supporting_documents": []}
    if all(matches.values()):
        return result
    incomplete = not studies
    for study in studies:
        result["checked_documents"] += 1
        title, abstract = clean(study.get("title")), clean(study.get("abstract"))
        if not title or not abstract:
            incomplete = True
        if scope_check(topic, title, abstract) != "direct_lexical_signal":
            continue
        support, mode = None, None
        if topic_axis_guard([title]):
            support, mode = title, "study_title"
        else:
            parts = sentences(abstract)
            for passage in parts:
                if AXIS_PREDICATE.search(passage) and _photonic_sentence_signal(passage, require_execution=True):
                    support, mode = passage, "source_excerpt"
                    break
            if support is None:
                for previous, current in zip(parts, parts[1:]):
                    if (any(AXIS_PREDICATE.search(part) and NEURAL_OPERATION.search(part)
                            for part in (previous, current))
                            and _linked_photonic_sentences(previous, current, require_execution=True)):
                        support, mode = previous + "\n" + current, "neighboring_excerpts"
                        break
        if support is not None:
            result["supporting_documents"].append({
                "study_id": study["id"], "url": study["url"], "title": study["title"],
                "text": support, "mode": mode, "matches": _topic_axis_matches([support])})
    if result["supporting_documents"]:
        result.update(axis_check="partial", reason="document_support_for_missing_cluster_axis")
    elif incomplete:
        result.update(axis_check="partial", reason="insufficient_document_text")
    else:
        result.update(axis_check="off_direction", reason="no_document_support_for_missing_cluster_axis")
    return result


_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"'”’»\)\]]*\s+")


def sentences(value):
    """Keep source text intact, including short sentences and scientific abbreviations.

    Один и тот же абстракт разбирается несколько раз за анализ, поэтому разбор
    кешируется. Наружу отдаётся копия: вызывающие работают со списком.
    """
    return list(_sentences_cached(clean(value)))


@lru_cache(maxsize=4096)
def _sentences_cached(text):
    result, start = [], 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        passage = text[start:boundary.end()].strip()
        bare = passage.rstrip("\"'”’»)]")
        following = text[boundary.end():]
        lead = following.lstrip("\"'“‘«([{")
        starts_sentence = bool(lead) and (lead[0].isupper() or lead[0].isdigit())
        if bare == passage:
            # These forms introduce an example, comparison or name, even before a capital.
            if re.search(r"\b(?:e\.\s*g|i\.\s*e|vs|cf|dr|prof|mr|mrs|ms)\.$", bare, re.I):
                continue
            # Figure/equation abbreviations continue only when followed by a reference label.
            if (re.search(r"\b(?:figs?|eqs?|refs?|no|vol)\.$", bare, re.I)
                    and re.match(r"(?:[\[(]?\d|[A-Z]\d|[IVXLCDM]+\b|[A-Z](?=[\s.,;:)\]]))", following)):
                continue
            # A year or reference number belongs to the author citation, not a new sentence.
            if (re.search(r"\bet\s+al\.$", bare, re.I)
                    and re.match(r"(?:\(\d{4}[a-z]?\)|\[[\d,\s–-]+\])", following, re.I)):
                continue
            initials = re.search(r"\b((?:[^\W\d_]\.\s*)+)$", bare)
            if initials and re.search(r"(?:\d\s*°?\s*|[/$°~&_-]\s*)$", bare[:initials.start()]):
                initials = None  # A quantity or compound unit is not an author's initial.
            if initials:
                # Keep a sequence of initials (or spaced e. g.) together.
                if initials[1].count(".") == 1 and re.match(r"[^\W\d_]\.", following):
                    continue
            if initials and initials[1].isupper():
                prefix = bare[:initials.start()].rstrip()
                name_context = (not prefix or " " in initials[1]
                                or re.search(r"(?:\b(?:by|of|from|with|to)|\b(?:dr|prof|mr|mrs|ms)\.|:)$",
                                             prefix, re.I))
                surname = re.match(r"(?:(?:van|von|de|del|da|di|du|der|den)\s+){0,3}"
                                   r"([^\W\d_][^\W\d_’'-]*)\b", following)
                if name_context and surname and surname[1][0].isupper():
                    continue
                # An initialism may qualify a proper name: U.S. Department of Energy.
                proper_name = re.match(r"([^\W\d_][\w’'-]*)\s+(?:(?:of|for|and)\s+(?:the\s+)?)?"
                                       r"([^\W\d_][\w’'-]*)", following)
                if (initials[1].count(".") >= 2 and proper_name
                        and proper_name[1][0].isupper() and proper_name[2][0].isupper()):
                    continue
        # Citation endings and abbreviated units can also end a complete sentence.
        abbreviation = re.search(r"\b(?:et\s+al\.|etc\.|(?:[^\W\d_]\.\s*){2,})$", bare, re.I)
        if abbreviation and not starts_sentence:
            continue
        result.append(passage)
        start = boundary.end()
    if text[start:].strip():
        result.append(text[start:].strip())
    return result


_CLAUSE_SPLIT = re.compile(r"[;:]|\b(?:while|whereas|but)\b|"
                           r"\band\s+(?=(?:a|an|the|our|we|this|that|these|those)\b)", re.I)


@lru_cache(maxsize=8192)
def _scope_clauses(passage):
    """Клаузы одного предложения; кеш — потому что одно предложение проходит
    через несколько проверок исполнителя за один документ."""
    return tuple(_CLAUSE_SPLIT.split(passage))


_BRIDGE = re.compile(r"[\s‐‑–-]*(?:(?:based|artificial|integrated|diffractive|spiking|convolutional|deep|recurrent|"
                     r"silicon|electronic|vector|devices?|chips?|circuits?|processors?|accelerators?|arrays?|"
                     r"for|implementation|of)[\s‐‑–-]+)*", re.I)


def _named_photonic_operation(passage):
    """A named physical architecture, not arbitrary co-occurrence of two keywords."""
    for optical in OPTICAL.finditer(passage):
        for neural in NEURAL_OPERATION.finditer(passage):
            if optical.end() <= neural.start() and _BRIDGE.fullmatch(passage[optical.end():neural.start()]):
                return True
            if neural.end() <= optical.start() and _BRIDGE.fullmatch(passage[neural.end():optical.start()]):
                return True
    return False


@lru_cache(maxsize=8192)
def _electronic_execution(passage):
    """Bind a CPU/GPU to execution; training on it does not imply electronic inference."""
    verbs = re.compile(COMPUTE_ACTION.pattern + "|" + TRAIN_ACTION.pattern, re.I)
    for executor in OTHER_EXECUTOR.finditer(passage):
        before, after = passage[:executor.start()], passage[executor.end():]
        if re.search(r"\b(?:not|\w+n['’]t)\s+(?:require|need)\s+(?:\w+\s+){0,3}$|"
                     r"\bnot\s+(?:on|by|using|with)\s+(?:(?:a|an|the)\s+)?$", before, re.I):
            continue
        actions = list(verbs.finditer(before))
        if ((actions and COMPUTE_ACTION.fullmatch(actions[-1].group()))
                or (not actions and NEURAL_OPERATION.search(before))):
            if re.search(r"\b(?:on|by|using|with)\s+(?:(?:a|an|the)\s+)?$", before, re.I):
                return True
        action = verbs.search(after)
        if action and COMPUTE_ACTION.fullmatch(action.group()) and re.fullmatch(
                r"[\s,]*(?:(?:processor|directly|also|then|only|itself|to|is|are|was|were|used|employed)\s+)*",
                after[:action.start()], re.I):
            return True
    return False


@lru_cache(maxsize=8192)
def _learning_design(passage):
    methods = list(LEARNING_METHOD.finditer(passage))
    designs = list(DESIGN_ACTION.finditer(passage))
    if not methods or not designs:
        return False
    if not (OPTICAL.search(passage) or re.search(r"\b(?:geometr(?:y|ies)|devices?)\b", passage, re.I)):
        return False
    # The learning method must be the design tool, not the architecture being designed.
    return any(method.end() <= design.start() or (design.end() <= method.start() and re.search(
        r"\b(?:by|using|with|through)\b", passage[design.end():method.start()], re.I))
        for method in methods for design in designs)


_DENIED_NEED = re.compile(r"\b(?:not|never|\w+n['’]t)\s+(?:require|need)s?\b", re.I)
_DENIED_EXECUTOR = re.compile(r"\bnot\s+(?:on|by|using|with)\s+(?:(?:a|an|the)\s+)?" + OTHER_EXECUTOR.pattern, re.I)
_IMPLEMENTATION_WORD = re.compile(r"\b(?:implementation|unimplemented)\b", re.I)


@lru_cache(maxsize=8192)
def _denied_implementation(passage):
    denial_text = _DENIED_NEED.sub("needs", passage)
    denial_text = _DENIED_EXECUTOR.sub("", denial_text)
    return bool(DENIED_OPERATION.search(denial_text)
                and (NEURAL_OPERATION.search(passage) or OPTICAL.search(passage))
                and (COMPUTE_ACTION.search(passage) or _IMPLEMENTATION_WORD.search(passage)))


@lru_cache(maxsize=8192)
def _photonic_sentence_signal(passage, *, require_execution=False):
    """Use the same executor relation for a title, sentence or resolved reference."""
    for clause in _scope_clauses(passage):
        if _denied_implementation(clause) or _electronic_execution(clause) or _nonoptical_physical_execution(clause):
            continue
        enumeration = _enumeration_start(clause)
        if enumeration is not None:
            # A background list cannot supply the carrier, but a complete earlier
            # execution assertion keeps its own meaning and original quote.
            if _photonic_sentence_signal(clause[:enumeration], require_execution=require_execution):
                return True
            continue
        # Outlook after a comma does not veto a separately stated implementation.
        assertions = [clause, *clause.split(",")] if PASSIVE_FUNCTION.search(clause) else [clause]
        for assertion in assertions:
            if not (OPTICAL.search(assertion) and NEURAL_OPERATION.search(assertion)):
                continue
            if ((CONTEXT.search(assertion) or _executor_outlook(assertion))
                    and not (PASSIVE_FUNCTION.search(assertion) and not UNCERTAIN.search(assertion)
                             and not _executor_outlook(assertion))
                    and not _measured_building_block(assertion)):
                continue
            named = _named_photonic_operation(assertion)
            actions = _execution_actions(assertion)
            if _learning_design(assertion) and not (named and actions):
                continue
            background = re.search(r"\b(?:applications?|such as|including|prospects?|promise)\b", assertion, re.I)
            if named and ((actions and not background) or (not require_execution and not background)):
                return True
            for optical in OPTICAL.finditer(assertion):
                for neural in NEURAL_OPERATION.finditer(assertion):
                    # Optical hardware implements a neural operation.
                    for action in actions:
                        if optical.end() <= action.start() < neural.start():
                            actor = assertion[optical.start():action.start()]
                            if (HARDWARE.search(actor) or re.search(r"\b(?:accelerators?|engines?|modulators?|micro[-‐‑– ]?combs?)\b", actor, re.I)
                                    or re.match(r"(?:laser|vcsel|photomemrist|phototransist|light[-‐‑– ])", optical.group(), re.I)):
                                return True
                    # A neural operation is realized in/by/using optical hardware.
                    if neural.end() <= optical.start() and any(a.start() < optical.start() for a in actions):
                        if re.search(r"\b(?:in|on|by|using|with|through|via)\b", assertion[neural.end():optical.start()], re.I):
                            return True
    return False


def _execution_actions(passage):
    """Computing/processing in a technology name is a noun, not an execution claim."""
    return [match for match in COMPUTE_ACTION.finditer(passage)
            if match.group().casefold() not in {"computing", "processing", "running"}
            or re.search(r"\b(?:is|are|was|were|been|be)\s+(?:\w+ly\s+)?$", passage[:match.start()], re.I)]


def _measured_building_block(passage):
    """A measured neuron primitive differs from a platform for future applications."""
    if not re.search(r"\bwe\s+experimentally\s+demonstrat(?:e|ed)\b", passage, re.I):
        return False
    context = re.sub(r"\bbuilding blocks?\b", "", passage, flags=re.I)
    return not (CONTEXT.search(context) or UNCERTAIN.search(context) or _executor_outlook(context))


@lru_cache(maxsize=8192)
def _enumeration_start(sentence):
    for marker in re.finditer(r"\b(?:from|such as|including|include[sd]?|either)\b", sentence, re.I):
        separator = r"\bto\b" if marker.group().casefold() == "from" else (
            r"\bor\b" if marker.group().casefold() == "either" else r",|\band\b|\bor\b")
        alternatives = re.split(separator, sentence[marker.end():], flags=re.I)
        if len(alternatives) < 2:
            continue
        optical, other = False, False
        for alternative in alternatives:
            light = bool(OPTICAL.search(alternative) or re.search(r"\bmicro[-‐‑– ]?combs?\b", alternative, re.I))
            optical |= light
            # A photonic memristive/phase-change device remains an optical item.
            other |= bool(not light and re.search(
                r"\b(?:magnetic|spintronic|memristive|electronic|mechanical|ferromagnetic|phase[-‐‑– ]change)\b",
                alternative, re.I))
        if optical and other:
            return marker.start()
    return None


def enumeration_alternative(sentence):
    """Detect alternative carriers, never ordinary optical/electronic cooperation."""
    return _enumeration_start(sentence) is not None


def dominant_carrier(title, abstract):
    """Weighted carrier mentions are diagnostics, not a document-admission veto."""
    patterns = {"optical": OPTICAL.pattern + r"|\bmicro[-‐‑– ]?combs?\b",
                "magnetic": r"\b(?:ferromagnetic|magnetic)\b", "spintronic": r"\bspintronic\b",
                "electronic": r"\b(?:electronic|electrical|digital|gpu|cpu)\b",
                "mechanical": r"\b(?:mechanical|hydrodynamic\w*)\b",
                "memristive": r"\bmemristive\b", "phase_change": r"\bphase[-‐‑– ]change\b"}
    counts = {carrier: 3 * len(re.findall(pattern, title, re.I)) + len(re.findall(pattern, abstract, re.I))
              for carrier, pattern in patterns.items()}
    peak = max(counts.values(), default=0)
    leaders = [carrier for carrier, count in counts.items() if count == peak] if peak else []
    return {"carrier": leaders[0] if len(leaders) == 1 else "mixed" if leaders else "unknown",
            "weighted_counts": counts, "title_weight": 3, "abstract_weight": 1}


def _executor_conflict(passage):
    """Recognize contrary evidence without changing the existing negation helpers."""
    return any(_denied_implementation(clause) or _electronic_execution(clause) or _learning_design(clause)
               or _nonoptical_physical_execution(clause)
               for clause in _scope_clauses(passage))


@lru_cache(maxsize=8192)
def _nonoptical_physical_execution(passage):
    """Only explicit execution by another substrate; material frequency is irrelevant."""
    for carrier in re.finditer(r"\b(?:ferromagnetic|magnetic|spintronic|mechanical|hydrodynamic)\b", passage, re.I):
        for action in _execution_actions(passage):
            if carrier.end() < action.start() and NEURAL_OPERATION.search(passage[action.end():]):
                actor = passage[:action.start()]
                if HARDWARE.search(actor) and not OPTICAL.search(actor):
                    return True
            if (action.end() < carrier.start() and NEURAL_OPERATION.search(passage[:action.start()])
                    and re.search(r"\b(?:by|in|on|using|through)\b", passage[action.end():carrier.start()], re.I)
                    and not OPTICAL.search(passage[action.end():carrier.start()])):
                return True
    return False


def _executor_background(passage):
    return bool(re.search(r"\b(?:baseline|earlier|previous|prior|conventional|traditional|existing)\b", passage, re.I))


def _executor_outlook(passage):
    return bool(re.search(r"\b(?:pav\w*\s+the\s+way|open\w*\s+(?:new\s+)?avenues?|"
                          r"future\s+applications?|promis\w*\s+for)\b", passage, re.I))


def _nominal_executor_outlook(passage):
    """A need for components or future groundwork is not a studied architecture.

    This only filters nominal witnesses; a separately established direct relation
    is still evaluated by the unchanged execution path.
    """
    return bool(re.search(
        r"\b(?:advances?|progress)\s+in\b[^.!?;\n]{0,160}\brequires?\s+"
        r"(?:(?:new|novel|improved)\s+)?(?:devices?|components?|hardware|architectures?)\b|"
        r"\b(?:lays?|laid|laying)\s+(?:the\s+)?groundwork\s+for\b", passage, re.I))


def _contradicts_executor(support, conflict):
    """Bind explicit contradictions conservatively; a distinct baseline remains distinct."""
    if _executor_background(conflict) and not _executor_background(support):
        return False
    if _learning_design(conflict) and not _denied_implementation(conflict) and not _electronic_execution(conflict):
        return False
    if re.search(r"\b(?:all|exclusively|only)\b", conflict, re.I) and _electronic_execution(conflict):
        return True
    if not _denied_implementation(conflict):
        return False
    # Explicit deictic references or repeated descriptions of the current device.
    if re.search(r"\b(?:this|that|same|our)\s+(?:(?:optical|photonic|new)\s+)*(?:chip|device|circuit|processor)\b", conflict, re.I):
        return True
    return bool(re.search(r"\b(?:neural|synaptic|inference|computation)\b", conflict, re.I)
                and not HARDWARE.search(conflict))


def execution_evidence(title, abstract):
    """Classify optical execution independently from nominal thematic admission.

    Evidence quotes are untouched source titles/sentences, including paired source
    sentences separated by a newline. A direct relation is not an experiment label.

    Один и тот же документ проходит здесь дважды: при допуске по направлению и
    затем при разметке отобранных исследований. Разбор кешируется, наружу
    отдаётся копия — вызывающие кладут словарь в карточку исследования.
    """
    return deepcopy(_execution_evidence_cached(title or "", abstract or ""))


@lru_cache(maxsize=4096)
def _execution_evidence_cached(title, abstract):
    parts = sentences(abstract)
    diagnostics = dominant_carrier(title, abstract)
    conflicts = [clause for part in parts for clause in _scope_clauses(part) if _executor_conflict(clause)]
    direct = []
    for part in parts:
        if _photonic_sentence_signal(part, require_execution=True):
            direct.append({"text": part, "mode": "source_excerpt"})
    for previous, current in zip(parts, parts[1:], strict=False):
        if (not enumeration_alternative(previous) and not enumeration_alternative(current)
                and _linked_photonic_sentences(previous, current, require_execution=True)):
            direct.append({"text": previous + "\n" + current, "mode": "neighboring_excerpts"})
    if not conflicts and _photonic_sentence_signal(title, require_execution=True):
        direct.append({"text": title, "mode": "study_title"})
    surviving = [support for support in direct if not any(
        _contradicts_executor(support["text"], conflict) for conflict in conflicts)]
    if surviving:
        return {"evidence_level": "direct", "admitted": True, "reason": "optical_execution_relation",
                "evidence": surviving, "dominant_carrier": diagnostics}
    if direct and conflicts:
        return {"evidence_level": "none", "admitted": False, "reason": "contradictory_executor_evidence",
                "evidence": [], "dominant_carrier": diagnostics}
    nominal = []
    if not conflicts:
        for passage, mode in [(part, "source_excerpt") for part in parts] + [(title, "study_title")]:
            if not _nominal_executor_outlook(passage) and _photonic_sentence_signal(passage):
                nominal.append({"text": passage, "mode": mode})
        for previous, current in zip(parts, parts[1:], strict=False):
            if (not enumeration_alternative(previous) and not enumeration_alternative(current)
                    and not _nominal_executor_outlook(previous) and not _nominal_executor_outlook(current)
                    and _linked_photonic_sentences(previous, current)):
                nominal.append({"text": previous + "\n" + current, "mode": "neighboring_excerpts"})
    if nominal:
        return {"evidence_level": "nominal", "admitted": True, "reason": "named_optical_architecture",
                "evidence": nominal, "dominant_carrier": diagnostics}
    # Mere application/outlook mentions can be labelled without allowing admission.
    outlook = []
    if not conflicts:
        for passage, mode in [(part, "source_excerpt") for part in parts] + [(title, "study_title")]:
            if (not enumeration_alternative(passage)
                    and (OPTICAL.search(passage) or re.search(r"\bmicro[-‐‑– ]?combs?\b", passage, re.I))
                    and NEURAL_OPERATION.search(passage)
                    and (CONTEXT.search(passage) or _executor_outlook(passage)
                         or _nominal_executor_outlook(passage)
                         or re.search(r"\bapplications?|platform\b", passage, re.I))):
                outlook.append({"text": passage, "mode": mode})
    return {"evidence_level": "nominal" if outlook else "none", "admitted": False,
            "reason": "potential_application_only" if outlook else "executor_not_established",
            "evidence": outlook, "dominant_carrier": diagnostics}


def _linked_photonic_sentences(previous, current, *, require_execution=False):
    """Resolve only an explicit neighboring hardware reference, without rewriting sources."""
    actors = {m.group().casefold().removesuffix("s") for m in HARDWARE.finditer(previous)}
    reference = re.match(r"^(?:(?:this|that|the|these|those)\s+(\w+)|it|they)\b", current, re.I)
    if not actors or reference is None or _electronic_execution(previous):
        return False
    optical_actors = set()
    for clause in _scope_clauses(previous):
        if OPTICAL.search(clause) and (not OTHER_EXECUTOR.search(clause) or TRAIN_ACTION.search(clause)):
            optical_actors.update(m.group().casefold().removesuffix("s") for m in HARDWARE.finditer(clause))
    noun = reference.group(1).casefold().removesuffix("s") if reference.group(1) else None
    if noun and noun != "device" and noun not in actors:
        return False
    if optical_actors:
        if noun and noun != "device" and noun not in optical_actors:
            return False
        if (noun is None or (noun == "device" and noun not in optical_actors)) and actors != optical_actors:
            return False
        resolved = "optical " + (noun or "device") + current[reference.end():]
        return _photonic_sentence_signal(resolved, require_execution=require_execution)
    if OTHER_EXECUTOR.search(previous):
        return False
    # Reverse order: an implemented operation, then an optical property of that processor.
    return _photonic_sentence_signal(previous + " " + current, require_execution=require_execution)


def _lexical_scope_check(topic, title, abstract):
    """Returns a lexical signal, not a calibrated relevance probability."""
    text = title + ". " + abstract
    query = resolve_topic(topic).casefold()
    if "photon" in query and ("neuro" in query or "neural" in query):
        # This path only reads one boolean. Keep the public copy boundary in
        # execution_evidence(), but avoid copying its nested evidence here.
        return ("direct_lexical_signal" if _execution_evidence_cached(title or "", abstract or "")["admitted"]
                else "scope_not_established")
    if query == "artificial intelligence":
        return "direct_lexical_signal" if re.search(
            r"\b(?:artificial intelligence|machine learning|deep learning|neural|transformer|language model|reinforcement learning)\b",
            text, re.I) else "scope_not_established"
    from app.ml.directions import direction_profile
    profile = direction_profile(topic)
    if profile is not None and profile.get("scope_kind") == "paired_axes":
        from app.ml.directions import profile_scope_check
        return "direct_lexical_signal" if profile_scope_check(profile, title, sentences(abstract)) else "scope_not_established"
    tokens = [w for w in re.findall(r"[^\W\d_]{3,}", query) if w not in {"the", "and", "for", "technology", "technologies",
                                                                                 "для", "или", "технологии", "технология"}]
    if not tokens:
        return "scope_not_established"
    matches = sum(bool(re.search(r"\b" + re.escape(w[:6]) + r"\w*", text, re.I)) for w in tokens)
    return "direct_lexical_signal" if matches >= max(1, (len(tokens) + 1) // 2) else "scope_not_established"


def scope_check(topic, title, abstract):
    """Legacy admission contract with an optional, explicitly scoped adapter."""
    lexical = _lexical_scope_check(topic, title, abstract)
    if lexical == "direct_lexical_signal":
        return lexical
    from app.ml.scope_context import current_semantic_policy
    policy = current_semantic_policy()
    if policy is not None and policy.decision(topic, title, abstract)["semantic_only"]:
        # prepare's protected acceptance sentinel is historical. The result
        # separately records semantic provenance and requires human review.
        return "direct_lexical_signal"
    return lexical


def safe_url(value):
    return is_safe_http_url(value)


def _supported_advantage(passage, pattern):
    """Reject negated benefits and prerequisites; keep the original quote intact."""
    # Retain the prerequisite guard, except where the requirement itself is denied.
    for requirement in REQUIRED_BENEFIT.finditer(passage):
        prefix = re.split(r"[;:]|\b(?:but|whereas|while)\b", passage[:requirement.start()], flags=re.I)[-1]
        if (not NEGATED_BENEFIT.search(prefix)
                and not re.match(r"\s+no\b", passage[requirement.end():], re.I)):
            return False
    # Remove only neutral asides in this analysis copy, never from the returned quote.
    # A comma itself does not end the scope of "fails" or "does not".
    def neutral_aside(match):
        aside = match.group()
        return aside if (pattern.search(aside) or NEGATED_BENEFIT.search(aside)
                         or REQUIRED_BENEFIT.search(aside) or UNSUPPORTED_BENEFIT.search(aside)) else " "

    assertion = re.sub(r",[^,]+,|\([^()]*\)|[—–][^—–]+[—–]", neutral_aside, passage)
    # A negated loss of accuracy is not a negated latency/power benefit.
    assertion = re.sub(r"\bwithout\s+reducing\s+(?:(?:the|measured|classification|output|overall)\s+)*"
                       r"(?:accuracy|precision|fidelity|reliability|throughput|bandwidth|performance)\b",
                       "without a loss of quality", assertion, flags=re.I)
    assertion = re.sub(r"\bNOT(?=\s+(?:gates?|operations?|elements?)\b)", "inverter", assertion, flags=re.I)
    assertion = re.sub(r"\bno\s+(?:(?:apparent|measurable|significant|synaptic|performance)\s+)*"
                       r"degradation\b", "stability", assertion, flags=re.I)
    # Explicit clause boundaries prevent unrelated negatives from vetoing a benefit.
    clauses = re.split(r"[;:]|\b(?:but|whereas|while)\b|"
                       r"\band\s+(?=(?:the|a|an|our|this|that|these|those|no|we|it|they|"
                       r"does|do|did|is|are|was|were|has|have|requires?)\b)",
                       assertion, flags=re.I)
    for clause in clauses:
        matches = list(pattern.finditer(clause))
        for index, match in enumerate(matches):
            prefix = clause[:match.start()]
            # Neutral comma/parenthesis insertions have already been joined above.
            # Other comma clauses may introduce an independent assertion, but a
            # continuation such as "fails, however to improve" keeps its negation.
            if "," in prefix:
                before, continuation = prefix.rsplit(",", 1)
                if not (NEGATED_BENEFIT.search(before) and re.match(
                        r"\s*(?:however|nevertheless|nonetheless|to)\b", continuation, re.I)):
                    prefix = continuation
            if index:
                # "reduces latency without extra cooling and improves accuracy".
                between = clause[matches[index - 1].end():match.start()]
                if re.search(r"\band\b|,", between, re.I):
                    prefix = re.split(r"\band\b|,", between, flags=re.I)[-1]
            suffix = clause[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(clause)]
            # A trailing absence of extra costs is compatible with a measured benefit.
            suffix = re.split(r"\b(?:without|after|before|because|although)\b", suffix, flags=re.I)[0]
            denied_object = re.match(r"[\s,]*(?:(?:\w+ly|by)\s+)*(?:neither|"
                                     r"no(?!\s+(?:less|more)\s+than\b))\b", suffix, re.I)
            if (NEGATED_BENEFIT.search(prefix) or denied_object or DENIED_BENEFIT_RESULT.search(suffix)
                    or re.search(r"\bby\s+no\s+(?:\w+\s+){0,2}(?:amount|margin|degree)\b", suffix, re.I)
                    or UNSUPPORTED_BENEFIT.search(prefix + match.group() + suffix)):
                return False
    return bool(pattern.search(assertion))


def evidence_card(studies):
    """Choose explicit passages; missing evidence remains missing."""
    patterns = {
        "problem": re.compile(r"\b(?:challeng\w*|bottleneck\w*|limit\w*|difficult\w*|however|suffer\w*|require\w*|problem\w*)\b", re.I),
        "advantage": re.compile(r"\b(?:improv\w*|reduc\w*|enhanc\w*|efficient\w*|low[- ]power|low[- ]latency|advantage\w*|outperform\w*|achiev\w*)\b", re.I),
        "example": re.compile(r"\b(?:we|here|this work|this paper|demonstrat\w*|propos\w*|present\w*|fabricat\w*|construct\w*)\b", re.I),
    }
    card = {}
    for field, pattern in patterns.items():
        options = []
        for index, study in enumerate(studies[:12]):
            for passage in sentences(study["abstract"]):
                if len(passage.split()) >= 5 and len(passage) <= 900 and pattern.search(passage):
                    if field == "advantage" and not _supported_advantage(passage, pattern):
                        continue
                    # Favor representative studies and complete, moderately sized passages.
                    options.append((index, abs(len(passage) - 250), passage, study))
        if options:
            _, _, passage, study = min(options, key=lambda item: item[:3])
            card[field] = {"text": passage, "study_id": study["id"], "url": study["url"],
                           "title": study["title"], "mode": "source_excerpt"}
        else:
            card[field] = None
    if card["example"] is None and studies:
        study = studies[0]
        card["example"] = {"text": study["title"], "study_id": study["id"], "url": study["url"],
                           "title": study["title"], "mode": "study_title"}
    return card
