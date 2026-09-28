"""Versioned local direction profiles; resolution never reads or relabels a corpus."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata


PROFILE_DIRECTORY = Path(__file__).with_suffix("")
_GENERAL_ALIASES = {
    "искусственный интеллект": "artificial intelligence",
    "ии": "artificial intelligence",
    "технологии в ии": "artificial intelligence",
    "технологии ии": "artificial intelligence",
    "квантовые вычисления": "quantum computing",
}


def _normalise(value):
    if not isinstance(value, str):
        raise TypeError("Направление должно быть строкой.")
    return " ".join(unicodedata.normalize("NFKC", value).split())


@lru_cache(maxsize=1)
def _profiles():
    profiles, aliases = {}, {}
    for path in sorted(PROFILE_DIRECTORY.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        required = ("id", "profile_version", "canonical_query", "display_name_ru", "aliases", "axes")
        if not isinstance(profile, dict) or any(key not in profile for key in required):
            raise ValueError(f"Неполный профиль направления: {path.name}")
        identifier = profile["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
            raise ValueError(f"Некорректный идентификатор профиля: {path.name}")
        if identifier in profiles or type(profile["profile_version"]) is not int or profile["profile_version"] < 1:
            raise ValueError(f"Конфликт идентификатора или версии профиля: {path.name}")
        canonical = _normalise(profile["canonical_query"])
        if not canonical or not re.search(r"[a-zA-Z]", canonical):
            raise ValueError(f"Нужен английский поисковый запрос: {path.name}")
        if (not isinstance(profile["aliases"], list) or not isinstance(profile["axes"], dict)
                or set(profile["axes"]) != {"carrier", "compute"}):
            raise ValueError(f"Некорректные псевдонимы или оси профиля: {path.name}")
        for patterns in profile["axes"].values():
            if not isinstance(patterns, list) or not patterns:
                raise ValueError(f"Пустая ось направления: {path.name}")
            for pattern in patterns:
                re.compile(pattern, re.IGNORECASE)
        profiles[identifier] = profile
        for name in [canonical, profile["display_name_ru"], *profile["aliases"]]:
            key = _normalise(name).casefold()
            if not key or (key in aliases and aliases[key] != identifier):
                raise ValueError(f"Конфликт псевдонима направления: {path.name}")
            aliases[key] = identifier
    if not profiles:
        raise ValueError("Локальные профили направлений отсутствуют.")
    return profiles, aliases


def direction_profile(value):
    """Return an independent profile copy, or None for an unconfigured direction."""
    profiles, aliases = _profiles()
    identifier = aliases.get(_normalise(value).casefold())
    return deepcopy(profiles[identifier]) if identifier is not None else None


def resolve_direction(value):
    """Map known aliases; preserve unknown input instead of guessing its meaning."""
    value = _normalise(value)
    profiles, aliases = _profiles()
    identifier = aliases.get(value.casefold())
    if identifier is not None:
        # Only return an immutable string; direction_profile() still copies
        # mutable profiles for callers that need the full definition.
        return profiles[identifier]["canonical_query"]
    return _GENERAL_ALIASES.get(value.casefold(), value)


def profile_fingerprint(value):
    """Stable hash of the complete effective profile; no profile means no hash."""
    profile = direction_profile(value)
    if profile is None:
        return None
    payload = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_OWN_WORK = re.compile(
    r"\b(?:we|our\s+(?:approach|work|study|scheme|system|model|results?)|here(?:in)?|"
    r"this\s+(?:work|paper|study|review))\b.{0,110}\b"
    r"(?:propos\w*|present\w*|develop\w*|implement\w*|demonstrat\w*|investigat\w*|"
    r"introduc\w*|analy[sz]\w*|construct\w*|review\w*|survey\w*|report\w*|show\w*|"
    r"explor\w*|deriv\w*|establish\w*|achiev\w*|reali[sz]\w*|formulat\w*|"
    r"evaluat\w*|consider\w*|model\w*|use\w*|study|studied|approach|apply\w*|employ\w*|"
    r"increas\w*|improv\w*|encod\w*|stor\w*|record\w*|retriev\w*|design\w*|"
    r"focus\w*|characteri[sz]\w*)\b|"
    r"\b(?:is|are)\s+(?:(?:here|experimentally|numerically)\s+)*"
    r"(?:proposed|presented|introduced|developed|demonstrated|investigated|implemented)\b", re.I)
_APPLICATION_LIST = re.compile(r"\b(?:including|such\s+as|applications?\s+(?:include|in|to)|"
                               r"ranging\s+from)\b", re.I)
_REFERS_BACK = re.compile(r"^(?:this|these|such|the\s+proposed)\s+(?:[a-z]+\s+){0,7}(?:system|approach|scheme|model|device|method|"
                          r"architecture|achievement|result|code)\w*\b|"
                          r"\b(?:we\s+(?:survey|review|investigate|study|evaluate|extend)\s+"
                          r"(?:these|this|such)|for\s+this\s+(?:channel|model|system))\b", re.I)
_QUANTUM_DOT = re.compile(r"\bquantum\s+dots?\b", re.I)
_QUANTUM_RESERVOIR = re.compile(r"\bquantum\s+(?:(?:physical|optical|photonic|neural|classical|"
                                r"next|generation|noise|induced|transport|extreme)\s+){0,3}"
                                r"reservoir\s+(?:comput\w*|process\w*|networks?)\b", re.I)
_QUANTUM_PHYSICS = re.compile(r"\b(?:qubits?|transmons?|hamiltonian\w*|lindblad\w*|boson\w*|"
                              r"quantum\s+(?:states?|dynamics?|circuits?|recurrent|neural|systems?|models?)|"
                              r"density\s+matri\w*|spin\s+chains?|schr[oö]dinger)\b", re.I)
_CLASSICAL_RESERVOIR = re.compile(r"(?<!quantum )\bclassical\s+(?:(?:electronic|digital|photonic|optical)\s+)?"
                                 r"reservoir\b", re.I)
_DNA_PHYSICAL = re.compile(r"\b(?:synthe(?:tic|sis|si[sz]\w*)|sequenc(?:ing|ed)|oligonucleotides?|"
                           r"wet\s+experiments?|nanopores?|ligat\w*|hybridi[sz]\w*|"
                           r"strand\s+displacement|origami|microarrays?)\b", re.I)
_CRYPTO = re.compile(r"\b(?:crypt\w*|encrypt\w*|elgamal|aes|xor)\b", re.I)
_CODING = re.compile(r"\b(?:codes?|coding|channels?|redundancy|deletions?|insertions?|"
                     r"substitutions?|tandem\s+duplication)\b", re.I)


def profile_scope_check(profile, title, parts):
    """Conservative topic relevance, not proof of a physical experiment or novelty.

    A named topic in the title, or the authors' own topic-related work, supports
    admission. Background application lists do not. Quantum-dot material names
    are not quantum computation; symbolic DNA encryption is not DNA storage.
    The established photonic executor checker remains a separate code path.
    """
    if profile.get("scope_kind") != "paired_axes":
        raise ValueError("Для этого профиля нужен отдельный фильтр исполнителя.")
    normalise = lambda text: re.sub(r"[-‐‑‒–—−]", " ", text)
    title, parts = normalise(title), [normalise(part) for part in parts]

    def both(text):
        if profile["id"] == "dna_data_storage":
            biological_storage = re.search(r"\b(?:diagnos\w*|molecular\s+testing|tuberculosis|"
                                            r"mycobacter\w*|blood\s+samples?|biobank\w*)\b", text, re.I)
            digital_content = re.search(r"\b(?:digital|data|information|files?|images?|bits?|"
                                         r"codes?|coding|channels?|encoding)\b", text, re.I)
            if biological_storage and not digital_content:
                return False
        return all(any(re.search(pattern, text, re.I) for pattern in patterns)
                   for patterns in profile["axes"].values())

    def foreground(text):
        # Keep an independent claim before a list, instead of banning the paper.
        marker = _APPLICATION_LIST.search(text)
        text = text[:marker.start()] if marker else text
        if profile["id"] == "dna_data_storage":
            alternatives = list(re.finditer(r"\b(?:logic\s+gates?|dna\s+circuits?|biosensing|"
                                             r"diagnostics|machine\s+learning|drug\s+delivery)\b", text, re.I))
            storage = re.search(r"\b(?:data|information)\s+storage\b", text, re.I)
            if len(alternatives) >= 2 and storage and alternatives[0].start() < storage.start():
                text = text[:alternatives[0].start()]
        return text

    own = [foreground(part) for part in parts if _OWN_WORK.search(part)]
    linked = []
    for previous, current in zip(parts, parts[1:], strict=False):
        reference_text = re.sub(r"\([^)]{1,30}\)", "", current)
        if _REFERS_BACK.search(reference_text) and (_OWN_WORK.search(previous) or _OWN_WORK.search(current)):
            linked.append(foreground(previous) + " " + foreground(current))
    candidates = own + linked
    if profile["id"] == "quantum_reservoir":
        def material_free(text):
            return re.sub(r"\bpost\s+quantum\b", "postclassical", _QUANTUM_DOT.sub("material dots", text), flags=re.I)
        computing_context = (any(both(material_free(part)) for part in [title, *parts])
                             or any(re.search(pattern, title, re.I) for pattern in profile["axes"]["compute"]))
        # Remove only the quantum-dot material label from carrier evidence.
        def quantum_support(text):
            text = material_free(text)
            text = re.split(r"\b(?:compared\s+(?:with|to)|advantage\s+over|in\s+comparison\s+with)\b", text, flags=re.I)[0]
            if _CLASSICAL_RESERVOIR.search(text) and not _QUANTUM_RESERVOIR.search(text):
                return False
            return bool((_QUANTUM_RESERVOIR.search(text) and both(text))
                        or (computing_context and re.search(r"\breservoir\w*\b", text, re.I)
                            and _QUANTUM_PHYSICS.search(text)))

        if any(quantum_support(part) for part in candidates):
            return True
        expanded = any(_QUANTUM_RESERVOIR.search(part) and re.search(r"\bqr[cp]\b", part, re.I) for part in parts)
        if expanded and any(re.search(r"\bqr[cp]\b", part, re.I) and not _CLASSICAL_RESERVOIR.search(part)
                            for part in own):
            return True
        if any(_CLASSICAL_RESERVOIR.search(part) for part in own):
            return False
        # State tomography is a task; its title alone does not specify the carrier.
        topic_title = material_free(title)
        quantum_task = re.search(r"\bquantum\s+(?:state\s+)?(?:tomography|measurements?|classification)\b", topic_title, re.I)
        named_title = bool(_QUANTUM_RESERVOIR.search(title)
                           or (both(topic_title) and not quantum_task)
                           or (computing_context and re.search(r"\bquantum\s+reservoir\s+complexity\b", title, re.I)))
        return named_title and (not _QUANTUM_DOT.search(title) or bool(_QUANTUM_PHYSICS.search(title)))

    if profile["id"] == "dna_data_storage":
        biological_dataset = re.search(r"\bdna\s+(?:methylation|sequencing|sequence|genomic)\s+data\b", title, re.I)
        physical_payload = any(re.search(r"\b(?:data|files?|bits?|information)\b.{0,65}\b(?:in|on|onto|into)\s+"
                                         r"(?:(?:synthetic|native|methylated)\s+)?dna\b", part, re.I) for part in candidates)
        if biological_dataset and not physical_payload:
            return False
        crypto_topic = bool(_CRYPTO.search(title))
        cloud_topic = bool(re.search(r"\bcloud\b", title, re.I))
        physical_own = any(both(part) and _DNA_PHYSICAL.search(part) for part in candidates)
        # Encryption is allowed when the work actually addresses DNA molecules.
        # Ordinary cloud encryption with an ACGT alphabet has no such evidence.
        if cloud_topic and not physical_own:
            physical_title = both(title) and bool(re.search(r"\b(?:using|in|on)\s+synthetic\s+dna\b", title, re.I))
            if not physical_title:
                return False
        if crypto_topic and not physical_own:
            if not any(both(part) and _DNA_PHYSICAL.search(part) for part in [title, *parts]):
                return False
        if both(title) or any(both(part) for part in candidates):
            return True
        # Coding theory for a DNA storage channel remains relevant even when the
        # paper proposes a mathematical construction rather than a wet experiment.
        storage_context = any(both(part) for part in parts)
        own_coding = any(_CODING.search(part) and re.search(r"\b(?:codes?|coding|channel\w*)\b", part, re.I)
                         for part in own)
        return bool(storage_context and own_coding)
    return False
