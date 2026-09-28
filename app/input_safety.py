"""Shared input boundaries without networking or dependencies on UI/ML."""

import re
from ipaddress import IPv4Address, IPv6Address
from urllib.parse import urlsplit

MAX_TITLE_CHARACTERS = 20_000
MAX_ABSTRACT_CHARACTERS = 200_000
MAX_URL_CHARACTERS = 4096
_HOST_LABEL = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\Z")


def is_safe_http_url(value: object) -> bool:
    """Accept browser HTTP(S) links, without rewriting or resolving their host."""
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_URL_CHARACTERS:
        return False
    # urlsplit strips some controls, while browsers reinterpret backslashes.
    # Check the original value before parsing so those forms cannot bypass us.
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 or char == "\\" for char in value):
        return False
    try:
        # A lone surrogate in the path/query survives urlsplit but later fails
        # UTF-8 encoding when an archived identity or HTTP response is hashed.
        value.encode("utf-8")
        parts = urlsplit(value)
        hostname = parts.hostname
        if (parts.scheme not in {"http", "https"} or not hostname
                or parts.username is not None or parts.password is not None):
            return False
        # Accessing port validates both numeric syntax and the 0..65535 range.
        _ = parts.port
        if parts.netloc.endswith(":"):
            return False
        if ":" in hostname:
            # Scoped/local addresses are not needed for publication links.
            if "%" in hostname:
                return False
            IPv6Address(hostname)
            return True
        ascii_host = hostname.encode("idna").decode("ascii").removesuffix(".")
        if len(ascii_host.split(".")) == 4 and all(label.isdecimal() for label in ascii_host.split(".")):
            IPv4Address(ascii_host)
        return bool(len(ascii_host) <= 253 and ascii_host and all(
            _HOST_LABEL.fullmatch(label) for label in ascii_host.split(".")))
    except (ValueError, UnicodeError):
        return False


def url_key(url: str) -> str:
    """Match the same publication across DOI case, host spelling and trailing slash."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    if host in {"doi.org", "dx.doi.org"}:
        path = path.lower()
    return f"{host}{path}?{parsed.query}"


def work_key(title: str) -> str:
    """Treat a preprint and a journal publication with the same title as one work."""
    return "t:" + re.sub(r"[^0-9a-zа-яё]+", "", title.casefold())[:160]
