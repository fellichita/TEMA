"""EPO OPS published-data adapter. Live access requires an EPO consumer key/secret."""

import base64
import json
import math
import re
import time
from datetime import date
from functools import partial
from threading import Event
from urllib.parse import urlencode
from xml.etree.ElementTree import ParseError, tostring

import httpx
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from app.backend.contracts import DatePrecision, DocumentRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.http_transport import BoundedHttpTransport, _cancelled, _positive_number

AUTH_ENDPOINT = "https://ops.epo.org/3.2/auth/accesstoken"
SEARCH_ENDPOINT = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"


class _AccessTokenRejected(BackendError):
    """Internal refresh signal; its public code/message contain no token."""

    def __init__(self):
        super().__init__("authentication_required", "EPO отклонил токен доступа.")


def _error_message(body: bytes) -> str:
    """Read only a protocol error identifier, not arbitrary response details."""
    try:
        # OPS documents XML errors, including errors from its JSON OAuth endpoint.
        if body.lstrip().startswith(b"{"):
            payload = json.loads(body)
            value = payload.get("error", "") if isinstance(payload, dict) else ""
        else:
            root = ElementTree.fromstring(body, forbid_dtd=True)
            if root.tag.rsplit("}", 1)[-1] != "error":
                return ""
            value = root.findtext("{*}message", "")
        return " ".join(value.casefold().split()) if isinstance(value, str) else ""
    except (ParseError, DefusedXmlException, ValueError, RecursionError):
        return ""


def _epo_client_error(status: int, headers: httpx.Headers, body: bytes,
                      *, authentication: bool = False) -> BackendError | None:
    message = _error_message(body)
    rejection = headers.get("X-Rejection-Reason", "").casefold()
    if status == 403 and (rejection in {
        "individualquotaperhour", "registeredquotaperweek", "registeredpayingquotaperweek",
        "individual quota exceeded",
    } or "fair use" in message or message == "individual quota exceeded"):
        return BackendError("rate_limited", "EPO ограничил доступ по квоте или правилам Fair Use. Повторите позже.")
    if message == "invalid_client":
        return BackendError("invalid_credentials", "EPO отклонил учётные данные. Проверьте ключ и секрет.")
    if message in {"invalid_request", "unsupported_grant_type"}:
        return BackendError("invalid_response", "EPO отклонил формат запроса доступа.")
    if message in {"developer account is blocked", "this request has been rejected"}:
        return BackendError("access_denied", "EPO запретил доступ. Проверьте состояние учётной записи и права доступа.")
    if not authentication and (message == "invalid_access_token" or status == 401):
        return _AccessTokenRejected()
    if status == 403:
        return BackendError("access_denied", "EPO запретил доступ. Проверьте состояние учётной записи и права доступа.")
    if authentication and status == 401:
        return BackendError("invalid_credentials", "EPO отклонил учётные данные. Проверьте ключ и секрет.")
    return None


def _text(node):
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def _normalize(node):
    country, number, kind = (node.get(key, "") for key in ("country", "doc-number", "kind"))
    if not country or not number or not kind:
        raise ValueError("Не указан полный номер публикации")
    bibliography = node.find("{*}bibliographic-data")
    if bibliography is None:
        raise ValueError("Отсутствует библиография")
    titles = bibliography.findall("{*}invention-title")
    title_node = next((title for title in titles if title.get("lang") == "en"), titles[0] if titles else None)
    title = _text(title_node)
    references = bibliography.findall("{*}publication-reference/{*}document-id")
    reference = next((ref for ref in references if ref.get("document-id-type") == "docdb"),
                     references[0] if references else None)
    raw_date = _text(reference.find("{*}date")) if reference is not None else ""
    year: int | None = None
    month: int | None = None
    published: date | None = None
    precision: DatePrecision = "unknown"
    if raw_date:
        if not re.fullmatch(r"\d{4}(\d{2})?(\d{2})?", raw_date):
            raise ValueError("Некорректная дата публикации")
        year, precision = int(raw_date[:4]), "year"
        if len(raw_date) >= 6:
            month, precision = int(raw_date[4:6]), "month"
        if len(raw_date) == 8:
            assert month is not None
            published, precision = date(year, month, int(raw_date[6:8])), "day"
    abstracts = node.findall("{*}abstract")
    abstract = next((item for item in abstracts if item.get("lang") == "en"),
                    abstracts[0] if abstracts else None)
    inventors = bibliography.findall("{*}parties/{*}inventors/{*}inventor")
    names = tuple(dict.fromkeys(name for inventor in inventors
                                if (name := _text(inventor.find("{*}inventor-name/{*}name")))))
    publication = country + number + kind
    record = DocumentRecord(
        source="epo", source_id=publication, patent_publication=publication,
        patent_family_id=node.get("family-id") or None, title=title,
        abstract=_text(abstract) or None, publication_year=year, publication_month=month,
        publication_date=published, date_precision=precision, authors=names,
        url="https://worldwide.espacenet.com/publicationDetails/biblio?" + urlencode({"CC": country, "NR": number, "KC": kind}),
        language=title_node.get("lang") if title_node is not None else None,
        document_type="patent", raw_metadata={"xml": tostring(node, encoding="unicode")},
    )
    return record.model_copy(update={"source_id": record.patent_publication})


class EpoOpsProvider:
    def __init__(self, consumer_key=None, consumer_secret=None, client=None,
                 page_size=100, timeout_seconds=15.0, max_retries=2, max_response_bytes=5_000_000,
                 retry_delay=1.0, *, page_delay=1.0, page_deadline_seconds=60.0):
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("EPO page_size должен быть 1–100")
        for value in (consumer_key, consumer_secret):
            if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 4096
                                      or any(not 33 <= ord(char) <= 126 for char in value)):
                raise BackendError("invalid_credentials", "Некорректный формат учётных данных EPO.")
        if consumer_key and ":" in consumer_key:
            raise BackendError("invalid_credentials", "Некорректный формат учётных данных EPO.")
        self.page_size = page_size
        self.page_delay = _positive_number("page_delay", page_delay, allow_zero=True)
        if self.page_delay > 30:
            raise ValueError("Недопустимая задержка")
        self._key, self._secret = consumer_key, consumer_secret
        self._token, self._expires = None, 0.0
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(follow_redirects=False, timeout=timeout_seconds)
        try:
            self._auth = BoundedHttpTransport(AUTH_ENDPOINT, self._client, timeout_seconds, 0,
                                             min(max_response_bytes, 64_000), retry_delay, page_deadline_seconds,
                                             client_error_handler=partial(_epo_client_error, authentication=True))
            self._search = BoundedHttpTransport(SEARCH_ENDPOINT, self._client, timeout_seconds, max_retries,
                                               max_response_bytes, retry_delay, page_deadline_seconds,
                                               client_error_handler=_epo_client_error)
        except BaseException:
            if self._owns_client:
                self._client.close()
            raise

    def _get_token(self, cancel):
        _cancelled(cancel)
        if not self._key or not self._secret:
            raise BackendError("credentials_required", "Для EPO задайте EPO_OPS_KEY и EPO_OPS_SECRET в окружении процесса.")
        if self._token is not None and time.monotonic() < self._expires:
            return self._token
        basic = base64.b64encode(f"{self._key}:{self._secret}".encode("ascii")).decode("ascii")
        body = self._auth.request_bytes(cancel, method="POST", data={"grant_type": "client_credentials"},
                                        headers={"Authorization": f"Basic {basic}",
                                                 "Content-Type": "application/x-www-form-urlencoded"})
        try:
            payload = json.loads(body)
            token, ttl = payload["access_token"], float(payload["expires_in"])
            if (not isinstance(token, str) or not 1 <= len(token) <= 8192
                or any(not 33 <= ord(char) <= 126 for char in token)
                or not math.isfinite(ttl) or not 0 < ttl <= 86400):
                raise ValueError("Некорректный токен")
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            raise BackendError("invalid_response", "EPO вернул некорректный ответ авторизации.") from None
        self._token, self._expires = token, time.monotonic() + max(0, ttl - 30)
        return token

    def iter_pages(self, request: SearchRequest, cancel: Event):
        _cancelled(cancel)
        if '"' in request.topic or "\\" in request.topic:
            raise BackendError("invalid_query", "В запросе EPO пока не поддерживаются кавычки и обратная косая черта.")
        # Date constraints apply to publication dates, never priority/application dates.
        query = f'ta all "{request.topic}" and pd <= "{request.until_date:%Y%m%d}"'
        if request.from_date:
            query = f'ta all "{request.topic}" and pd within "{request.from_date:%Y%m%d} {request.until_date:%Y%m%d}"'
        scanned, limit = 0, min(request.max_results, 2000)
        while scanned < limit:
            _cancelled(cancel)
            if scanned and cancel.wait(self.page_delay):
                raise CancelledError()
            end = min(scanned + self.page_size, limit)
            headers = {"Accept": "application/exchange+xml", "X-OPS-Range": f"{scanned + 1}-{end}",
                       "Authorization": f"Bearer {self._get_token(cancel)}"}
            try:
                body = self._search.request_bytes(cancel, params={"q": query}, headers=headers)
            except _AccessTokenRejected:
                # One refresh, no unbounded retry loop on invalid access.
                self._token = None
                headers["Authorization"] = f"Bearer {self._get_token(cancel)}"
                body = self._search.request_bytes(cancel, params={"q": query}, headers=headers)
            try:
                root = ElementTree.fromstring(body, forbid_dtd=True)
            except (ParseError, DefusedXmlException, ValueError, RecursionError):
                raise BackendError("invalid_response", "Некорректный XML-ответ EPO.") from None
            search = root.find(".//{*}biblio-search")
            if search is None:
                raise BackendError("invalid_response", "EPO не вернул поисковую выдачу.")
            total_raw = search.get("total-result-count")
            if not total_raw or not total_raw.isdecimal() or len(total_raw) > 10:
                raise BackendError("invalid_response", "Некорректное число результатов EPO.")
            total = int(total_raw)
            items = search.findall(".//{*}exchange-document")
            if len(items) > end - scanned or (not items and total > scanned):
                raise BackendError("invalid_response", "Неполный или некорректный ответ EPO.")
            documents = []
            for node in items:
                _cancelled(cancel)
                try:
                    documents.append(_normalize(node))
                except (ValueError, TypeError, OverflowError):
                    continue
            scanned += len(items)
            exhausted = scanned >= total
            yield SourcePage(documents=tuple(documents), scanned=len(items), skipped=len(items) - len(documents),
                             total_available=total, exhausted=exhausted)
            if exhausted:
                return

    def close(self):
        self._key = self._secret = self._token = None
        if self._owns_client:
            self._client.close()
