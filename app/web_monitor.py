"""Живая сводка веб-сервиса для панели управления владельца.

Кто сейчас на сайте и что делает, какой анализ идёт и на каком он этапе, что
случилось с начала запуска. Здесь же рычаги владельца над посетителями:
блокировка, завершение сеанса и сообщения. Всё, кроме списка блокировок, живёт
в памяти процесса API и начинается заново с его перезапуском — как и связь
посетителей с их анализами. Браузеры и адреса посетителей на диск не пишутся;
в файл блокировок попадает только то, что владелец заблокировал сам.
"""

from __future__ import annotations

from collections import Counter, deque
from datetime import UTC, datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
from threading import Lock
import time
from typing import Any, Callable, cast

# Фоновую вкладку браузер будит раз в минуту, а страница отмечается раз в
# 15 секунд: полторы минуты без отметок — посетитель ушёл.
ONLINE_SECONDS = 90
# Давно ушедшие вкладки исчезают из списка; посетитель и его счётчики остаются.
FORGET_SECONDS = 6 * 3600
MAX_SESSIONS = 300
MAX_VISITORS = 1_000
MAX_PEOPLE = 1_000
MAX_EVENTS = 500
MAX_RUNS = 60
MAX_SEGMENTS = 400
MAX_ACTIONS = 80
MAX_MESSAGES = 5
MAX_MESSAGE_CHARACTERS = 500
MAX_BLOCKS = 500
# Повтор того же действия (листание, опрос истории) обновляет прежнюю запись.
REPEAT_SECONDS = 90
# Попытка заблокированного зайти попадает в журнал не чаще раза в пять минут.
ATTEMPT_SECONDS = 300
SESSION = re.compile(r"[A-Za-z0-9_-]{8,64}\Z")
VISITOR = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
STAGE = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
PERSON = re.compile(r"[0-9a-f]{12}\Z")
BLOCK_ID = re.compile(r"[A-Za-z0-9_-]{8,32}\Z")
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
PAGES = {"login": "Экран входа", "main": "Главная", "saved": "Анализ из истории",
         "report": "Отчёт по технологии"}
# Что видно на главной: форма, свой анализ, результат, ожидание и т. п.
VIEWS = frozenset({"idle", "waiting", "running", "result", "busy", "paused", "error", "history"})
PRESENCE_EVENTS = frozenset({"login", "login_failed"})
MODE_NAMES = {"fast": "быстрый", "deep": "глубокий"}
# Этапы пайплайна словами владельца; неизвестный этап показывается своим кодом.
STAGE_LABELS = {
    "plan": "Границы направления",
    "discovery": "Сбор публикаций",
    "relevance": "Смысловые группы",
    "labels": "Названия тем",
    "history": "Статистика по годам",
    "evidence": "Проверка доказательств",
    "antecedents": "Ранние аналоги",
    "enrichment": "Патенты",
    "verify": "Проверка карточек",
    "external_sources": "Дополнительные источники",
    "publish": "Оценка публикаций моделью",
    "reconcile": "Сверка статусов",
    "radar": "ТОП-15 технологий",
    "complete": "Завершение",
}
BROWSERS = ((r"YaBrowser/", "Яндекс Браузер"), (r"Edg(?:e|A|iOS)?/", "Edge"), (r"OPR/|Opera", "Opera"),
            (r"SamsungBrowser/", "Samsung Internet"), (r"Firefox/|FxiOS/", "Firefox"),
            (r"Chrome/|CriOS/", "Chrome"), (r"Safari/", "Safari"))
SYSTEMS = ((r"Android", "Android"), (r"iPhone|iPad|iPod", "iOS"), (r"Windows", "Windows"),
           (r"Mac OS X|Macintosh", "macOS"), (r"CrOS", "ChromeOS"), (r"Linux", "Linux"))


def stage_label(stage: object) -> str | None:
    if not isinstance(stage, str) or STAGE.fullmatch(stage) is None:
        return None
    return STAGE_LABELS.get(stage, stage)


def describe_agent(agent: object) -> str | None:
    """«Chrome · Windows» из строки браузера; без строки — ничего."""
    if not isinstance(agent, str) or not agent.strip():
        return None
    browser = next((name for pattern, name in BROWSERS if re.search(pattern, agent)), "Браузер")
    system = next((name for pattern, name in SYSTEMS if re.search(pattern, agent)), None)
    parts = [browser, system] if system else [browser]
    if "Mobile" in agent and system not in {"Android", "iOS"}:
        parts.append("телефон")
    return " · ".join(parts)


def clean_ip(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


# Локальная сеть владельца (RFC 1918 и уникальные локальные IPv6).
LOCAL_NETWORKS = tuple(map(ipaddress.ip_network, ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")))


def is_local_ip(value: str | None) -> bool:
    """Адрес этого компьютера или локальной сети: его блокировка закрыла бы сайт владельцу."""
    if value is None:
        return True
    address = ipaddress.ip_address(value)
    return (address.is_loopback or address.is_link_local or address.is_unspecified
            or any(address in network for network in LOCAL_NETWORKS if network.version == address.version))


def doing_text(page: str, view: str | None, detail: str | None) -> str:
    """Чем занят посетитель, словами для владельца."""
    if page == "login":
        return "на экране входа"
    if page == "saved":
        return f"смотрит анализ из истории «{detail}»" if detail else "смотрит анализ из истории"
    if page == "report":
        return f"читает отчёт о технологии «{detail}»" if detail else "читает отчёт о технологии"
    if view == "running" and detail:
        return f"следит за своим анализом «{detail}»"
    if view == "result" and detail:
        return f"смотрит результат анализа «{detail}»"
    return {"idle": "на странице запроса", "waiting": "ждёт своей очереди на анализ",
            "running": "следит за своим анализом",
            "result": "смотрит результат анализа", "busy": "ждёт, пока освободится сервис",
            "paused": "ждёт, пока владелец снова откроет запуск", "error": "видит ошибку анализа",
            "history": "открыл историю анализов"}.get(view or "", "на главной")


def _iso(moment: float | None) -> str | None:
    return None if moment is None else datetime.fromtimestamp(moment, UTC).isoformat()


def _clip(text: object, limit: int) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds} с"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {seconds} с" if seconds else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин"


class Blocklist:
    """Заблокированные браузеры и адреса; файл переживает перезапуски сервиса.

    Потокобезопасность обеспечивает владелец объекта (Monitor держит свой замок).
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path is None:
            return
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for entry in saved if isinstance(saved, list) else []:
            if (isinstance(entry, dict) and isinstance(entry.get("id"), str) and BLOCK_ID.fullmatch(entry["id"])
                    and (entry.get("kind") == "visitor" and isinstance(entry.get("value"), str)
                         and VISITOR.fullmatch(entry["value"])
                         or entry.get("kind") == "ip" and clean_ip(entry.get("value")) == entry.get("value"))):
                self.entries[entry["id"]] = {"id": entry["id"], "kind": entry["kind"], "value": entry["value"],
                                             "label": _clip(entry.get("label"), 120) or "—",
                                             "at": entry.get("at") if isinstance(entry.get("at"), str) else None}
            if len(self.entries) >= MAX_BLOCKS:
                break

    def match(self, visitor: str | None, ip: str | None) -> dict[str, Any] | None:
        return next((entry for entry in self.entries.values()
                     if entry["kind"] == "visitor" and entry["value"] == visitor
                     or entry["kind"] == "ip" and ip is not None and entry["value"] == ip), None)

    def add(self, kind: str, value: str, label: str, at: str | None) -> dict[str, Any]:
        existing = next((entry for entry in self.entries.values()
                         if entry["kind"] == kind and entry["value"] == value), None)
        if existing is not None:
            return existing
        while len(self.entries) >= MAX_BLOCKS:
            del self.entries[next(iter(self.entries))]
        entry: dict[str, Any] = {"id": secrets.token_urlsafe(9), "kind": kind, "value": value,
                                 "label": label, "at": at}
        self.entries[entry["id"]] = entry
        self._save()
        return entry

    def remove(self, block_id: str) -> dict[str, Any] | None:
        entry = self.entries.pop(block_id, None)
        if entry is not None:
            self._save()
        return entry

    def _save(self) -> None:
        if self.path is None:
            return
        # Сначала целиком во временный файл: оборванная запись не стирает список.
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(list(self.entries.values()), ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            temporary.unlink(missing_ok=True)


class Monitor:
    """Журнал, присутствие и действия посетителей, этапы анализов и рычаги владельца."""

    def __init__(self, *, anonymous: str = "Посетитель", clock: Callable[[], float] = time.time,
                 blocklist: Blocklist | None = None):
        self._clock = clock
        self._lock = Lock()
        self._anonymous = anonymous
        self.started_at = clock()
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self._sequence = 0
        # session → вкладка браузера; group → человек за вкладками; visitor → вошедший гость.
        self._sessions: dict[str, dict[str, Any]] = {}
        self._people: dict[str, dict[str, Any]] = {}
        self._keys: dict[str, str] = {}
        self._visitors: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._counters: Counter[str] = Counter()
        self._blocks = blocklist if blocklist is not None else Blocklist()
        # Завершённые владельцем сеансы: такой вход больше не принимается.
        self._revoked: set[str] = set()

    # --- журнал -------------------------------------------------------------

    def event(self, kind: str, text: str, *, level: str = "info", at: float | None = None) -> None:
        with self._lock:
            self._event(kind, text, level=level, at=at)

    def _event(self, kind: str, text: str, *, level: str = "info", at: float | None = None) -> None:
        self._sequence += 1
        self._events.append({"seq": self._sequence, "at": _iso(self._clock() if at is None else at),
                             "kind": kind, "level": level, "text": text[:700]})

    # --- люди ---------------------------------------------------------------

    def label(self, visitor: str | None) -> str:
        with self._lock:
            return self._label(visitor)

    def _label(self, visitor: str | None) -> str:
        if visitor is None:
            return self._anonymous
        known = self._visitors.get(visitor)
        return f"Гость {known['number']}" if known is not None else "Гость"

    def _group_label(self, group: str) -> str:
        if group.startswith("visitor:"):
            return self._label(group.removeprefix("visitor:"))
        if group.startswith("login:"):
            return "Не вошёл"
        return self._anonymous

    def _subject(self, group: str) -> str:
        """Кто действует — в начале строки журнала."""
        return "Кто-то" if group.startswith("login:") else self._group_label(group)

    def _visitor(self, visitor: str, now: float) -> tuple[dict[str, Any], bool]:
        known = self._visitors.get(visitor)
        if known is not None:
            return known, False
        if len(self._visitors) >= MAX_VISITORS:
            del self._visitors[min(self._visitors, key=lambda key: self._visitors[key]["last"])]
        self._counters["visitors"] += 1
        known = {"number": self._counters["visitors"], "first": now, "last": now, "runs": 0}
        self._visitors[visitor] = known
        return known, True

    def _person(self, group: str, now: float) -> dict[str, Any]:
        person = self._people.get(group)
        if person is None:
            if len(self._people) >= MAX_PEOPLE:
                oldest = min(self._people, key=lambda key: self._people[key]["last"])
                self._keys.pop(self._people.pop(oldest)["key"], None)
            key = hashlib.sha256(group.encode("utf-8")).hexdigest()[:12]
            person = {"key": key, "first": now, "last": now, "actions": deque(maxlen=MAX_ACTIONS),
                      "messages": [], "attempt": 0.0}
            self._people[group] = person
            self._keys[key] = group
        return person

    def _act(self, group: str, text: str, now: float, *, repeat: str | None = None, feed: bool = True,
             level: str = "info") -> None:
        """Действие человека: в его ленту и, если это новость, в общий журнал."""
        person = self._person(group, now)
        actions = person["actions"]
        if repeat is not None and actions and actions[-1]["repeat"] == repeat and now - actions[-1]["t"] < REPEAT_SECONDS:
            actions[-1].update(text=text, t=now, at=_iso(now))
            return
        actions.append({"t": now, "at": _iso(now), "text": text, "repeat": repeat, "level": level})
        if feed:
            self._event("action", f"{self._subject(group)} {text}", level=level)

    def visitor_action(self, visitor: str | None, text: str, *, repeat: str | None = None) -> None:
        """Действие посетителя, замеченное самим API: история, публикации, анализ из истории."""
        now = self._clock()
        with self._lock:
            self._act("visitor:" + visitor if visitor is not None else "anonymous", text, now, repeat=repeat)

    @staticmethod
    def _group(visitor: str | None, page: str, address: str | None, session: str) -> str:
        """Кого представляет вкладка: вошедшего гостя, владельца без пароля или
        ещё не вошедшего человека (их различает только адрес)."""
        if visitor is not None:
            return "visitor:" + visitor
        if page == "login":
            return "login:" + (address or session)
        return "anonymous"

    def _online(self, group: str) -> bool:
        return any(not tab["gone"] and tab["group"] == group for tab in self._sessions.values())

    def _refusal(self, visitor: str | None, address: str | None, group: str, now: float) -> str | None:
        """Закрыт ли вход: блокировка важнее завершённого сеанса."""
        if self._blocks.match(visitor, address) is not None:
            person = self._person(group, now)
            if now - person["attempt"] > ATTEMPT_SECONDS:
                person["attempt"] = now
                self._counters["blocked_attempts"] += 1
                self._act(group, "пытается зайти, но заблокирован", now, level="warning")
            return "blocked"
        if visitor is not None and visitor in self._revoked:
            return "signed_out"
        return None

    def access(self, visitor: str | None, ip: object = None) -> str | None:
        """Отказ для этого браузера и адреса — `blocked` или `signed_out` — или ничего."""
        now = self._clock()
        address = clean_ip(ip)
        with self._lock:
            return self._refusal(visitor, address, self._group(visitor, "login", address, "check-only"), now)

    def presence(self, session: str, visitor: str | None, page: str, *, agent: object = None,
                 ip: object = None, event: str | None = None, view: str | None = None,
                 detail: object = None) -> dict[str, Any]:
        """Отметка открытой вкладки сайта: при загрузке страницы и раз в 15 секунд.

        Перезагрузка страницы — новая вкладка Streamlit, поэтому «пришёл» и «ушёл»
        считаются по человеку (группе вкладок), а не по вкладке. Ответ говорит
        странице, закрыт ли ей вход и что ей написал владелец.
        """
        if (SESSION.fullmatch(session) is None or page not in PAGES
                or event is not None and event not in PRESENCE_EVENTS or view is not None and view not in VIEWS):
            raise ValueError("invalid presence")
        now = self._clock()
        agent_text = _clip(agent, 400)
        address = clean_ip(ip)
        detail_text = _clip(detail, 160)
        with self._lock:
            self._sweep(now)
            group = self._group(visitor, page, address, session)
            refusal = self._refusal(visitor, address, group, now)
            if refusal is not None:
                return {"access": refusal, "messages": []}
            where = ", ".join(part for part in (describe_agent(agent_text), address) if part)
            where = f" ({where})" if where else ""
            was_online = self._online(group)
            known, fresh = self._visitor(visitor, now) if visitor is not None else (None, False)
            if known is not None:
                known["last"] = now
            person = self._person(group, now)
            person["last"] = now
            tab = self._sessions.get(session)
            if tab is None:
                if len(self._sessions) >= MAX_SESSIONS:
                    del self._sessions[min(self._sessions, key=lambda key: self._sessions[key]["last"])]
                tab = self._sessions[session] = {"first": now, "beats": 0, "doing": None}
            doing = doing_text(page, view, detail_text)
            changed = tab.get("group") == group and tab["doing"] != doing
            tab.update(visitor=visitor, page=page, group=group, last=now, gone=False, beats=tab["beats"] + 1,
                       doing=doing)
            if agent_text:
                tab["agent"] = agent_text
            if address:
                tab["ip"] = address
            who = self._label(visitor)
            if event == "login":
                self._counters["logins"] += 1
                self._act(group, "вошёл по паролю" + where, now, level="success")
            elif event == "login_failed":
                self._counters["login_failures"] += 1
                self._act(group, "ввёл неверный пароль" + where, now, level="warning")
            elif not was_online:
                if group.startswith("login:"):
                    self._act(group, "открыл экран входа" + where, now, feed=False)
                    self._event("visit", f"Кто-то открыл экран входа{where}")
                else:
                    self._act(group, ("открыл сайт" if fresh or visitor is None else "снова на сайте") + where,
                              now, feed=False)
                    self._event("visit", f"{who} {'открыл сайт' if fresh or visitor is None else 'снова на сайте'}{where}")
            elif changed:
                self._act(group, doing, now)
            messages = [message["text"] for message in person["messages"]]
            if messages:
                person["messages"].clear()
                self._act(group, f"получил сообщение владельца ({len(messages)})", now, feed=False)
                self._event("message", f"{self._group_label(group)} получил сообщение владельца", level="success")
            return {"access": None, "messages": messages}

    def _sweep(self, now: float) -> None:
        """Отметить ушедших и забыть давно закрытые вкладки."""
        for session, tab in list(self._sessions.items()):
            if now - tab["last"] > FORGET_SECONDS:
                del self._sessions[session]
                continue
            if tab["gone"] or now - tab["last"] <= ONLINE_SECONDS:
                continue
            tab["gone"] = True
            # Вторая вкладка того же человека ещё открыта — он не ушёл. Уход с экрана
            # входа не событие: так выглядит и удачный вход.
            if not tab["group"].startswith("login:") and not self._online(tab["group"]):
                self._person(tab["group"], now)["actions"].append(
                    {"t": tab["last"], "at": _iso(tab["last"]), "text": "ушёл с сайта", "repeat": None,
                     "level": "info"})
                self._event("visit", f"{self._group_label(tab['group'])} ушёл с сайта", at=tab["last"])

    # --- рычаги владельца ---------------------------------------------------

    def _target(self, key: str) -> str:
        if not isinstance(key, str) or PERSON.fullmatch(key) is None or key not in self._keys:
            raise KeyError("person")
        return self._keys[key]

    def _last_ip(self, group: str) -> str | None:
        tabs = sorted((tab for tab in self._sessions.values() if tab["group"] == group and tab.get("ip")),
                      key=lambda tab: -tab["last"])
        return tabs[0]["ip"] if tabs else None

    def block(self, key: str) -> dict[str, Any]:
        """Закрыть вход браузеру человека и его адресу (кроме адресов этого компьютера и сети)."""
        now = self._clock()
        with self._lock:
            group = self._target(key)
            label = self._group_label(group)
            visitor = group.removeprefix("visitor:") if group.startswith("visitor:") else None
            address = self._last_ip(group)
            blocked = []
            if visitor is not None:
                blocked.append(self._blocks.add("visitor", visitor, label, _iso(now)))
            if address is not None and not is_local_ip(address):
                blocked.append(self._blocks.add("ip", address, label, _iso(now)))
            if not blocked:
                raise LookupError("Нечего блокировать: у посетителя нет входа по паролю и внешнего адреса.")
            what = " и ".join("браузер" if entry["kind"] == "visitor" else f"адрес {entry['value']}"
                              for entry in blocked)
            self._counters["blocks"] += 1
            self._act(group, f"заблокирован владельцем ({what})", now, feed=False, level="error")
            self._event("owner", f"Владелец заблокировал: {label} — {what}", level="warning")
            return {"visitor": visitor, "message": f"{label} заблокирован: {what}."}

    def unblock(self, block_id: str) -> str:
        with self._lock:
            entry = self._blocks.remove(block_id) if isinstance(block_id, str) else None
            if entry is None:
                raise KeyError("block")
            what = "браузер" if entry["kind"] == "visitor" else f"адрес {entry['value']}"
            self._event("owner", f"Владелец снял блокировку: {entry['label']} — {what}")
            return f"Блокировка снята: {entry['label']}, {what}."

    def sign_out(self, key: str) -> str:
        """Завершить вход гостя: чтобы вернуться, ему придётся снова ввести пароль."""
        now = self._clock()
        with self._lock:
            group = self._target(key)
            if not group.startswith("visitor:"):
                raise LookupError("Выгнать можно только вошедшего по паролю гостя.")
            self._revoked.add(group.removeprefix("visitor:"))
            label = self._group_label(group)
            self._act(group, "выгнан владельцем: сеанс завершён", now, feed=False, level="warning")
            self._event("owner", f"Владелец завершил сеанс: {label}", level="warning")
            return f"{label}: сеанс завершён, для входа нужен пароль."

    def message(self, key: str, text: str) -> str:
        """Сообщение появится на странице человека при её следующей отметке (до 15 секунд)."""
        cleaned = _clip(text, MAX_MESSAGE_CHARACTERS)
        if cleaned is None:
            raise ValueError("empty message")
        now = self._clock()
        with self._lock:
            group = self._target(key)
            person = self._person(group, now)
            person["messages"] = (person["messages"] + [{"text": cleaned, "at": _iso(now)}])[-MAX_MESSAGES:]
            label = self._group_label(group)
            self._act(group, f"владелец пишет: «{cleaned}»", now, feed=False)
            self._event("owner", f"Владелец → {label}: «{cleaned}»")
            return f"Сообщение для «{label}» появится на его странице в течение 15 секунд."

    def blocked_visitor(self, visitor: str) -> str | None:
        """Отказ API для запроса от имени этого посетителя."""
        with self._lock:
            if self._blocks.match(visitor, None) is not None:
                return "blocked"
            return "signed_out" if visitor in self._revoked else None

    # --- анализы ------------------------------------------------------------

    def run_started(self, run_id: str, query: str, mode: str, visitor: str | None) -> None:
        now = self._clock()
        with self._lock:
            if visitor is not None:
                known, _ = self._visitor(visitor, now)
                known["runs"] += 1
            self._counters["runs_started"] += 1
            self._runs[run_id] = {"query": query, "mode": mode, "visitor": visitor, "started": now,
                                  "segments": [], "state": "queued", "finished": None}
            while len(self._runs) > MAX_RUNS:
                del self._runs[next(iter(self._runs))]
            text = f"запустил анализ «{_clip(query, 160)}» · {MODE_NAMES.get(mode, mode)}"
            self._act("visitor:" + visitor if visitor is not None else "anonymous", text, now, feed=False)
            self._event("run", f"{self._label(visitor)} {text}")

    def cancel_requested(self, run_id: str, who: str, *, visitor: str | None = None) -> None:
        now = self._clock()
        with self._lock:
            run = self._runs.get(run_id)
            query = f" «{_clip(run['query'], 120)}»" if run is not None else ""
            if visitor is not None:
                self._act("visitor:" + visitor, f"отменяет свой анализ{query}", now, feed=False)
            self._event("run", f"{who} отменяет анализ{query}", level="warning")

    def observe(self, run_id: str, row: dict[str, Any]) -> None:
        """Сверить запуск с его сохранённой строкой: новый этап, завершение, ошибка."""
        state, stage = row.get("state"), row.get("stage")
        now = self._clock()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run["state"] in TERMINAL:
                return
            segments = run["segments"]
            if (state not in TERMINAL and stage_label(stage) is not None
                    and (not segments or segments[-1]["stage"] != stage)):
                if segments:
                    segments[-1]["ended"] = now
                if len(segments) < MAX_SEGMENTS:
                    segments.append({"stage": stage, "started": now, "ended": None})
                # Этап попадает в журнал один раз; возвраты к нему видны в хронологии.
                if sum(segment["stage"] == stage for segment in segments) == 1:
                    self._event("stage", f"Этап: {stage_label(stage)}")
            if state not in TERMINAL:
                run["state"] = state if isinstance(state, str) else run["state"]
                return
            if segments and segments[-1]["ended"] is None:
                segments[-1]["ended"] = now
            run.update(state=state, finished=now)
            query = _clip(run["query"], 120)
            took = _duration(now - run["started"])
            self._counters["runs_" + state] += 1
            if state == "succeeded":
                text, level = f"анализ «{query}» готов за {took}", "success"
            elif state == "cancelled":
                text, level = f"анализ «{query}» отменён через {took}", "warning"
            else:
                error = _clip(row.get("error"), 300) or "причина не указана"
                text, level = f"анализ «{query}» не завершён через {took}: {error}", "error"
            group = "visitor:" + run["visitor"] if run["visitor"] is not None else "anonymous"
            self._act(group, text, now, feed=False, level=level)
            self._event("run", text[0].upper() + text[1:], level=level)

    def run_record(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else {"query": run["query"], "mode": run["mode"],
                                             "visitor": self._label(run["visitor"])
                                             if run["visitor"] is not None else None}

    def timeline(self, run_id: str) -> list[dict[str, Any]]:
        """Этапы запуска в порядке появления с суммарным временем каждого."""
        now = self._clock()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return []
            stages: dict[str, dict[str, Any]] = {}
            for segment in run["segments"]:
                entry = stages.setdefault(segment["stage"], {
                    "stage": segment["stage"], "label": stage_label(segment["stage"]),
                    "started_at": _iso(segment["started"]), "seconds": 0.0, "active": False})
                entry["seconds"] += (segment["ended"] or now) - segment["started"]
                entry["active"] = segment["ended"] is None
                # Этап мог вернуться: часы «сколько на этапе» идут от последнего входа в него.
                entry["current_since"] = _iso(segment["started"]) if entry["active"] else None
            return [{**entry, "seconds": round(entry["seconds"])} for entry in stages.values()]

    # --- сводка -------------------------------------------------------------

    def snapshot(self, after: int = 0, person: str | None = None) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            self._sweep(now)
            people: dict[str, dict[str, Any]] = {}
            # Свежая вкладка человека первой: её страница, браузер и адрес — текущие.
            for tab in sorted(self._sessions.values(), key=lambda tab: -tab["last"]):
                group = tab["group"]
                entry = people.get(group)
                if entry is None:
                    visitor = tab.get("visitor")
                    known = self._visitors.get(visitor) if visitor is not None else None
                    record = self._person(group, now)
                    address = self._last_ip(group)
                    entry = people[group] = {
                        "key": record["key"], "label": self._group_label(group),
                        "number": known["number"] if known is not None else None,
                        "page": PAGES.get(cast(str, tab.get("page")), "—"), "doing": tab.get("doing"),
                        "agent": describe_agent(tab.get("agent")), "agent_raw": tab.get("agent"), "ip": address,
                        "first_seen": _iso(known["first"] if known is not None else record["first"]),
                        "last_seen": _iso(tab["last"]), "idle_seconds": round(now - tab["last"]),
                        "runs": known["runs"] if known is not None else 0, "tabs": 0, "online": False,
                        "signed_in": visitor is not None,
                        "blocked": self._blocks.match(visitor, address) is not None,
                        "block_ids": [entry["id"] for entry in self._blocks.entries.values()
                                      if entry["kind"] == "visitor" and entry["value"] == visitor
                                      or entry["kind"] == "ip" and entry["value"] == address],
                        "signed_out": visitor is not None and visitor in self._revoked,
                        "can_block_ip": address is not None and not is_local_ip(address),
                        "pending_messages": len(record["messages"])}
                entry["tabs"] += not tab["gone"]
                entry["online"] = entry["online"] or not tab["gone"]
            visitors = sorted(people.values(), key=lambda entry: (not entry["online"], entry["idle_seconds"]))
            events = [dict(event) for event in self._events if event["seq"] > after]
            counters = dict(self._counters)
            counters["online"] = sum(entry["online"] for entry in visitors)
            snapshot: dict[str, Any] = {
                "monitor_started_at": _iso(self.started_at), "visitors": visitors, "events": events,
                "last_seq": self._sequence, "counters": counters,
                "blocks": [dict(entry) for entry in reversed(self._blocks.entries.values())]}
            group = self._keys.get(person) if isinstance(person, str) else None
            if group is not None:
                visitor = group.removeprefix("visitor:") if group.startswith("visitor:") else None
                snapshot["person"] = {
                    "key": person, "label": self._group_label(group),
                    "actions": [{"at": action["at"], "text": action["text"], "level": action["level"]}
                                for action in reversed(self._people[group]["actions"])],
                    "runs": [{"id": run_id, "query": run["query"], "mode": run["mode"], "state": run["state"],
                              "started_at": _iso(run["started"]),
                              "seconds": round((run["finished"] or now) - run["started"])}
                             for run_id, run in reversed(self._runs.items())
                             if visitor is not None and run["visitor"] == visitor
                             or visitor is None and group == "anonymous" and run["visitor"] is None]}
            return snapshot


class SilentMonitor:
    """Для служебных подделок сервиса в тестах, созданных без инициализатора."""

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_args, **_kwargs: None
