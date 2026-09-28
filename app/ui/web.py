"""Локальный веб-интерфейс. Запуск из корня: python -m streamlit run app/ui/web.py."""

from dataclasses import replace
from datetime import date, datetime, timedelta
import hashlib
from html import escape
import hmac
import ipaddress
import json
import os
import secrets
from pathlib import Path
import time
from urllib.parse import urlsplit

import streamlit as st

from app.input_safety import url_key, work_key
from app.ui.api_client import (AccessRefused, ApiClient, ApiError, AnalysisResult, AnalysisStatus, HISTORY_PAGE_SIZE,
                               HistoryEntry, MAX_PUBLICATION_OFFSET, MAX_QUERY_LENGTH, PUBLICATION_PAGE_SIZE,
                               Publication, Signal, SimilarActivity, Translation, Trend, VISITOR)
from app.ui.radar_client import RadarStatus
from app.ui.radar_view import find_report, radar_markup, report_markup

# Режим выбирает одна кнопка: не нажата — быстрый обзор, нажата — глубокий
# анализ с полной выборкой, самой длинной историей и патентами.
# Подписи переключателя режима; значки — Material Symbols из Streamlit.
MODE_LABELS = {"fast": ":material/bolt: Быстрый", "deep": ":material/travel_explore: Глубокий"}
ACTIVE_STATES = frozenset({"waiting", "queued", "running", "cancelling"})
OPENALEX_LIMIT_NOTE = (" OpenAlex отказал из-за лимита бесплатных запросов, поэтому история публикаций "
                       "недоступна и тренды не могли быть подтверждены. Бесплатный ключ OpenAlex снимает "
                       "это ограничение: добавьте его в настройках приложения.")
CONFIDENCE_LABELS = {"low": ("низкая", 1), "medium": ("средняя", 2), "high": ("высокая", 3)}
FEATURE_LABELS = {"growth": "рост", "persistence": "устойчивость", "novelty": "новизна",
                  "independence": "независимые группы", "application": "применение"}
CATEGORY_LABELS = {
    "confirmed_trend": "Подтверждённый тренд · по данным методики",
    "early_signal": "Ранний сигнал · гипотеза",
    "weak_signal_candidate": "Кандидат в слабые сигналы · автоматическая оценка",
    "emerging_candidate": "Зарождающийся кандидат · автоматическая оценка",
}

# Временная ссылка Cloudflare отвечает и по http://. Страница, открытая так не
# на этом компьютере, сразу уходит на https:// — пароль не набирается без шифрования.
HTTPS_ONLY = """<script>
(() => {
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  if (location.protocol === "http:" && !local) location.replace("https:" + location.href.slice(5));
})();
</script>"""

ENTER_TO_SUBMIT = """<script>
(() => {
  if (window.__trendEnterToSubmit) return;
  window.__trendEnterToSubmit = true;
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.altKey ||
        event.metaKey || event.repeat || event.isComposing || event.keyCode === 229 ||
        !(event.target instanceof HTMLTextAreaElement)) return;
    const form = event.target.closest('[data-testid="stForm"]');
    const submit = form?.querySelector('.st-key-start_analysis button:not(:disabled)');
    if (!submit) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    submit.click();
  }, true);
})();
</script>"""

# Часы панели хода анализа идут в браузере без запроса к серверу.
RUN_CLOCK = """<script>
(() => {
  if (window.__trendRunClock) return;
  window.__trendRunClock = true;
  const pad = (value) => String(value).padStart(2, "0");
  const anchors = new Map();
  // performance.now() keeps a steady cadence. On some systems it pauses during
  // sleep, so account for the missing wall time when the page resumes.
  let lastWall = Date.now(), lastMono = performance.now(), sleepOffset = 0;
  const steadyNow = () => {
    const wall = Date.now(), mono = performance.now();
    const wallAdvance = wall - lastWall;
    const missed = wallAdvance - (mono - lastMono);
    if (wallAdvance >= 1500 && missed > 1000) sleepOffset += missed;
    lastWall = wall;
    lastMono = mono;
    return mono + sleepOffset;
  };
  function paint() {
    const now = steadyNow();
    const active = new Set();
    for (const clock of document.querySelectorAll("[data-ta-clock]")) {
      const key = clock.dataset.taClock;
      const reported = Number(clock.dataset.taElapsed);
      if (!key || !Number.isFinite(reported) || reported < 0) continue;
      active.add(key);
      let anchor = anchors.get(key);
      if (!anchor) {
        anchor = {base: reported, since: now, reported};
      } else if (anchor.reported !== reported) {
        const running = anchor.base + (now - anchor.since) / 1000;
        // A normal server refresh must not restart the second's fractional part.
        // A running clock cannot go backwards if a poll reports older seconds.
        if (reported > running + 1) {
          anchor = {base: reported, since: now, reported};
        } else {
          anchor.reported = reported;
        }
      }
      anchors.set(key, anchor);
      const seconds = Math.max(reported, Math.floor(anchor.base + (now - anchor.since) / 1000));
      const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds / 60) % 60;
      const label = (hours ? hours + ":" + pad(minutes) : pad(minutes)) + ":" + pad(seconds % 60);
      if (clock.textContent !== label) clock.textContent = label;
    }
    for (const key of anchors.keys()) if (!active.has(key)) anchors.delete(key);
  }
  let timer;
  function pump() {
    if (!document.hidden) paint();
    timer = setTimeout(pump, document.hidden ? 1000 : 200);
  }
  function resume() {
    clearTimeout(timer);
    pump();
  }
  document.addEventListener("visibilitychange", resume);
  window.addEventListener("pageshow", resume);
  pump();
})();
</script>"""

# «Основание оценки» раскрывается в потоке, поэтому растягивает строку сетки.
# Чтобы у соседей не оставалось пустого поля, раскрытие синхронно в ряду:
# открытие у одной плитки открывает его у плиток на той же высоте. toggle не
# всплывает, поэтому слушатель стоит на фазе перехвата; сравнение состояний
# обрывает цепочку ответных событий.
ROW_DETAILS = """<script>
(() => {
  if (window.__trendRowDetails) return;
  window.__trendRowDetails = true;
  document.addEventListener("toggle", (event) => {
    const details = event.target;
    if (!(details instanceof HTMLDetailsElement) || !details.matches(".ta-model-details")) return;
    const card = details.closest(".ta-card");
    if (!card || !card.parentElement) return;
    for (const other of card.parentElement.children) {
      if (other === card || other.offsetTop !== card.offsetTop) continue;
      const twin = other.querySelector(".ta-model-details");
      if (twin && twin.open !== details.open) twin.open = details.open;
    }
  }, true);
})();
</script>"""

# Встроенные глифы одной сетки 16×16 и одной толщины штриха: внешних файлов и
# шрифтовых иконок нет, поэтому разметка не зависит от доступности сети.
MARK_ICON = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
             'focusable="false"><path d="M3 17l5-5 4 4 8-8"/><path d="M15 8h5v5"/></svg>')
LINK_ICON = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
             'focusable="false"><path d="M14 4h6v6"/><path d="M20 4l-9 9"/>'
             '<path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>')

# Фоновый граф: узлы — материалы, рёбра — связи между ними. Рисунок
# постоянный и не отражает никаких реальных данных.
GRAPH = ('<svg class="ta-graph" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false"><line x1="118" y1="168" x2="232" y2="286"/><line x1="232" y1="286" x2="96" y2="402"/><line x1="232" y1="286" x2="268" y2="452"/><line x1="96" y1="402" x2="178" y2="606"/><line x1="268" y1="452" x2="178" y2="606"/><line x1="178" y1="606" x2="318" y2="716"/><line x1="268" y1="452" x2="498" y2="618"/><line x1="118" y1="168" x2="404" y2="128"/><line x1="404" y1="128" x2="520" y2="262"/><line x1="520" y1="262" x2="612" y2="158"/><line x1="612" y1="158" x2="824" y2="214"/><line x1="520" y1="262" x2="706" y2="452"/><line x1="498" y1="618" x2="628" y2="742"/><line x1="498" y1="618" x2="742" y2="586"/><line x1="628" y1="742" x2="742" y2="586"/><line x1="742" y1="586" x2="866" y2="706"/><line x1="742" y1="586" x2="706" y2="452"/><line x1="706" y1="452" x2="958" y2="452"/><line x1="824" y1="214" x2="962" y2="122"/><line x1="824" y1="214" x2="1044" y2="268"/><line x1="1044" y1="268" x2="1176" y2="372"/><line x1="1044" y1="268" x2="1322" y2="196"/><line x1="1176" y1="372" x2="1238" y2="546"/><line x1="1238" y1="546" x2="1104" y2="662"/><line x1="1238" y1="546" x2="1346" y2="712"/><line x1="1104" y1="662" x2="866" y2="706"/><line x1="958" y1="452" x2="1238" y2="546"/><line x1="958" y1="452" x2="1044" y2="268"/><line x1="318" y1="716" x2="628" y2="742"/><line x1="1176" y1="372" x2="958" y2="452"/><circle cx="118" cy="168" r="3.5" class="ta-node-live"/><circle cx="232" cy="286" r="2.5"/><circle cx="96" cy="402" r="2"/><circle cx="268" cy="452" r="4" class="ta-node-live"/><circle cx="178" cy="606" r="2.5"/><circle cx="318" cy="716" r="2"/><circle cx="404" cy="128" r="2.5"/><circle cx="520" cy="262" r="2"/><circle cx="612" cy="158" r="3" class="ta-node-live"/><circle cx="498" cy="618" r="3"/><circle cx="628" cy="742" r="2.5" class="ta-node-live"/><circle cx="742" cy="586" r="4"/><circle cx="866" cy="706" r="2.5"/><circle cx="706" cy="452" r="2"/><circle cx="824" cy="214" r="2.5"/><circle cx="962" cy="122" r="2"/><circle cx="1044" cy="268" r="3.5" class="ta-node-live"/><circle cx="1176" cy="372" r="2.5"/><circle cx="1322" cy="196" r="2"/><circle cx="1238" cy="546" r="4" class="ta-node-live"/><circle cx="1104" cy="662" r="2.5"/><circle cx="1346" cy="712" r="2"/><circle cx="958" cy="452" r="2.5"/><circle class="ta-spark" cx="232" cy="286" r="3" style="--ta-dx: 36px; --ta-dy: 166px"/><circle class="ta-spark ta-spark-b" cx="742" cy="586" r="3" style="--ta-dx: -36px; --ta-dy: -134px"/><circle class="ta-spark ta-spark-c" cx="1044" cy="268" r="3" style="--ta-dx: 132px; --ta-dy: 104px"/></svg>')


@st.cache_resource(show_spinner=False, max_entries=256)
def get_client(api_url: str, visitor: str | None = None) -> ApiClient:
    # Здесь кэшируется только клиент. Модель принадлежит отдельному API-процессу.
    return ApiClient(api_url, visitor)


def api() -> ApiClient:
    """Клиент API этой страницы; с паролем — от имени её посетителя."""
    return get_client(os.environ.get("API_URL", "http://127.0.0.1:8000"), st.session_state.get("visitor"))


# Панель владельца видит открытые вкладки: страница отмечается при загрузке и
# раз в 15 секунд, пока вкладка открыта.
PRESENCE_SECONDS = 15
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def forwarded_address(*values: str | None) -> str | None:
    """Адрес человека из цепочки прокси: справа налево до первого внешнего."""
    chain = []
    for value in values:
        for part in (value or "").split(","):
            try:
                chain.append(ipaddress.ip_address(part.strip()))
            except ValueError:
                continue
    if not chain:
        return None
    return str(next((address for address in reversed(chain) if address.is_global), chain[0]))


def client_address() -> str | None:
    """Адрес посетителя для панели владельца и её блокировок.

    За туннелем соединение приходит с этого же компьютера. Сервер туннеля пишет
    адрес человека в X-Forwarded-For, а клиент туннеля дописывает после него
    свой (127.0.0.1), поэтому цепочка читается справа налево до первого внешнего
    адреса: подставной адрес, присланный самим посетителем, стоит левее.
    """
    headers = st.context.headers
    forwarded = forwarded_address(headers.get("X-Forwarded-For"), headers.get("X-Real-Ip"))
    if forwarded is not None:
        return forwarded
    address = st.context.ip_address
    if address is None and (headers.get("Host") or "").rsplit(":", 1)[0].lower() in LOCAL_HOSTS:
        return "127.0.0.1"
    return address


def check_access() -> str | None:
    """Закрыл ли владелец вход этому браузеру или адресу: `blocked`, `signed_out` или ничего."""
    try:
        return api().check_access(client_address())
    except Exception:
        return None  # Панель владельца не должна ронять сайт.


def sign_out():
    """Владелец завершил сеанс: забыть вход и не восстанавливать его из cookie."""
    for key in ("authenticated", "visitor", "visitor_cookie"):
        st.session_state.pop(key, None)
    st.session_state.cookie_revoked = True
    st.session_state.forget_cookie = True


def render_blocked():
    st.markdown('<section class="ta-login"><div class="ta-login-mark">T</div><h1>Доступ закрыт</h1>'
                '<p>Владелец сервиса закрыл доступ с этого браузера или адреса.</p></section>',
                unsafe_allow_html=True)


def report_presence(page: str, event: str | None = None, *, view: str | None = None,
                    detail: str | None = None) -> dict:
    session = st.session_state.setdefault("presence_id", secrets.token_urlsafe(12))
    try:
        return api().report_presence(session, page, event=event, agent=st.context.headers.get("User-Agent"),
                                     ip=client_address(), view=view, detail=detail)
    except Exception:
        # Отметка нужна только панели владельца; страница от неё не зависит.
        return {"access": None, "messages": []}


@st.fragment(run_every=f"{PRESENCE_SECONDS}s")
def presence_heartbeat(page: str, view: str | None, detail: str | None):
    answer = report_presence(page, view=view, detail=detail)
    if answer["messages"]:
        st.session_state.setdefault("owner_messages", []).extend(answer["messages"])
        st.session_state.owner_messages_new = True
    if answer["access"] == "signed_out":
        sign_out()
    # При загрузке страницы сообщение покажет её же проход; между загрузками —
    # перерисовка страницы, как и при закрытом владельцем входе.
    if answer["access"] or answer["messages"] and not st.session_state.get("presence_inline"):
        st.rerun(scope="app")


def track_presence(page: str, *, view: str | None = None, detail: str | None = None):
    # Контейнер скрыт стилями: фрагмент ничего не рисует, только отмечается.
    st.session_state.presence_inline = True
    try:
        with st.container(key="ta-presence"):
            presence_heartbeat(page, view, detail)
    finally:
        st.session_state.presence_inline = False


def render_owner_messages():
    """Сообщения владельца сервиса этому посетителю — пока их не закроют."""
    messages = st.session_state.get("owner_messages") or []
    if not messages:
        return
    if st.session_state.pop("owner_messages_new", False):
        st.toast("Сообщение от владельца сервиса", icon=":material/mail:")
    with st.container(key="ta-owner-note"):
        st.markdown('<div class="ta-owner-note" role="status"><b>Сообщение от владельца сервиса</b>'
                    + "".join(f"<p>{escape(text)}</p>" for text in messages[-3:]) + "</div>",
                    unsafe_allow_html=True)
        if st.button("Понятно", key="owner_note_close", type="tertiary"):
            st.session_state.owner_messages = []
            st.rerun()


def page_view(status: AnalysisStatus, busy: bool) -> str:
    """Что видит посетитель на главной — для панели владельца."""
    if st.session_state.get("history_open"):
        return "history"
    if status.state == "waiting":
        return "waiting"
    if busy:
        return "running"
    if st.session_state.get("error") or status.state in {"failed", "interrupted"}:
        return "error"
    if status.service_busy:
        return "busy"
    if status.state == "succeeded":
        return "result"
    return "paused" if status.paused else "idle"


def queue_analysis():
    query = st.session_state.query.strip()
    st.session_state.pop("error", None)
    st.session_state.pop("error_run_id", None)
    if not query:
        st.session_state.error = "Введите тему, которую хотите исследовать."
        return
    try:
        client = api()
        mode = "deep" if st.session_state.get("analysis_mode") == "deep" else "fast"
        st.session_state.analysis_status = client.start_analysis(query, mode)
    except ApiError as error:
        st.session_state.error = str(error)
    except Exception:
        st.session_state.error = "Не удалось начать анализ. Попробуйте перезапустить интерфейс."


def cancel_analysis(run_id: str):
    try:
        client = api()
        st.session_state.analysis_status = client.cancel_analysis(run_id)
        st.session_state.pop("error", None)
        st.session_state.pop("error_run_id", None)
    except ApiError as error:
        st.session_state.error = str(error)
        st.session_state.error_run_id = run_id
    except Exception:
        st.session_state.error = "Не удалось отменить анализ. Попробуйте ещё раз."
        st.session_state.error_run_id = run_id


THEMES = {"Светлая": "light", "Тёмная": "dark"}
# Вход посетителя переживает переходы по ссылкам (отчёты, анализы из истории):
# браузер хранит его случайный идентификатор, подписанный паролем запуска.
# Новый запуск — новый пароль, и старые подписи перестают подходить.
VISITOR_COOKIE = "ta_visitor"


def visitor_pass(visitor: str, password: str) -> str:
    key = hashlib.sha256(b"trendanalizer-visitor\0" + password.encode("utf-8")).digest()
    return visitor + "." + hmac.new(key, visitor.encode("ascii"), "sha256").hexdigest()


def visitor_from_pass(value: object, password: str) -> str | None:
    if not isinstance(value, str) or len(value) > 200:
        return None
    visitor = value.partition(".")[0]
    if VISITOR.fullmatch(visitor) is None:
        return None
    return visitor if hmac.compare_digest(visitor_pass(visitor, password), value) else None


FORGET_VISITOR = (f'<script>document.cookie = "{VISITOR_COOKIE}=; path=/; max-age=0; SameSite=Strict"'
                  + '+ (location.protocol === "https:" ? "; Secure" : "");</script>')


def remember_visitor(value: str) -> str:
    """Скрипт, кладущий подписанный вход в cookie браузера на полсуток."""
    return ('<script>(() => { const secure = location.protocol === "https:" ? "; Secure" : ""; '
            f'document.cookie = "{VISITOR_COOKIE}=" + {json.dumps(value)} + '
            '"; path=/; max-age=43200; SameSite=Strict" + secure; })();</script>')


def authenticated() -> bool:
    """Keep public demo sessions behind an ephemeral server-side password."""
    mode = os.environ.get("TREND_WEB_REQUIRE_AUTH")
    if mode == "0":
        return True
    if mode != "1":
        st.error("Режим доступа к вебу не задан. Запустите его штатным файлом или явно "
                 "укажите TREND_WEB_REQUIRE_AUTH=0 только для локальной разработки.")
        return False
    expected = os.environ.get("TREND_WEB_ACCESS_PASSWORD", "")
    if len(expected) < 16:
        st.error("Веб-демо запущено без безопасного пароля. Остановите его и запустите штатным файлом.")
        return False
    if st.session_state.get("authenticated") is True:
        return True
    # После завершения сеанса владельцем прежняя cookie не пускает: нужен пароль.
    visitor = (None if st.session_state.get("cookie_revoked")
               else visitor_from_pass(st.context.cookies.get(VISITOR_COOKIE), expected))
    if visitor is not None:
        st.session_state.authenticated = True
        st.session_state.visitor = visitor
        return True
    st.markdown('<section class="ta-login"><div class="ta-login-mark">T</div>'
                '<h1>Trendanalyser</h1><p>Закрытая демонстрация</p>'
                '<p class="ta-login-note">Анализ выполняется на компьютере владельца сервиса. '
                'Ваши запросы и результаты видны только в этом браузере: другие посетители их '
                'не видят. Соединение зашифровано.</p></section>',
                unsafe_allow_html=True)
    with st.form("login"):
        entered = st.text_input("Пароль доступа", type="password", autocomplete="current-password",
                                max_chars=128)
        submitted = st.form_submit_button("Войти", type="primary", use_container_width=True)
    if submitted:
        if hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8")):
            visitor = secrets.token_urlsafe(18)
            st.session_state.authenticated = True
            st.session_state.visitor = visitor
            st.session_state.visitor_cookie = visitor_pass(visitor, expected)
            st.session_state.pop("cookie_revoked", None)
            report_presence("login", "login")
            st.rerun()
        else:
            report_presence("login", "login_failed")
            st.error("Неверный пароль.")
    return False


def render_theme():
    """Переключатель темы: только светлая и тёмная, без режима «по системе»."""
    with st.container(key="ta-theme"):
        choice = st.segmented_control("Тема", list(THEMES), default="Светлая",
                                      key="theme", label_visibility="collapsed")
    # Маркер невидим; палитру поднимает до :root селектор :has() в CSS.
    st.markdown(f'<div class="ta-theme-{THEMES.get(choice or "Светлая")}" hidden></div>',
                unsafe_allow_html=True)


MODE_NAMES = {"fast": "Быстрый", "deep": "Глубокий"}
HISTORY_STATES = {"succeeded": "Готов", "running": "Выполняется", "queued": "В очереди",
                  "failed": "Ошибка", "cancelled": "Отменён", "interrupted": "Прерван"}
# Родительный падеж: «26 сен», но «26 мая».
DATE_MONTHS = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
CHEVRON_ICON = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
                'focusable="false"><path d="M9 6l6 6-6 6"/></svg>')


def history_date(moment: datetime, today: date | None = None) -> str:
    """Время запуска по часам сервера: недавние — словами, остальные — датой."""
    local = moment.astimezone()
    today = today or datetime.now().astimezone().date()
    clock = f"{local.hour:02d}:{local.minute:02d}"
    if local.date() == today:
        return f"сегодня, {clock}"
    if local.date() == today - timedelta(days=1):
        return f"вчера, {clock}"
    year = "" if local.year == today.year else f" {local.year}"
    return f"{local.day} {DATE_MONTHS[local.month - 1]}{year}, {clock}"


def history_item_markup(entry: HistoryEntry, *, current: bool, openable: bool,
                        today: date | None = None) -> str:
    """Строка истории: запрос крупно, под ним время, режим и состояние."""
    meta = [escape(history_date(entry.created_at, today))]
    if entry.mode is not None:
        meta.append(MODE_NAMES[entry.mode])
    if current:
        meta.append('<span class="ta-history-state ta-history-current">Открыт сейчас</span>')
    elif entry.state != "succeeded":
        meta.append(f'<span class="ta-history-state">{HISTORY_STATES[entry.state]}</span>')
    classes = "ta-history-item" + ("" if openable or current else " ta-history-idle")
    mark = f'<span class="ta-history-go">{CHEVRON_ICON}</span>' if openable else ""
    return (f'<div class="{classes}"><div class="ta-history-text">'
            f'<span class="ta-history-query">{escape(entry.query)}</span>'
            f'<span class="ta-history-meta">{" · ".join(meta)}</span></div>{mark}</div>')


def open_history():
    st.session_state.history_open = True
    st.session_state.history_offset = 0


def close_history():
    st.session_state.history_open = False


def shift_history(step: int):
    st.session_state.history_offset = max(0, st.session_state.get("history_offset", 0) + step)


def render_history_button():
    """Кнопка истории в левом верхнем углу, напротив переключателя темы."""
    with st.container(key="ta-history"):
        st.button("История", key="history_toggle", icon=":material/history:", on_click=open_history)


def leave_saved():
    """С сохранённого анализа — на главную страницу текущего."""
    st.query_params.pop("run", None)
    st.query_params.pop("report", None)
    # Форма снова возьмёт запрос и режим текущего анализа.
    st.session_state.pop("viewed_analysis_id", None)
    st.session_state.pop("publication_pages", None)


@st.dialog("История анализов", width="medium", on_dismiss=close_history)
def render_history():
    """Сохранённые анализы профиля; готовый открывается своей страницей `?run=`.

    Такая страница ничего не меняет у сервиса: её можно открыть, пока идёт
    новый анализ, обновить и отправить ссылкой. Флаг `history_open` держит окно
    открытым, когда страницу перерисовывает опрос перевода или хода анализа.
    """
    client = api()
    offset = st.session_state.get("history_offset", 0)
    try:
        page = client.history(offset)
    except ApiError as error:
        st.error(str(error))
        return
    if not page.entries:
        st.markdown('<p class="ta-history-note">'
                    + ("Сохранённых анализов пока нет: запущенный анализ появится здесь."
                       if offset == 0 else "Здесь анализов больше нет.") + '</p>', unsafe_allow_html=True)
    shown = st.query_params.get("run") or page.current_id
    for entry in page.entries:
        current = entry.id == shown
        # Текущий анализ живёт на главной странице: там виден его ход и отмена.
        openable = not current and (entry.state == "succeeded" or entry.id == page.current_id)
        with st.container(key=f"ta-hrow-{entry.id}"):
            st.markdown(history_item_markup(entry, current=current, openable=openable),
                        unsafe_allow_html=True)
            # Кнопка растянута на всю строку (CSS); подпись — её доступное имя.
            if openable and st.button(f"Открыть анализ «{entry.query[:80]}»", key=f"ta-hopen-{entry.id}",
                                      width="stretch"):
                st.session_state.history_open = False
                if entry.id == page.current_id:
                    leave_saved()
                else:
                    st.query_params.pop("report", None)
                    st.query_params["run"] = entry.id
                    st.session_state.pop("publication_pages", None)
                    st.session_state.history_opening = entry.query
                st.rerun()
    if offset or page.has_more:
        with st.container(key="ta-history-pager", horizontal=True, horizontal_alignment="distribute"):
            st.button("Новее", key="history_newer", icon=":material/chevron_left:", type="tertiary",
                      disabled=not offset, on_click=shift_history, args=(-HISTORY_PAGE_SIZE,))
            st.button("Старее", key="history_older", icon=":material/chevron_right:", icon_position="right",
                      type="tertiary", disabled=not page.has_more,
                      on_click=shift_history, args=(HISTORY_PAGE_SIZE,))


def render_about():
    """Справка под строкой ввода: что это, как читать результат и правила."""
    st.markdown(f'<section class="ta-about">'
                f'<div class="ta-about-brand"><span class="ta-mark">{MARK_ICON}</span>'
                '<b>TEMA</b></div>'
                '<p class="ta-about-lead">Поиск материалов и зарождающихся технологий '
                'по произвольному направлению, локально.</p>'
                '<div class="ta-about-cols">'
                # В обеих колонках по четыре пункта сопоставимой длины: иначе
                # столбцы получаются разной высоты и ряд выглядит косым.
                '<div><h2>Как читать результат</h2>'
                '<ul><li>ТОП-15 технологий: раскройте карточку, чтобы посмотреть '
                'источники именно по этой технологии.</li>'
                '<li>Уверенность модели показана у технологии. Балл публикации, если он есть, '
                'показывает соответствие запросу, а не вероятность достоверности.</li></ul></div>'
                '<div><h2>Правила работы</h2>'
                '<ul><li>Один запрос за раз. Повторов нет.</li>'
                '<li>Анализ занимает несколько минут; вкладку можно обновить.</li>'
                '<li>Текущий анализ можно отменить. Готовые результаты сохраняются: '
                'прошлые анализы открываются из «Истории».</li></ul></div>'
                '</div></section>', unsafe_allow_html=True)


def render_head():
    st.markdown(GRAPH +
                '<div class="ta-hero"><h1>Какое направление изучаем?</h1></div>', unsafe_allow_html=True)


# Ход анализа крупными шагами. Этапы пайплайна сведены к пяти понятным
# человеку фазам; порядок — порядок работы сервиса. Внутри активной фазы
# виден реальный прогресс её этапа, общий процент не выдумывается.
RUN_PHASES = (
    ("Понимаем запрос", "Локальная модель очерчивает границы направления.", frozenset({"plan"})),
    ("Собираем публикации", "Crossref и открытые площадки: статьи, препринты, репозитории.",
     frozenset({"discovery"})),
    ("Выделяем темы", "Публикации группируются по смыслу, у групп появляются названия.",
     frozenset({"relevance", "labels"})),
    ("Проверяем рост и доказательства", "Годовая статистика, ранние упоминания, проверка карточек.",
     frozenset({"history", "evidence", "antecedents", "enrichment", "verify"})),
    ("Собираем ТОП", "ТОП-15 технологий и оценка публикаций моделью.",
     frozenset({"external_sources", "publish", "reconcile", "radar", "complete"})),
)
# Короткие факты о методике на время ожидания; сменяются по кругу.
FACT_SECONDS = 8
RUN_FACTS = (
    "Модели анализа работают на этом компьютере; в сеть уходят только поисковые запросы к источникам.",
    "Темы-кандидаты выделяются по смыслу: модель сравнивает запрос с каждой найденной публикацией.",
    "Название темы проверяется: фраза должна встречаться в заголовках её публикаций.",
    "Рост считается по годовым подсчётам публикаций, а не по порядку в поисковой выдаче.",
    "Уверенность в тренде складывается из пяти признаков: рост, устойчивость, новизна, "
    "независимые группы и применение.",
    "ТОП-15 технологий считается параллельно с проверкой тем, по той же собранной выборке.",
    "Когда анализ закончится, ТОП переведётся на русский; оригинал останется под рукой.",
)
CHECK_ICON = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
              'focusable="false"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>')


SOURCE_LABELS = {
    "openalex": "OpenAlex", "crossref": "Crossref", "europe_pmc": "Europe PMC",
    "epo": "EPO", "report": "Отчёты",
    "arxiv": "arXiv", "biorxiv": "bioRxiv", "openreview": "OpenReview",
    "zenodo": "Zenodo", "nist_news": "NIST News",
    "mit_research_news": "MIT Research News", "horizon_magazine": "Horizon Magazine",
    "github": "GitHub", "hacker_news": "Hacker News", "habr": "Хабр", "gdelt": "GDELT",
    "google_news": "Google News", "semantic_scholar": "Semantic Scholar", "doaj": "DOAJ",
    "cyberleninka": "КиберЛенинка", "hal": "HAL", "osti": "OSTI", "nasa_ntrs": "NASA NTRS", "dblp": "dblp",
    "stack_exchange": "Stack Overflow", "huggingface": "Hugging Face", "chemrxiv": "ChemRxiv",
    "openaire": "OpenAIRE", "jstage": "J-STAGE", "npm": "npm",
}
SOURCE_STATES = {"complete": "Запрос обработан", "partial": "Частичный сбор",
                 "unavailable": "Источник недоступен"}
MATERIAL_TYPES = {
    "publication": "Публикация", "article": "Статья", "journal-article": "Научная статья",
    "journal_article": "Журнальная запись",
    "proceedings-article": "Статья конференции", "book-chapter": "Глава книги",
    "preprint": "Препринт", "research_artifact": "Исследовательский материал",
    "institution_news": "Новость организации", "news_aggregate": "Новость",
    "repository": "Репозиторий", "community": "Обсуждение", "report": "Отчёт",
    "posted-content": "Препринт", "book": "Книга", "monograph": "Монография",
    "edited-book": "Сборник", "reference-entry": "Справочная статья", "dissertation": "Диссертация",
}


def publication_date(publication: Publication) -> str:
    if publication.published_at:
        prefix = {"indexed": "Обнаружено", "created": "Создано",
                  "published": "Опубликовано", "unknown": "Дата"}[publication.date_basis]
        return f"{prefix} {publication.published_at.isoformat()}"
    if publication.publication_year:
        return f"{publication.publication_year} г."
    return "Дата не указана"


def link_label(url: str) -> str:
    """Подпись дополнительной ссылки: у DOI — сам идентификатор, иначе несколько
    ссылок подряд читались бы одинаково «doi.org»."""
    parsed = urlsplit(url)
    host = (parsed.hostname or url).lower().removeprefix("www.")
    if host in {"doi.org", "dx.doi.org"} and parsed.path.strip("/"):
        return "DOI " + parsed.path.strip("/")
    return host


class ReadingTexts(dict[str, str]):
    """Перевод текстов ТОПа вместе с кодом его языка (для атрибута lang)."""

    def __init__(self, texts: dict[str, str], language: str = "ru"):
        super().__init__(texts)
        self.language = language


def text_markup(text: str, ru: dict[str, str]) -> str:
    """Текст карточки: машинный перевод, если он есть, иначе оригинал."""
    russian = ru.get(text)
    language = getattr(ru, "language", "ru")
    return f'<span lang="{escape(language)}">{escape(russian)}</span>' if russian else escape(text)


def signal_markup(signal: Signal, *, named: bool = True, ru: dict[str, str] | None = None) -> str:
    """Плашка сигнала анализа прямо на карточке ТОПа: технология и категория.

    У сигнала без публикации в выдаче его название уже стоит заголовком карточки.
    """
    name = f'<strong>{text_markup(signal.title, ru or {})}</strong>' if named else ""
    return (f'<div class="ta-signal"><div class="ta-signal-head">'
            f'<span class="ta-signal-label">Сигнал анализа</span>{name}</div>'
            f'<span class="ta-category">{escape(CATEGORY_LABELS[signal.category])}</span></div>')


def model_confidence_markup(publication: Publication, *, top: bool) -> str:
    """Оценка моделью конкретной публикации; балл не является вероятностью."""
    assessment = publication.model_confidence
    if assessment is None:
        if not top:
            return ""
        return ('<div class="ta-model-confidence ta-model-confidence-none"><div class="ta-meter">'
                '<span class="ta-meter-label">Соответствие запросу</span>'
                '<b class="ta-meter-value">не оценено</b></div></div>')
    basis = ("заголовок и начало описания" if assessment.basis == "title_and_summary"
             else "только заголовок")
    # Одна строка на показатель: подпись, короткая полоска, значение.
    return (f'<div class="ta-model-confidence"><div class="ta-meter">'
            f'<span class="ta-meter-label">Соответствие запросу</span>'
            f'<span class="ta-scale" style="--ta-value: {assessment.score}%" aria-hidden="true">'
            f'<i></i></span>'
            f'<b class="ta-meter-value">{assessment.score}<small>/100</small></b></div>'
            f'<details class="ta-model-details"><summary>Основание оценки</summary>'
            f'<ul class="ta-model-criteria">'
            f'<li>Оценка модели E5: лучший результат с исходным или английским запросом.</li>'
            f'<li>Текст: {basis}.</li>'
            f'<li>Шкала: 0,75 → 0; 0,90 → 100. Дата, тип источника и тренд '
            f'не учитываются.</li>'
            f'</ul></details></div>')


MONTH_NAMES = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")


def month_label(month: str) -> str:
    return f"{MONTH_NAMES[int(month[5:7]) - 1]} {month[:4]}"


def similar_markup(activity: SimilarActivity | None) -> str:
    """Сколько похожих материалов собранной выборки выходило по месяцам."""
    if activity is None:
        return ""
    head = ('<div class="ta-meter"><span class="ta-meter-label">Похожие в выборке</span>'
            f'<b class="ta-meter-value">{activity.total}<small> за {len(activity.months)} мес.</small></b></div>')
    peak = max(1, max(row.count for row in activity.months))
    step, gap, height = 20, 4, 40
    bars = []
    for index, row in enumerate(activity.months):
        # У пустого месяца остаётся риска, чтобы было видно, что месяц учтён.
        bar = max(2.0, height * row.count / peak)
        tone = "" if row.count else ' class="ta-similar-empty"'
        hint = (f"{row.count} из {row.collected} собранных" if row.collected
                else "материалов за месяц не собрано")
        bars.append(f'<rect{tone} x="{index * step}" y="{height - bar:.1f}" width="{step - gap}" '
                    f'height="{bar:.1f}" rx="1.5"><title>{month_label(row.month)}: {hint}</title></rect>')
    spoken = ", ".join(f"{month_label(row.month)} — {row.count}" for row in activity.months if row.count)
    first, last = activity.months[0].month, activity.months[-1].month
    return (f'<div class="ta-similar">{head}'
            f'<svg class="ta-similar-chart" viewBox="0 0 {len(activity.months) * step - gap} {height}" '
            f'preserveAspectRatio="none" role="img" '
            f'aria-label="{escape(f"Похожие материалы по месяцам: {spoken}", quote=True)}">'
            f'{"".join(bars)}</svg>'
            f'<div class="ta-similar-axis"><span>{month_label(first)}</span>'
            f'<span>пик {peak}</span><span>{month_label(last)}</span></div></div>')


def trend_markup(publication: Publication | None, signal: Signal | None, *,
                 ru: dict[str, str] | None = None) -> str:
    """Отдельная оценка методики для тренда, а не для конкретной публикации."""
    if signal is not None and signal.confidence is not None:
        return f'<div class="ta-card-trend">{confidence_markup(signal)}</div>'
    trend = publication.trend if publication is not None else None
    if trend is not None:
        return (f'<div class="ta-card-trend">{confidence_markup(trend)}'
                f'<p class="ta-trend-topic">Тема: <b>{text_markup(trend.title, ru or {})}</b></p></div>')
    return ""


def top_card(publication: Publication | None, signal: Signal | None, number: int | str, *,
             top: bool = True, ru: dict[str, str] | None = None) -> str:
    """Плитка выдачи: публикация, сигнал на её основе или сигнал без публикации в выдаче.

    Плитка целиком — ссылка на материал: заголовок растянут на всю карточку.
    Дополнительные ссылки сигнала лежат поверх и открываются сами по себе.
    `ru` — машинный перевод текстов ТОПа; у переведённого заголовка оригинал
    остаётся во всплывающей подсказке.
    """
    ru = ru or {}
    if publication is not None:
        url, heading = publication.url, publication.title
        label = SOURCE_LABELS.get(publication.source_id, publication.source_id)
        material_type = MATERIAL_TYPES.get(publication.kind, publication.kind)
        # Дата не рвётся посередине, когда строка переносится.
        meta = (f'{escape(label)} · {escape(material_type)} · '
                f'<span class="ta-date">{escape(publication_date(publication))}</span>')
        summary = publication.summary or (signal.summary if signal else None)
    else:
        assert signal is not None
        url, heading = signal.source_urls[0], signal.title
        meta = escape(urlsplit(url).hostname or url)
        summary = signal.summary
    links = []
    if signal is not None:
        shown = {url_key(url)}
        for source in signal.source_urls:
            if url_key(source) in shown:
                continue
            shown.add(url_key(source))
            links.append(f'<a href="{escape(source, quote=True)}" target="_blank" '
                         f'rel="noopener noreferrer">{LINK_ICON}<span>{escape(link_label(source))}</span></a>')
    signal_block = (signal_markup(signal, named=publication is not None, ru=ru)
                    if signal is not None else "")
    body = f'<p class="ta-card-summary">{text_markup(summary, ru)}</p>' if summary else ""
    original = f' title="{escape(heading, quote=True)}"' if heading in ru else ""
    extra = f'<div class="ta-links">{"".join(links)}</div>' if links else ""
    classes = "ta-card ta-publication-card" + (" ta-card-signal" if signal is not None else "")
    model_block = model_confidence_markup(publication, top=top) if publication is not None else ""
    similar_block = similar_markup(publication.similar) if publication is not None else ""
    trend_block = trend_markup(publication, signal, ru=ru)
    assessment = (f'<div class="ta-card-assessment">{model_block}{similar_block}{trend_block}</div>'
                  if model_block or similar_block or trend_block else "")
    return (f'<article class="{classes}"><div class="ta-card-top">'
            f'<span class="ta-index">{escape(str(number))}</span>'
            f'<span class="ta-publication-meta">{meta}</span>{LINK_ICON}</div>'
            f'<h3><a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer"{original}>'
            f'{text_markup(heading, ru)}</a></h3>{signal_block}{body}'
            f'{assessment}{extra}</article>')


def publication_card(publication: Publication, number: int) -> str:
    return top_card(publication, None, number, top=False)


def top_entries(result: AnalysisResult, *, initial_count: int | None = None,
                page_offset: int | None = None) -> tuple[list[tuple[Publication, Signal | None]],
                                                         list[tuple[Publication | None, Signal, int | str]],
                                                         list[tuple[Publication, int]]]:
    """Ровно серверный ТОП публикаций; остальные сигналы доступны отдельным разделом."""
    publications = result.publications
    top_publications = result.top_publications or publications[:15]
    top_ids = {publication.publication_id for publication in top_publications}
    by_key: dict[str, Publication] = {}
    for publication in publications:
        by_key.setdefault(url_key(publication.url), publication)
    ranks = {publication.publication_id: index for index, publication in enumerate(publications, start=1)}
    if initial_count is not None and page_offset is not None:
        ranks.update((publication.publication_id, page_offset + index)
                     for index, publication in enumerate(publications[initial_count:], start=1))
    attached: dict[str, Signal] = {}
    other_signals: list[tuple[Publication | None, Signal, int | str]] = []
    used_elsewhere: set[str] = set()
    for index, signal in enumerate(result.signals, start=1):
        matches = [by_key[key] for key in map(url_key, signal.source_urls)
                   if key in by_key and by_key[key].publication_id not in attached
                   and by_key[key].publication_id not in used_elsewhere]
        match = next((item for item in matches if item.publication_id in top_ids),
                     matches[0] if matches else None)
        if match is not None and match.publication_id in top_ids:
            attached[match.publication_id] = signal
        else:
            if match is not None:
                used_elsewhere.add(match.publication_id)
            other_signals.append((match, signal,
                                  ranks[match.publication_id] if match is not None else f"С{index}"))
    entries = [(publication, attached.get(publication.publication_id))
               for publication in top_publications]
    remaining = [(publication, ranks[publication.publication_id]) for publication in publications
                 if publication.publication_id not in top_ids
                 and publication.publication_id not in used_elsewhere]
    return entries, other_signals, remaining


TOP_LANGUAGES = ("Русский", "Оригинал")


def reading_language() -> dict[str, str]:
    """Язык перевода ТОПа, который посетитель выбрал сам; пусто — язык сайта."""
    code = st.session_state.get("reading_language")
    return {"language": code} if isinstance(code, str) and code else {}


def page_status(run_id: str) -> AnalysisStatus:
    """Состояние анализа, который показывает страница: из истории (`?run=`) или текущего."""
    client = api()
    if st.query_params.get("run") == run_id:
        return client.saved_analysis(run_id, **reading_language())
    return client.current_analysis(**reading_language())


@st.fragment(run_every="2s")
def render_translation_progress(run_id: str, shown: int):
    """Ход перевода ТОПа; страница перерисовывается, когда приходит новая порция."""
    try:
        status = page_status(run_id)
    except ApiError:
        return
    translation = status.translation
    if (status.id != run_id or translation is None or translation.state != "running"
            or len(translation.texts) != shown):
        st.rerun(scope="app")
    language = dict(translation.languages).get(translation.language or "", "Русский").lower()
    what = "аннотации" if translation.texts else f"ТОП на {language}"
    st.markdown(f'<p class="ta-translation-note" role="status">Переводим {what}: '
                f'{translation.completed} из {translation.total}…</p>', unsafe_allow_html=True)


def render_translation(translation: Translation | None, run_id: str | None) -> dict[str, str]:
    """Переключатель языка карточек ТОПа; возвращает перевод, который надо показать.

    Перевод приходит двумя порциями: сначала заголовки и темы, затем аннотации.
    Всё лежит в одном контейнере постоянного места: иначе появление
    переключателя сдвигало элементы страницы, и сетка ТОПа рисовалась заново.
    """
    with st.container(key="ta-translation"):
        if translation is None:
            return {}
        if translation.state == "unavailable":
            message = translation.message or "Перевод недоступен."
            st.markdown(f'<p class="ta-translation-note">Перевод ТОПа недоступен: {escape(message)} '
                        'Карточки показаны на языке оригинала.</p>', unsafe_allow_html=True)
            return {}
        if translation.state == "running" and run_id is not None:
            render_translation_progress(run_id, len(translation.texts))
        if not translation.texts and not translation.languages:
            return {}
        # Языки перевода, установленные владельцем, и оригинал; без выбора — прежние два.
        names = {code: name for code, name in translation.languages}
        options = (*names.values(), TOP_LANGUAGES[1]) if names else TOP_LANGUAGES
        current = names.get(translation.language or "", options[0])
        with st.container(key="ta-top-language"):
            choice = st.segmented_control("Язык карточек ТОПа", options, default=current,
                                          key="top_language", label_visibility="collapsed")
        if choice == TOP_LANGUAGES[1]:
            return {}
        chosen = next((code for code, name in names.items() if name == choice), None)
        if chosen is not None and chosen != translation.language:
            # Другой язык: перевод на него готовит сервис, страница перерисуется с ним.
            st.session_state["reading_language"] = chosen
            st.rerun(scope="app")
        return ReadingTexts(translation.texts, translation.language or "ru")


def render_publications(result: AnalysisResult, translation: Translation | None = None,
                        run_id: str | None = None, initial_count: int | None = None,
                        page_offset: int | None = None):
    if not result.publications and not result.signals:
        return
    entries, other_signals, remaining = top_entries(result, initial_count=initial_count,
                                                    page_offset=page_offset)
    # Карточки публикаций идут сразу за ТОПом технологий, без отдельного заголовка.
    if not entries:
        st.markdown('<div class="ta-section"><h2>Сигналы анализа</h2></div>', unsafe_allow_html=True)
    ru = render_translation(translation, run_id)
    if entries:
        # Все плитки ТОПа — в одном блоке-сетке и точно в порядке API.
        cards = ''.join(top_card(publication, signal, number, ru=ru)
                        for number, (publication, signal) in enumerate(entries, start=1))
        st.markdown(f'<div class="ta-grid">{cards}</div>', unsafe_allow_html=True)
    if other_signals:
        if entries:
            st.markdown('<div class="ta-section"><h2>Другие сигналы анализа</h2></div>'
                        '<p class="ta-section-note">Эти сигналы относятся к материалам вне ТОП-15 '
                        'или не сопоставлены с публикацией в общем списке.</p>', unsafe_allow_html=True)
        cards = ''.join(top_card(publication, signal, number, top=False)
                        for publication, signal, number in other_signals)
        st.markdown(f'<div class="ta-grid ta-signal-grid">{cards}</div>', unsafe_allow_html=True)
    initial_count = initial_count if initial_count is not None else len(result.publications)
    if page_offset is None:
        count = (f'<p class="ta-section-note ta-list-limit">Показано {initial_count} из '
                 f'{result.publication_total} найденных публикаций.</p>')
    else:
        last = page_offset + len(result.publications) - initial_count
        count = (f'<p class="ta-section-note ta-list-limit">Показаны первые {initial_count} и '
                 f'публикации {page_offset + 1}–{last} из '
                 f'{result.publication_total} найденных.</p>')
    if remaining:
        cards = ''.join(publication_card(publication, number)
                        for publication, number in remaining)
        # Подпись о лимите — внутри того же блока, что и список: отдельным
        # блоком она ложилась на нижнюю рамку раскрывающегося списка.
        expanded = ' open' if page_offset is not None else ''
        st.markdown(f'<div class="ta-more"><details class="ta-more-publications"{expanded}><summary>'
                    f'Остальные публикации в списке ({len(remaining)})</summary>'
                    f'<div class="ta-grid ta-more-publications-body">{cards}</div></details>'
                    f'{count}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ta-more">{count}</div>', unsafe_allow_html=True)
    if run_id is None:
        return
    next_offset = initial_count if page_offset is None else page_offset + len(result.publications) - initial_count
    previous_offset = None if page_offset == initial_count else (
        page_offset - PUBLICATION_PAGE_SIZE if page_offset is not None else None)
    back = (page_offset is not None
            and st.button("Предыдущая страница", key="previous_publications"))
    forward = (next_offset < result.publication_total and next_offset <= MAX_PUBLICATION_OFFSET
               and st.button("Следующая страница", key="next_publications"))
    if next_offset < result.publication_total and next_offset > MAX_PUBLICATION_OFFSET:
        st.warning("Остальные публикации пока нельзя открыть в веб-интерфейсе.")
    if not back and not forward:
        return
    state = st.session_state.get("publication_pages")
    if not isinstance(state, dict):
        return
    if back and previous_offset is None:
        st.session_state.publication_pages = {**state, "offset": None, "extra": ()}
        st.rerun()
        return
    target = previous_offset if back else next_offset
    assert target is not None
    client = api()
    try:
        page = client.publication_page(
            run_id, offset=target, expected_total=result.publication_total,
            known_ids=frozenset(item.publication_id for item in result.publications))
    except ApiError as error:
        st.error(str(error))
    else:
        st.session_state.publication_pages = {**state, "offset": target, "extra": page}
        st.rerun()


def visible_result(result: AnalysisResult, run_id: str | None) -> tuple[AnalysisResult, int | None]:
    """Keep only one optional page beside the initial list for the matching run."""
    if run_id is None:
        return result, None
    initial_ids = frozenset(item.publication_id for item in result.publications)
    identity = (run_id, result.publication_total,
                tuple(item.publication_id for item in result.publications))
    state = st.session_state.get("publication_pages")
    extra = state.get("extra") if isinstance(state, dict) else None
    offset = state.get("offset") if isinstance(state, dict) else None
    valid_page = (offset is None and extra == () or
                  type(offset) is int and isinstance(extra, tuple)
                  and len(result.publications) <= offset <= MAX_PUBLICATION_OFFSET
                  and offset < result.publication_total
                  and (offset - len(result.publications)) % PUBLICATION_PAGE_SIZE == 0
                  and len(extra) == min(PUBLICATION_PAGE_SIZE, result.publication_total - offset)
                  and all(isinstance(item, Publication) for item in extra)
                  and not initial_ids.intersection(item.publication_id for item in extra)
                  and len({item.publication_id for item in extra}) == len(extra))
    if not isinstance(state, dict) or state.get("identity") != identity or not valid_page:
        state = {"identity": identity, "offset": None, "extra": ()}
        st.session_state.publication_pages = state
    return replace(result, publications=result.publications + state["extra"]), state["offset"]


def confidence_markup(signal: Signal | Trend) -> str:
    """Уверенность методики в тренде: уровень, сколько признаков проверено и каких нет."""
    if signal.confidence is None:
        return ""
    label, filled = CONFIDENCE_LABELS[signal.confidence]
    total = len(signal.checked_features) + len(signal.unchecked_features)
    steps = "".join('<i class="on"></i>' if index < filled else '<i></i>' for index in range(3))
    missing = ", ".join(FEATURE_LABELS[name] for name in signal.unchecked_features)
    # «Не подтверждён» — только о проверенном росте; непроверенный стоит в перечне ниже.
    detail = ("Рост подтверждён." if signal.growth_confirmed else
              "Рост не подтверждён." if "growth" in signal.checked_features else "")
    # Когда не проверено ничего, перечень всех пяти повторял бы «0 из 5».
    if missing and signal.checked_features:
        detail = (detail + f" Не проверены: {missing}.").strip()
    detail = f"Проверено {len(signal.checked_features)} из {total} признаков. {detail}".strip()
    return (f'<div class="ta-confidence ta-confidence-{signal.confidence}"><div class="ta-meter">'
            f'<span class="ta-meter-label">Уверенность в тренде</span>'
            f'<span class="ta-steps" aria-hidden="true">{steps}</span>'
            f'<b class="ta-meter-value">{label}</b></div></div>'
            f'<p class="ta-confidence-detail">{escape(detail)}</p>')


def render_sources(result: AnalysisResult, *, coverage_only: bool = False):
    sources: dict[str, tuple[str, list[str]]] = {}
    if not coverage_only:
        for publication in result.publications:
            label = SOURCE_LABELS.get(publication.source_id, publication.source_id)
            entry = sources.setdefault(publication.url, (label, []))
            entry[1].append(publication.title)
        for signal in result.signals:
            for url in signal.source_urls:
                host = urlsplit(url).hostname or url
                entry = sources.setdefault(url, (host, []))
                entry[1].append(signal.title)
    if not sources and not result.source_coverage:
        return
    statuses = []
    for coverage in result.source_coverage:
        label = SOURCE_LABELS.get(coverage.source_id, coverage.source_id)
        state = ("Частичная выдача" if coverage.limit_reached and coverage.state != "unavailable"
                 else SOURCE_STATES[coverage.state])
        statuses.append(f'<li><strong>{escape(label)}</strong>'
                        f'<span>{escape(state)}</span>'
                        f'<span>Записей: {coverage.accepted}</span></li>')
    coverage_markup = ('<ul class="ta-coverage-list" aria-label="Состояние источников">'
                       + ''.join(statuses) + '</ul>') if statuses else ''
    rows = []
    for url, (label, titles) in sources.items():
        rows.append(f'<tr><td>{escape(label)}</td>'
                    f'<td>{escape("; ".join(dict.fromkeys(titles)))}</td>'
                    f'<td><a href="{escape(url, quote=True)}" target="_blank" '
                    f'rel="noopener noreferrer">{escape(url)}</a></td></tr>')
    coverage_note = ""
    if result.source_coverage:
        available = sum(item.state != "unavailable" for item in result.source_coverage)
        coverage_note = (f' Ответы получены от {available} из '
                         f'{len(result.source_coverage)} опрошенных площадок.')
    links_note = ("Общий список ссылок на показанные материалы и сигналы."
                  if sources else "Статус опрошенных площадок.")
    table = ('<div class="ta-table-wrap"><table class="ta-table">'
             f'<caption>Ссылок в списке: {len(sources)}</caption>'
             '<thead><tr><th scope="col">Площадка</th><th scope="col">Материал или сигнал</th>'
             '<th scope="col">Ссылка</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>') if sources else ''
    # Список источников свёрнут: он нужен для проверки, а не для чтения подряд.
    summary = "Источники"
    if sources:
        summary += f" · ссылок: {len(sources)}"
    if result.source_coverage:
        available = sum(item.state != "unavailable" for item in result.source_coverage)
        summary += f" · ответили {available} из {len(result.source_coverage)} площадок"
    st.markdown(f'<details class="ta-sources"><summary>{escape(summary)}</summary>'
                f'<p class="ta-section-note">{links_note}{coverage_note}</p>'
                f'{coverage_markup}{table}</details>', unsafe_allow_html=True)


@st.fragment(run_every="3s")
def render_radar_progress(run_id: str, done: int):
    """Пока радар считается, страница перерисовывается на каждом новом кандидате."""
    try:
        status = page_status(run_id)
    except ApiError:
        return
    radar = status.radar
    if status.id != run_id or radar is None or radar.state != "running" or radar.completed != done:
        st.rerun(scope="app")


def radar_source_publications(result: AnalysisResult | None,
                              radar: RadarStatus | None) -> dict[str, Publication]:
    """Match evidence to a unique TOP work, including preprint/journal URL changes."""
    if result is None or radar is None or radar.result is None:
        return {}
    top = result.top_publications or result.publications[:15]
    by_url: dict[str, list[Publication]] = {}
    by_work: dict[str, list[Publication]] = {}
    for publication in top:
        by_url.setdefault(url_key(publication.url), []).append(publication)
        by_work.setdefault(work_key(publication.title), []).append(publication)
    matched = {}
    for technology in (*radar.result.technologies, *radar.result.excluded):
        for source in technology.sources:
            direct = by_url.get(url_key(source.url), ())
            same_work = by_work.get(work_key(source.title), ())
            matched_publication = (direct[0] if len(direct) == 1 else
                                   same_work[0] if not direct and len(same_work) == 1 else None)
            if matched_publication is not None:
                matched[source.url] = matched_publication
    return matched


def radar_source_scores(result: AnalysisResult | None, radar: RadarStatus | None) -> dict[str, int]:
    """Publication scores for matching radar evidence, including report pages."""
    return {url: publication.model_confidence.score
            for url, publication in radar_source_publications(result, radar).items()
            if publication.model_confidence is not None}


def render_radar(radar: RadarStatus | None, run_id: str | None, *,
                 result: AnalysisResult | None = None, saved: bool = False):
    if radar is None:
        return
    # Отчёты по технологиям сохранённого анализа ведут к нему же, а не к текущему.
    st.markdown(radar_markup(radar, run_id if saved else None,
                             radar_source_scores(result, radar)), unsafe_allow_html=True)
    if radar.state == "running" and run_id is not None:
        render_radar_progress(run_id, radar.completed)


def render_result(result: AnalysisResult, translation: Translation | None = None,
                  run_id: str | None = None, radar: RadarStatus | None = None, *, saved: bool = False):
    initial_count = len(result.publications)
    result, page_offset = visible_result(result, run_id)
    if not result.signals and not result.publications and radar is None:
        coverage = (' Охват источников неполный: часть материалов могла не попасть в анализ.'
                    if result.incomplete_coverage else '')
        if result.openalex_rate_limited:
            coverage += OPENALEX_LIMIT_NOTE
        st.markdown('<section class="ta-empty"><h2>Нет результатов для показа</h2>'
                    f'<p>По этому запросу не найдено публикаций или сигналов с '
                    f'допустимыми ссылками.{coverage}</p></section>',
                    unsafe_allow_html=True)
        render_sources(result)
        return
    if result.source_coverage:
        source_ids = {coverage.source_id for coverage in result.source_coverage
                      if coverage.accepted > 0}
    else:
        source_ids = {publication.source_id for publication in result.publications}
    if not source_ids and not result.source_coverage:
        source_ids = {urlsplit(url).hostname or url for signal in result.signals
                      for url in signal.source_urls}
    st.markdown('<div class="ta-section"><h2>Результаты поиска</h2></div>'
                '<div class="ta-stats">'
                f'<dl class="ta-stat"><dt>Публикаций</dt><dd>{result.publication_total}</dd></dl>'
                f'<dl class="ta-stat"><dt>Сигналов анализа</dt><dd>{len(result.signals)}</dd></dl>'
                f'<dl class="ta-stat"><dt>Источников</dt><dd>{len(source_ids)}</dd></dl>'
                '</div>', unsafe_allow_html=True)
    render_radar(radar, run_id, result=result, saved=saved)
    # Если радар не дал ни одной карточки с источниками, список публикаций
    # остаётся доступным; готовый ТОП состоит только из карточек технологий.
    ready = radar is not None and radar.state == "ready" and radar.result is not None
    linked = (any(technology.sources for technology in radar.result.technologies)
              if ready and radar is not None and radar.result is not None else False)
    if not linked:
        render_publications(result, translation, run_id, initial_count, page_offset)
    render_sources(result, coverage_only=linked)


def duration_text(seconds: int) -> str:
    if seconds < 60:
        return "меньше минуты"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} мин"
    hours, rest = divmod(minutes, 60)
    return f"{hours} ч {rest} мин" if rest else f"{hours} ч"


def remaining_text(status: AnalysisStatus) -> str:
    """Остаток по прогнозу, зафиксированному при запуске; без прогноза — ничего."""
    elapsed, expected = status.elapsed_seconds, status.expected_seconds
    if status.state == "waiting":
        ahead = (status.queue_position or 1) - 1
        return "вы следующий" if not ahead else f"перед вами {ahead} {plural(ahead, 'анализ', 'анализа', 'анализов')}"
    if elapsed is None or expected is None or status.state == "cancelling":
        return ""
    if elapsed >= expected:
        return f"дольше прогноза ({duration_text(max(60, expected))})"
    left = expected - elapsed
    return "осталось меньше минуты" if left < 60 else f"осталось около {duration_text(left)}"


def plural(number: int, one: str, few: str, many: str) -> str:
    """Форма слова после числа: 1 анализ, 2 анализа, 5 анализов."""
    tail, tens = number % 10, number % 100
    return one if tail == 1 and tens != 11 else few if 2 <= tail <= 4 and not 12 <= tens <= 14 else many


def clock_text(seconds: int) -> str:
    minutes, seconds = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def progress_markup(status: AnalysisStatus, stage_elapsed_seconds: int | None = None) -> str:
    """Show saved progress for the current stage, not an invented overall percent.

    У этапа без счётчика рядом с полосой идут его собственные часы: даже при
    отключённых анимациях видно, что работа продолжается.
    """
    if status.state == "cancelling":
        detail = "Останавливаем анализ…"
    else:
        detail = status.message or "Подготавливаем анализ…"
    if status.total:
        completed = min(status.completed, status.total)
        percent = round(100 * completed / status.total)
        value = f'<span class="ta-progress-value">{completed} из {status.total}</span>'
        attributes = f' aria-valuenow="{percent}" aria-valuetext="{completed} из {status.total}"'
        fill = f' style="width:{percent}%"'
        animation = ""
    else:
        stage_key = escape(f"stage:{status.id}:{status.stage}", quote=True)
        value = ("" if stage_elapsed_seconds is None else
                 f'<span class="ta-progress-value">на этапе <b data-ta-clock="{stage_key}" '
                 f'data-ta-elapsed="{stage_elapsed_seconds}">'
                 f'{clock_text(stage_elapsed_seconds)}</b></span>')
        attributes = ' aria-valuetext="Выполняется"'
        fill = ""
        animation = " ta-progress-indeterminate"
    return ('<div class="ta-progress">'
            f'<p class="ta-progress-detail">{escape(detail)}</p>'
            f'<div class="ta-progress-row"><div class="ta-progress-track" role="progressbar" '
            f'aria-label="Прогресс этапа" aria-valuemin="0" aria-valuemax="100"{attributes}>'
            f'<span class="ta-progress-fill{animation}"{fill}></span></div>{value}</div></div>')


def run_phase(status: AnalysisStatus, reached: int = -1) -> int:
    """Фаза текущего этапа; назад не откатывается, если поздний этап дособирает раннее."""
    current = next((index for index, (_, _, stages) in enumerate(RUN_PHASES)
                    if status.stage in stages), 0 if status.state not in {"queued", "waiting"} else -1)
    return max(current, reached)


def run_markup(status: AnalysisStatus, phase: int,
               stage_elapsed_seconds: int | None = None) -> str:
    """Панель хода анализа: заголовок с часами, пять фаз и подсказка о методике.

    Живость не держится на анимациях — при системном «уменьшении движения» их
    нет: часы тикают в браузере (скрипт RUN_CLOCK) от секунд сервера, а
    подсказка сменяется по времени анализа при каждом опросе сервера.
    """
    if status.state == "cancelling":
        title = "Останавливаем анализ…"
    elif status.state == "waiting":
        title = "Ваш анализ в очереди: все места заняты"
    elif status.state == "queued":
        title = "Ставим анализ в очередь…"
    else:
        title = "Идёт анализ направления"
    remaining = remaining_text(status)
    remaining_markup = f'<span class="ta-run-left">{escape(remaining)}</span>' if remaining else ""
    elapsed = status.elapsed_seconds or 0
    clock_key = escape(f"run:{status.id}", quote=True)
    clock = (f'<div class="ta-run-clock"><b data-ta-clock="{clock_key}" data-ta-elapsed="{elapsed}">'
             f'{clock_text(elapsed)}</b>{remaining_markup}</div>')
    steps = []
    for index, (name, hint, _) in enumerate(RUN_PHASES):
        if index < phase:
            state, mark, body = "done", CHECK_ICON, ""
        elif index == phase:
            state, mark, body = "active", "", progress_markup(status, stage_elapsed_seconds)
        else:
            state, mark, body = "next", "", f'<p class="ta-step-hint">{escape(hint)}</p>'
        current = ' aria-current="step"' if state == "active" else ""
        steps.append(f'<li class="ta-step ta-step-{state}"{current}><span class="ta-step-mark">{mark}</span>'
                     f'<div class="ta-step-body"><span class="ta-step-name">{escape(name)}</span>'
                     f'{body}</div></li>')
    fact = RUN_FACTS[(status.elapsed_seconds or 0) // FACT_SECONDS % len(RUN_FACTS)]
    return (f'<section class="ta-run" role="status" aria-label="Ход анализа">'
            f'<div class="ta-run-head"><span class="ta-run-pulse" aria-hidden="true"></span>'
            f'<div class="ta-run-title"><p>{escape(title)}</p>'
            f'<span class="ta-run-query">{escape(status.query)}</span></div>{clock}</div>'
            f'<ol class="ta-steps-list">{"".join(steps)}</ol>'
            f'<div class="ta-facts"><span class="ta-facts-label">Пока идёт анализ</span>'
            f'<p class="ta-fact">{escape(fact)}</p></div></section>')


@st.fragment(run_every="5s")
def render_service_wait():
    """Пока сервис занят чужим анализом, страница ждёт, когда он освободится."""
    try:
        status = api().current_analysis(**reading_language())
    except AccessRefused as refusal:
        if refusal.reason == "signed_out":
            sign_out()
        st.rerun(scope="app")
    except ApiError:
        return
    if not status.service_busy:
        st.rerun(scope="app")


@st.fragment(run_every="2s")
def render_active_analysis():
    client = api()
    try:
        status = client.current_analysis(**reading_language())
    except AccessRefused as refusal:
        if refusal.reason == "signed_out":
            sign_out()
        st.rerun(scope="app")
    except ApiError as error:
        st.error(str(error))
        return
    if (status.state not in ACTIVE_STATES
            or status.id != st.session_state.get("viewed_analysis_id")):
        st.rerun(scope="app")
    reached = st.session_state.get("run_phase")
    phase = run_phase(status, reached[1] if reached and reached[0] == status.id else -1)
    st.session_state.run_phase = (status.id, phase)
    # Начало этапа — по первому опросу, в котором он виден; для его часов.
    stage = st.session_state.get("run_stage")
    if (not isinstance(stage, tuple) or len(stage) != 4
            or stage[:3] != (status.id, status.stage, "monotonic")):
        stage = (status.id, status.stage, "monotonic", time.monotonic())
        st.session_state.run_stage = stage
    stage_elapsed = max(0, int(time.monotonic() - stage[3]))
    st.markdown(run_markup(status, phase, stage_elapsed), unsafe_allow_html=True)


def saved_head_markup(saved: AnalysisStatus, current: AnalysisStatus) -> str:
    """Шапка анализа из истории: что это за анализ и когда он сделан."""
    kicker = "Анализ из истории"
    if saved.created_at is not None:
        kicker += " · " + history_date(saved.created_at)
    note = ""
    if current.state in ACTIVE_STATES and current.id != saved.id:
        note = (f'<p class="ta-saved-note">Сейчас идёт анализ «{escape(current.query)}»: он продолжается, '
                'результат появится на главной странице.</p>')
    return (GRAPH + f'<section class="ta-saved"><p class="ta-saved-kicker">{escape(kicker)}</p>'
            f'<h1>{escape(saved.query)}</h1>{note}</section>')


def render_saved(run_id: str, current: AnalysisStatus):
    """Страница готового анализа из истории: результат, ТОП технологий и их отчёты.

    Она ничего не меняет у сервиса, поэтому открывается и во время нового анализа.
    """
    busy = current.state in ACTIVE_STATES

    def back():
        with st.container(key="ta-saved-back"):
            st.button("К идущему анализу" if busy else "Новый анализ", key="leave_saved",
                      icon=":material/arrow_back:", on_click=leave_saved)

    try:
        saved = page_status(run_id)
    except ApiError as error:
        st.markdown(GRAPH, unsafe_allow_html=True)
        st.error(str(error))
        back()
        return
    report = st.query_params.get("report")
    if report and saved.radar is not None and saved.radar.result is not None:
        technology = find_report(saved.radar.result, report)
        if technology is not None:
            st.markdown(report_markup(technology, saved.radar.result, report, run_id,
                                      radar_source_scores(saved.result, saved.radar)),
                        unsafe_allow_html=True)
            return
    st.markdown(saved_head_markup(saved, current), unsafe_allow_html=True)
    back()
    if saved.result is not None:
        render_result(saved.result, saved.translation, saved.id, saved.radar, saved=True)


def main():
    st.set_page_config(page_title="Trendanalyser — анализ технологий", layout="wide")
    style = Path(__file__).with_name("web_styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
    if os.environ.get("TREND_WEB_REQUIRE_AUTH") == "1":
        st.html(HTTPS_ONLY, unsafe_allow_javascript=True)
    # Блокировку владельца проверяем до того, как страница что-то покажет.
    access = check_access()
    if access == "blocked":
        render_blocked()
        return
    if access == "signed_out":
        sign_out()
    if st.session_state.pop("forget_cookie", False):
        st.html(FORGET_VISITOR, unsafe_allow_javascript=True)
    known = st.session_state.get("visitor")
    if not authenticated():
        track_presence("login")
        return
    if st.session_state.get("visitor") != known:
        # Вход только что восстановлен из cookie: этого гостя могли выгнать или заблокировать.
        access = check_access()
        if access == "blocked":
            render_blocked()
            return
        if access == "signed_out":
            sign_out()
            st.rerun()
    remember = st.session_state.pop("visitor_cookie", None)
    st.html(ENTER_TO_SUBMIT + ROW_DETAILS + RUN_CLOCK + (remember_visitor(remember) if remember else ""),
            unsafe_allow_javascript=True)
    client = api()
    # Сохранённый результат собирается из архива несколько секунд, а прежняя
    # страница до ответа остаётся на экране: уведомление говорит, что идёт.
    opening = st.session_state.pop("history_opening", None)
    if opening:
        st.toast(f"Открываем «{opening[:80]}»…", icon=":material/history:")
    try:
        status: AnalysisStatus = client.current_analysis(**reading_language())
    except AccessRefused as refusal:
        if refusal.reason == "signed_out":
            sign_out()
        st.rerun()
    except ApiError as error:
        track_presence("main", view="error")
        st.error(str(error))
        return
    if status.id is not None and st.session_state.get("viewed_analysis_id") != status.id:
        st.session_state.viewed_analysis_id = status.id
        st.session_state.query = status.query
        st.session_state.analysis_mode = status.mode
        st.session_state.pop("error", None)
        st.session_state.pop("error_run_id", None)
    if (status.state not in ACTIVE_STATES and status.id is not None
            and st.session_state.get("error_run_id") == status.id):
        st.session_state.pop("error", None)
        st.session_state.pop("error_run_id", None)
    render_theme()
    busy = status.state in ACTIVE_STATES
    render_history_button()
    if st.session_state.get("history_open"):
        render_history()
    saved = st.query_params.get("run")
    # Страница-отчёт по технологии открывается ссылкой из ТОПа и заменяет весь экран.
    report = st.query_params.get("report")
    technology = (find_report(status.radar.result, report)
                  if report and status.state == "succeeded" and status.radar is not None
                  and status.radar.result is not None else None)
    if saved:
        track_presence("saved")
    elif technology is not None:
        track_presence("report", detail=technology.title_ru or technology.title)
    else:
        track_presence("main", view=page_view(status, busy), detail=status.query if status.id else None)
    render_owner_messages()
    if saved:
        render_saved(saved, status)
        return
    if (technology is not None and status.radar is not None
            and status.radar.result is not None and report is not None):
        st.markdown(report_markup(technology, status.radar.result, report,
                                  source_scores=radar_source_scores(status.result, status.radar)),
                    unsafe_allow_html=True)
        return
    render_head()
    with st.form("analysis", clear_on_submit=False, enter_to_submit=not busy):
        st.text_area("Технологическое направление", key="query", height=96,
                     label_visibility="collapsed", disabled=busy,
                     placeholder="Опишите технологическое направление",
                     max_chars=MAX_QUERY_LENGTH)
        # Переключатель из двух режимов: плашка выбранного плавно переезжает (CSS).
        # Внутри формы выбор не перерисовывает страницу, поэтому пустой выбор
        # (повторное нажатие) означает быстрый режим — как и выглядит.
        st.session_state.setdefault("analysis_mode", "fast")
        st.segmented_control("Режим анализа", list(MODE_LABELS), format_func=lambda value: MODE_LABELS[value],
                             key="analysis_mode", disabled=busy, label_visibility="collapsed")
        # Та же круглая кнопка отправляет запрос и затем отменяет его. Подпись
        # остаётся доступной скринридеру; CSS меняет стрелку на квадрат.
        if busy:
            st.form_submit_button("Отменяем анализ" if status.state == "cancelling"
                                  else "Отменить анализ", key="cancel_analysis",
                                  type="primary", disabled=status.state == "cancelling",
                                  on_click=cancel_analysis, args=(status.id,),
                                  help="Остановить текущий анализ")
        else:
            st.form_submit_button("Анализировать", key="start_analysis", type="primary",
                                  on_click=queue_analysis, help="Анализировать",
                                  disabled=status.service_busy or status.paused)
    if status.paused and not busy:
        st.markdown('<p class="ta-busy-note" role="status">Владелец сервиса временно приостановил новые '
                    'анализы. Готовые результаты и история доступны.</p>', unsafe_allow_html=True)
    if status.service_busy and not busy:
        st.markdown('<p class="ta-busy-note" role="status">Сейчас заняты все места для анализа, и очередь '
                    'заполнена. Запустить свой можно через несколько минут; страница обновится сама.</p>',
                    unsafe_allow_html=True)
        render_service_wait()
    elif status.slots_full and not busy:
        ahead = status.queue_length
        st.markdown('<p class="ta-busy-note" role="status">Сейчас идут анализы других посетителей. Новый '
                    'встанет в очередь' + (f' — перед ним {ahead} {plural(ahead, "анализ", "анализа", "анализов")}'
                                           if ahead else ' и начнётся первым') + ', как только освободится место.</p>',
                    unsafe_allow_html=True)
    if status.deep_disabled and not busy:
        st.markdown('<p class="ta-busy-note" role="status">Глубокий режим временно выключен владельцем '
                    'сервиса; доступен быстрый.</p>', unsafe_allow_html=True)
    if status.state == "idle" and not st.session_state.get("error"):
        render_about()
    if busy:
        render_active_analysis()
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    elif status.state == "succeeded" and status.result is not None:
        render_result(status.result, status.translation, status.id, status.radar)
    elif status.state == "cancelled":
        st.info("Анализ отменён. Можно начать новый запрос.")
    elif status.state in {"failed", "interrupted"}:
        st.error(status.error or "Анализ завершился без результата. Можно начать новый запрос.")


if __name__ == "__main__":
    main()
