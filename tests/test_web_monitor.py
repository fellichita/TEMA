"""Живая сводка для панели владельца: присутствие, журнал и этапы анализов."""

import json

import pytest

from app.web_monitor import ONLINE_SECONDS, Monitor, describe_agent

CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/140.0.0.0 Safari/537.36")
VISITOR = "visitor-0123456789abcdef"


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


def texts(monitor, after=0):
    return [event["text"] for event in monitor.snapshot(after)["events"]]


def test_a_visitor_from_the_login_screen_to_leaving_is_one_story():
    clock = Clock()
    monitor = Monitor(anonymous="Не вошёл", clock=clock)
    monitor.presence("login-tab-01", None, "login", agent=CHROME, ip="203.0.113.7")
    monitor.presence("login-tab-01", None, "login", agent=CHROME, ip="203.0.113.7", event="login_failed")
    monitor.presence("login-tab-01", VISITOR, "login", agent=CHROME, ip="203.0.113.7", event="login")
    # Перезагрузка страницы — новая вкладка Streamlit, но не новый приход.
    clock.now += 1
    monitor.presence("main-tab-002", VISITOR, "main", agent=CHROME, ip="203.0.113.7")
    assert texts(monitor) == ["Кто-то открыл экран входа (Chrome · Windows, 203.0.113.7)",
                              "Кто-то ввёл неверный пароль (Chrome · Windows, 203.0.113.7)",
                              "Гость 1 вошёл по паролю (Chrome · Windows, 203.0.113.7)"]
    [person] = monitor.snapshot()["visitors"]
    assert {key: person[key] for key in ("label", "number", "page", "agent", "ip", "runs", "tabs", "online",
                                         "signed_in", "blocked", "can_block_ip")} == {
        "label": "Гость 1", "number": 1, "page": "Главная", "agent": "Chrome · Windows", "ip": "203.0.113.7",
        "runs": 0, "tabs": 2, "online": True, "signed_in": True, "blocked": False, "can_block_ip": True}

    clock.now += ONLINE_SECONDS + 1
    snapshot = monitor.snapshot(3)
    assert [event["text"] for event in snapshot["events"]] == ["Гость 1 ушёл с сайта"]
    assert snapshot["visitors"][0]["online"] is False and snapshot["counters"]["online"] == 0
    clock.now += 5
    monitor.presence("main-tab-003", VISITOR, "main", agent=CHROME, ip="203.0.113.7")
    assert texts(monitor, 4) == ["Гость 1 снова на сайте (Chrome · Windows, 203.0.113.7)"]
    counters = monitor.snapshot()["counters"]
    assert (counters["logins"], counters["login_failures"], counters["visitors"], counters["online"]) == (1, 1, 1, 1)


def test_a_second_open_tab_keeps_the_visitor_on_the_site():
    clock = Clock()
    monitor = Monitor(clock=clock)
    monitor.presence("first-tab-01", VISITOR, "main")
    monitor.presence("second-tab-2", VISITOR, "saved")
    clock.now += ONLINE_SECONDS - 5
    monitor.presence("second-tab-2", VISITOR, "saved")
    clock.now += 10
    assert "Гость 1 ушёл с сайта" not in texts(monitor)
    [person] = monitor.snapshot()["visitors"]
    assert person["online"] and person["tabs"] == 1 and person["page"] == "Анализ из истории"


def test_presence_rejects_what_the_site_never_sends():
    monitor = Monitor()
    for session, page, event in (("short", "main", None), ("valid-tab-01", "admin", None),
                                 ("valid-tab-01", "main", "logout")):
        with pytest.raises(ValueError):
            monitor.presence(session, None, page, event=event)
    monitor.presence("valid-tab-01", None, "main", ip="not an address", agent="x" * 5_000)
    [person] = monitor.snapshot()["visitors"]
    assert person["ip"] is None and len(person["agent_raw"]) == 400


def test_stage_timeline_sums_returns_and_reports_the_finish_once():
    clock = Clock()
    monitor = Monitor(clock=clock)
    monitor.presence("visitor-tab1", VISITOR, "main")
    monitor.run_started("run-1", "solid-state batteries", "fast", VISITOR)
    for stage, seconds in (("plan", 10), ("discovery", 20), ("history", 5), ("discovery", 3)):
        monitor.observe("run-1", {"state": "running", "stage": stage})
        clock.now += seconds
    timeline = monitor.timeline("run-1")
    assert [(entry["label"], entry["seconds"], entry["active"]) for entry in timeline] == [
        ("Границы направления", 10, False), ("Сбор публикаций", 23, True), ("Статистика по годам", 5, False)]
    assert timeline[1]["current_since"] != timeline[1]["started_at"]
    monitor.observe("run-1", {"state": "succeeded", "stage": "complete"})
    monitor.observe("run-1", {"state": "succeeded", "stage": "complete"})
    assert texts(monitor, 1) == ["Гость 1 запустил анализ «solid-state batteries» · быстрый",
                                 "Этап: Границы направления", "Этап: Сбор публикаций", "Этап: Статистика по годам",
                                 "Анализ «solid-state batteries» готов за 38 с"]
    assert not any(entry["active"] for entry in monitor.timeline("run-1"))
    snapshot = monitor.snapshot()
    assert snapshot["counters"]["runs_started"] == snapshot["counters"]["runs_succeeded"] == 1
    assert snapshot["visitors"][0]["runs"] == 1


def test_a_failed_run_is_an_error_event_with_its_reason():
    clock = Clock()
    monitor = Monitor(clock=clock)
    monitor.run_started("run-2", "quantum sensors", "deep", None)
    monitor.observe("run-2", {"state": "running", "stage": "discovery"})
    clock.now += 125
    monitor.observe("run-2", {"state": "failed", "error": "OpenAlex отказал: лимит запросов."})
    [event] = monitor.snapshot(2)["events"]
    assert event["level"] == "error"
    assert event["text"] == ("Анализ «quantum sensors» не завершён через 2 мин 5 с: "
                             "OpenAlex отказал: лимит запросов.")
    # Неизвестный запуск (начат не через веб или до перезапуска API) журнал не засоряет.
    monitor.observe("other", {"state": "failed"})
    assert monitor.snapshot()["last_seq"] == 3


@pytest.mark.parametrize(("agent", "expected"), [
    (CHROME, "Chrome · Windows"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 "
     "YaBrowser/25.8.0.0 Safari/537.36", "Яндекс Браузер · Windows"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 "
     "Safari/537.36 Edg/140.0", "Edge · Windows"),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/18.0 Mobile/15E148 Safari/604.1", "Safari · iOS"),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 "
     "Mobile Safari/537.36", "Chrome · Android"),
    ("", None),
], ids=["chrome", "yandex", "edge", "iphone", "android", "empty"])
def test_browsers_are_named_the_way_the_owner_knows_them(agent, expected):
    assert describe_agent(agent) == expected


def test_what_a_visitor_does_becomes_their_trail_and_the_live_feed():
    clock = Clock()
    monitor = Monitor(clock=clock)
    monitor.presence("visitor-tab1", VISITOR, "main", view="idle")
    for view, detail in (("idle", None), ("running", "солнечные элементы"), ("result", "солнечные элементы")):
        clock.now += 15
        monitor.presence("visitor-tab1", VISITOR, "main", view=view, detail=detail)
    monitor.visitor_action(VISITOR, "листает публикации: 20 из 312", repeat="publications:run")
    monitor.visitor_action(VISITOR, "листает публикации: 40 из 312", repeat="publications:run")
    clock.now += 15
    monitor.presence("visitor-tab1", VISITOR, "report", detail="Perovskite tandem cells")
    snapshot = monitor.snapshot()
    [person] = snapshot["visitors"]
    assert person["doing"] == "читает отчёт о технологии «Perovskite tandem cells»"
    trail = [action["text"] for action in monitor.snapshot(person=person["key"])["person"]["actions"]]
    # Повтор листания обновляет запись, а не плодит строки; новые — сверху.
    assert trail == ["читает отчёт о технологии «Perovskite tandem cells»", "листает публикации: 40 из 312",
                     "смотрит результат анализа «солнечные элементы»",
                     "следит за своим анализом «солнечные элементы»", "открыл сайт"]
    assert "Гость 1 смотрит результат анализа «солнечные элементы»" in [event["text"] for event in snapshot["events"]]


def test_owner_blocks_a_browser_and_its_address_and_the_block_survives_a_restart(tmp_path):
    from app.web_monitor import Blocklist

    path = tmp_path / "blocklist.json"
    monitor = Monitor(clock=Clock(), blocklist=Blocklist(path))
    monitor.presence("visitor-tab1", VISITOR, "main", agent=CHROME, ip="198.51.100.20")
    [person] = monitor.snapshot()["visitors"]
    assert monitor.block(person["key"]) == {"visitor": VISITOR,
                                            "message": "Гость 1 заблокирован: браузер и адрес 198.51.100.20."}
    # Тот же браузер с другого адреса и другой браузер с того же адреса — оба за дверью.
    assert monitor.access(VISITOR, "203.0.113.9") == "blocked"
    assert monitor.access(None, "198.51.100.20") == "blocked"
    assert monitor.blocked_visitor(VISITOR) == "blocked"
    assert monitor.presence("other-tab-02", None, "login", ip="198.51.100.20") == {"access": "blocked", "messages": []}
    assert monitor.access(None, "203.0.113.9") is None

    restarted = Monitor(clock=Clock(), blocklist=Blocklist(path))
    blocks = restarted.snapshot()["blocks"]
    assert {(entry["kind"], entry["value"]) for entry in blocks} == {("visitor", VISITOR), ("ip", "198.51.100.20")}
    for entry in blocks:
        restarted.unblock(entry["id"])
    assert restarted.access(VISITOR, "198.51.100.20") is None
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_this_computer_and_local_network_are_never_blocked_by_address():
    monitor = Monitor(clock=Clock())
    monitor.presence("local-tab-01", None, "main", ip="127.0.0.1")
    [person] = monitor.snapshot()["visitors"]
    assert person["can_block_ip"] is False
    with pytest.raises(LookupError):
        monitor.block(person["key"])
    monitor.presence("lan-tab-0001", VISITOR, "main", ip="192.168.1.20")
    guest = next(entry for entry in monitor.snapshot()["visitors"] if entry["signed_in"])
    monitor.block(guest["key"])
    assert [entry["kind"] for entry in monitor.snapshot()["blocks"]] == ["visitor"]
    assert monitor.access(None, "192.168.1.20") is None


def test_signed_out_guest_needs_the_password_again_and_messages_arrive_once():
    monitor = Monitor(clock=Clock())
    monitor.presence("visitor-tab1", VISITOR, "main")
    [person] = monitor.snapshot()["visitors"]
    assert "15 секунд" in monitor.message(person["key"], "  Сервис перезапустится   через 5 минут ")
    answer = monitor.presence("visitor-tab1", VISITOR, "main")
    assert answer == {"access": None, "messages": ["Сервис перезапустится через 5 минут"]}
    assert monitor.presence("visitor-tab1", VISITOR, "main")["messages"] == []
    monitor.sign_out(person["key"])
    assert monitor.presence("visitor-tab1", VISITOR, "main")["access"] == "signed_out"
    assert monitor.blocked_visitor(VISITOR) == "signed_out"
    with pytest.raises(ValueError):
        monitor.message(person["key"], "   ")
    with pytest.raises(KeyError):
        monitor.sign_out("0" * 12)
    texts = [event["text"] for event in monitor.snapshot()["events"]]
    assert "Владелец → Гость 1: «Сервис перезапустится через 5 минут»" in texts
    assert "Гость 1 получил сообщение владельца" in texts and "Владелец завершил сеанс: Гость 1" in texts
