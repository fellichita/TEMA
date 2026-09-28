"""Native passport limits preserve complete legal quotations and keyboard access."""

from copy import deepcopy
from datetime import UTC, datetime
import gc
from hashlib import sha256
import re
import tkinter as tk
from tkinter import ttk
from tkinter import font
from types import SimpleNamespace
import weakref

from app.pilot.contracts import Candidate, Claim, Evidence, TrendCard
from app.ui.display import Display
from app.ui.pilot_passport import show_passport
from app.ui.theme import apply_theme
from app.ui.viewport import ScrollViewport
from app.ui.windows import WindowRegistry
from tests.test_pilot_operation_ownership import DeferredController
from tests.ui.test_desktop import TkCase
from tests.ui.test_pilot_panel import descendants


def legal_card(count=200, length=500, *, url_length=0, claims=0, limitations=1):
    candidate = Candidate(candidate_id="candidate", plan_hash="a" * 64, label="Rendering fixture",
        definition="Synthetic public-text rendering fixture; no scientific claim.",
        admission_rule_version="fixture", admission_rule_hash="b" * 64,
        discovery_snapshot_id="snapshot", discovery_study_ids=("study",), specificity="uncertain")
    evidence = []
    for index in range(count):
        prefix = f"Exact quotation {index:03d}: "
        quote = prefix + ("Original citation; αβ scientific text. " * length)[:length - len(prefix)]
        url = f"https://example.org/source/{index}"
        if url_length:
            url += "?q=" + "x" * (url_length - len(url) - 3)
        evidence.append(Evidence(evidence_id=f"evidence-{index}", revision_id=f"revision-{index}",
            study_id=f"study-{index}", source="openalex", source_url=url, retrieved_at=datetime(2026, 9, 12, tzinfo=UTC),
            text_hash=sha256(quote.encode()).hexdigest(), text_field="abstract", start=0, end=len(quote), quote=quote))
    statements = tuple(Claim(claim_id=f"claim-{index}", role=("novelty", "problem", "advantage", "case")[index % 4],
        text=(f"Claim {index:02d}: " + "Long original narrative. " * 300)[:6000], support="unverified", evidence_ids=())
        for index in range(claims))
    return TrendCard(candidate=candidate, category="early_signal", quality="partial", claims=statements,
        evidence=tuple(evidence), limitations=tuple(f"Limitation {index:04d}. " + "x" * 100 for index in range(limitations))).model_dump(mode="json")


class PassportPagesTests(TkCase):
    def setUp(self):
        super().setUp()
        apply_theme(self.root)
        self.calls = DeferredController()
        self.panel = SimpleNamespace(app=SimpleNamespace(root=self.root, closing=False,
            child_windows=WindowRegistry(), controller=self.calls), error=lambda error: self.fail(str(error)))

    def open(self, card):
        show_passport(self.panel, card, {"result": {"run_id": "saved-run"}})
        return self.panel.app.child_windows[-1]

    def content(self, window):
        text = []
        for widget in descendants(window):
            if isinstance(widget, tk.Text):
                text.append(widget.get("1.0", "end-1c"))
            elif isinstance(widget, ttk.Label):
                variable = str(widget.cget("textvariable"))
                text.append(str(widget.getvar(variable)) if variable else str(widget.cget("text")))
        return text

    def button(self, window, title):
        return next(widget for widget in descendants(window)
                    if isinstance(widget, ttk.Button) and widget.cget("text") == title)

    def test_initial_evidence_count_is_bounded_and_normal_twenty_stays_on_one_page(self):
        window = self.open(legal_card(21))
        self.assertLessEqual(sum("Exact quotation" in text for text in self.content(window)), 20)
        self.assertIn("1–20 из 21", "\n".join(self.content(window)))
        window.destroy()
        window = self.open(legal_card(20))
        self.assertEqual(sum("Exact quotation" in text for text in self.content(window)), 20)
        self.assertFalse(any(widget.cget("text") == "Далее" for widget in descendants(window) if isinstance(widget, ttk.Button)))

    def test_all_two_hundred_quotes_and_links_are_exact_in_order_with_bounded_callbacks(self):
        card = legal_card()
        unchanged = deepcopy(card)
        window = self.open(card)
        seen = []
        command_counts = []
        while True:
            texts = self.content(window)
            shown = [item for item in card["evidence"] if any(item["quote"] in text for text in texts)]
            self.assertLessEqual(len(shown), 20)
            for item in shown:
                seen.append(item["quote"])
                link = next(widget for widget in descendants(window)
                            if isinstance(widget, ttk.Label) and str(widget.cget("text")) == item["source_url"])
                # Invoke the real binding without requiring off-screen quotes to be painted.
                callback = re.search(r"\[([^\s]+)", link.bind("<Return>"))
                self.assertIsNotNone(callback)
                self.root.tk.call(callback.group(1))
                self.assertEqual(self.calls.arguments("open_url")[-1], (item["source_url"],))
                self.calls.complete("open_url")
            command_counts.append(len(self.root.tk.call("info", "commands")))
            following = self.button(window, "Далее")
            if following.instate(["disabled"]):
                break
            following.invoke()
        self.assertEqual(seen, [item["quote"] for item in card["evidence"]])
        self.assertLessEqual(max(command_counts) - min(command_counts), 4)
        self.button(window, "Первая").invoke()
        self.assertIn(card["evidence"][0]["quote"], "\n".join(self.content(window)))
        self.button(window, "Последняя").invoke()
        self.assertIn(card["evidence"][-1]["quote"], "\n".join(self.content(window)))
        self.assertEqual(card, unchanged)

    def test_maximum_quote_and_url_use_bounded_readonly_widgets_at_double_scale(self):
        self.root._display = Display(self.root, scale=2)
        card = legal_card(200, 12000, url_length=4096)
        window = self.open(card)
        texts = self.content(window)
        self.assertLessEqual(sum(len(text) for text in texts if "Exact quotation" in text or text.startswith("https://")), 16100)
        self.assertIn(card["evidence"][0]["quote"], "\n".join(texts))
        self.assertIn(card["evidence"][0]["source_url"], texts)
        self.root.deiconify()
        viewport = next(widget for widget in window.winfo_children() if isinstance(widget, ScrollViewport))
        self.wait_mapped([window, viewport.canvas])
        self.pump(lambda: viewport.content.winfo_height() > 1)
        self.assertLess(viewport.content.winfo_reqheight(), 6000)
        self.assertEqual(len([widget for widget in descendants(window) if isinstance(widget, tk.Text)]), 2)
        for widget in descendants(window):
            if isinstance(widget, tk.Text):
                self.assertEqual(str(widget.cget("state")), "disabled")
                self.assertLessEqual(int(widget.cget("height")), 6)
        source = self.button(window, "Открыть первоисточник")
        url_box = next(widget for widget in descendants(window)
                       if isinstance(widget, tk.Text) and widget.get("1.0", "end-1c") == card["evidence"][0]["source_url"])
        viewport.focus_when_visible(url_box)
        window.focus_force()
        self.pump(lambda: self.root.focus_get() is url_box)
        url_box.event_generate("<Tab>")
        self.pump(lambda: self.root.focus_get() is source)
        source.invoke()
        self.assertEqual(self.calls.arguments("open_url"), [(card["evidence"][0]["source_url"],)])
        self.button(window, "Последняя").invoke()
        self.assertIn(card["evidence"][-1]["quote"], "\n".join(self.content(window)))
        self.button(window, "Первая").invoke()
        self.assertIn(card["evidence"][0]["quote"], "\n".join(self.content(window)))

    def test_maximum_claims_and_large_limitations_do_not_create_unbounded_layout(self):
        card = legal_card(1, claims=40, limitations=500)
        window = self.open(card)
        self.assertLess(sum(len(text) for text in self.content(window)), 100000)
        self.root.deiconify()
        viewport = next(widget for widget in window.winfo_children() if isinstance(widget, ScrollViewport))
        self.wait_mapped([window, viewport.canvas])
        # Fonts and native borders differ between Aqua and X11. Bound each
        # long text surface by its real line metrics, not an arbitrary total
        # page pixel height (the same bounded fixture is 6061px on X11).
        for widget in descendants(window):
            if isinstance(widget, tk.Text):
                line_height = font.Font(root=self.root, font=widget.cget("font")).metrics("linespace")
                paragraph_spacing = sum(widget.winfo_pixels(str(widget.cget(option)))
                                        for option in ("spacing1", "spacing3"))
                inset = 2 * sum(widget.winfo_pixels(str(widget.cget(option)))
                                for option in ("borderwidth", "highlightthickness", "pady"))
                self.assertLessEqual(int(widget.cget("height")), 6)
                self.assertLessEqual(widget.winfo_reqheight(), 6 * (line_height + paragraph_spacing) + inset)
            elif isinstance(widget, ttk.Label):
                self.assertLessEqual(len(str(widget.cget("text"))), 2000)
        self.assertLess(len(list(descendants(window))), 180)

    def test_keyboard_paging_and_destroy_release_old_widgets_and_registry(self):
        card = legal_card(41)
        window = self.open(card)
        self.root.deiconify()
        following = self.button(window, "Далее")
        viewport = next(widget for widget in window.winfo_children() if isinstance(widget, ScrollViewport))
        self.wait_mapped([window, viewport.canvas, following])
        # Native activation is asynchronous on macOS. Wait for the window's
        # focus before asking the viewport to pass it to the mapped button.
        window.focus_force()
        self.pump(lambda: self.root.focus_get() is window)
        viewport.focus_when_visible(following)
        self.pump(lambda: self.root.focus_get() is following)
        link = next(widget for widget in descendants(window)
                    if isinstance(widget, ttk.Label) and str(widget.cget("text")) == card["evidence"][0]["source_url"])
        reference = weakref.ref(link)
        following.event_generate("<space>")
        self.pump(lambda: "21–40 из 41" in "\n".join(self.content(window)))
        self.assertFalse(link.winfo_exists())
        del link
        gc.collect()
        self.assertIsNone(reference())
        window.destroy()
        self.assertEqual(len(self.panel.app.child_windows), 0)
        self.assertEqual(self.callback_errors, [])

    def test_source_failure_is_local_retryable_and_late_close_does_not_touch_widgets(self):
        card = legal_card(1, 12000, url_length=4096)
        window = self.open(card)
        source = self.button(window, "Открыть первоисточник")
        source.invoke()
        source.invoke()
        self.assertEqual(len(self.calls.arguments("open_url")), 1)
        self.assertIn("Дождитесь завершения", "\n".join(self.content(window)))
        self.calls.complete("open_url", error=OSError("private-user-directory"))
        text = "\n".join(self.content(window))
        self.assertIn("Не удалось открыть источник", text)
        self.assertNotIn("private-user-directory", text)
        source.invoke()
        self.assertEqual(self.calls.arguments("open_url"), [(card["evidence"][0]["source_url"],)] * 2)
        window.destroy()
        self.calls.complete("open_url", error=OSError("private-user-directory"))
        self.assertEqual(len(self.panel.app.child_windows), 0)
        self.assertEqual(self.callback_errors, [])

    def test_source_admission_rejection_is_visible_and_does_not_block_retry(self):
        card = legal_card(1, 12000, url_length=4096)
        window = self.open(card)
        source = self.button(window, "Открыть первоисточник")
        real_call = self.calls.call
        self.calls.call = lambda *args, **kwargs: False
        source.invoke()
        self.assertIn("Открытие источника сейчас недоступно", "\n".join(self.content(window)))
        self.assertEqual(self.calls.arguments("open_url"), [])
        self.calls.call = real_call
        source.invoke()
        self.assertEqual(self.calls.arguments("open_url"), [(card["evidence"][0]["source_url"],)])
        self.calls.complete("open_url")
        self.assertIn("Источник открыт в браузере", "\n".join(self.content(window)))
