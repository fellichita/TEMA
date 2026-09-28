"""Real Tk controls for the local multisource import and review workflow."""

from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

from app.ui.pilot_signals import SignalsWindow
from app.ui.theme import apply_theme
from tests.ui.test_desktop import TkCase
from tests.ui.test_pilot_panel import DeferredController


class SignalWindowTests(TkCase):
    def setUp(self):
        super().setUp()
        apply_theme(self.root)
        self.fixture_controller = DeferredController()
        self.window = SignalsWindow(SimpleNamespace(root=self.root, controller=self.fixture_controller))
        self.root.deiconify()
        self.root.update()

    def tearDown(self):
        self.window.window.destroy()
        super().tearDown()

    def test_wordstat_import_requires_confirmed_identity_and_rights_then_starts_job(self):
        self.window.path.set("/tmp/wordstat.csv")
        self.window._import()
        self.assertIn("Сначала", self.window.message.get())
        self.window.query.set("Молекулярная память")
        self.window.phrase.set("ДНК память")
        self.window.definition.insert("1.0", "Запись цифровых данных в молекулы")
        self.window._confirm_query()
        self.fixture_controller.complete("pilot_create_signal_query", {"query_profile_hash": "a" * 64,
            "concept_hash": "b" * 64, "concept_id": "id", "phrase": "ДНК память"})
        self.window._import()
        self.assertIn("право", self.window.message.get())
        self.window.rights.set(True)
        self.window.mapping["expected_from"].set("2025-01")
        self.window.mapping["expected_to"].set("2026-08")
        self.window._import()
        method, args, kwargs = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_import_signal_csv")
        self.assertEqual(args[2], "wordstat")
        self.assertEqual(kwargs["retention_confirmed"], True)
        self.assertEqual(args[3]["phrase"], "ДНК память")
        self.fixture_controller.complete("pilot_import_signal_csv", {"receipt_hash": "c" * 64,
            "source": "wordstat", "accepted": 20, "rejected": 0, "snapshot_hash": "d" * 64})
        self.window._start()
        method, args, kwargs = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_start_signals")
        self.assertEqual(args[0], "a" * 64)
        self.assertEqual(kwargs["wordstat_receipt_hash"], "c" * 64)

    def test_saved_profile_displays_label_and_evidence_without_recomputing(self):
        finding = {"finding_id": "f1", "concept_id": "concept-1", "queue": "attention",
                   "origins": ["user", "wordstat"], "search_state": "sustained_growth",
                   "funding_state": None, "scientific_category": None,
                   "explanation": "Устойчивый нормированный рост", "next_check": "Проверить исследования",
                   "rule_id": "normalized_search_growth", "observation_hashes": ["a" * 64]}
        profile = {"query_profile_hash": "b" * 64, "concept_artifact_hashes": ["c" * 64],
                   "findings": [finding], "attention_ids": ["f1"], "watch_ids": []}
        self.window._render({"profile": profile, "view_id": "run-1", "imports": {"wordstat": "d" * 64},
                             "concepts": {"concept-1": "ДНК память"}, "confirmed_association_hashes": ()})
        self.assertEqual(self.window.cards.item("f1", "values")[1], "ДНК память")
        self.window.cards.selection_set("f1")
        self.window._selected()
        self.assertIn("Устойчивый нормированный рост", self.window.detail.get("1.0", "end"))
        self.assertEqual(self.window.receipts["wordstat"], "d" * 64)
        self.assertEqual(self.fixture_controller.calls[-1][0], "pilot_signal_watch_state")
        self.fixture_controller.complete("pilot_signal_watch_state", {"concept_id": "concept-1", "watched": False})
        self.window.watch_button.invoke()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_set_signal_watch")
        self.assertEqual(args, ("run-1", "concept-1", True))
        self.fixture_controller.complete("pilot_set_signal_watch", {"concept_id": "concept-1", "watched": True})
        self.assertEqual(self.window.watch_button.cget("text"), "Убрать из наблюдения")

    def test_largest_event_scenario_uses_selected_card_and_explicit_tie_choice(self):
        finding = {"finding_id": "f1", "concept_id": "11111111-1111-4111-8111-111111111111",
                   "queue": "attention", "origins": ["cordis"], "search_state": None,
                   "funding_state": "multiple_units", "scientific_category": None,
                   "explanation": "Два гранта", "next_check": "Проверить проекты",
                   "rule_id": "confirmed_funding_units", "observation_hashes": []}
        profile = {"query_profile_hash": "b" * 64, "concept_artifact_hashes": ["c" * 64],
                   "findings": [finding], "attention_ids": ["f1"], "watch_ids": []}
        self.window._render({"profile": profile, "view_id": "run-1", "imports": {},
                             "concepts": {finding["concept_id"]: "ДНК память"},
                             "confirmed_association_hashes": ()})
        self.window.cards.selection_set("f1")
        self.window.scenario.set("Без крупнейшего события")
        self.window._scenario()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_signal_largest_event_options")
        self.assertEqual(args, ("run-1", finding["concept_id"]))
        self.fixture_controller.complete(method, ({"event_hash": "a" * 64, "event_id": "e1",
            "source_event_id": "101", "event_kind": "grant_project", "currency": "EUR", "amount": "2000000"},))
        dialogs = [item for item in self.window.window.winfo_children() if isinstance(item, tk.Toplevel)]
        self.assertEqual(len(dialogs), 1)
        button = next(item for item in dialogs[0].winfo_children() if isinstance(item, ttk.Button))
        button.invoke()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_signal_scenario")
        self.assertEqual(args, ("run-1", "exclude_largest_disclosed_event", finding["concept_id"],
                                "grant_project", "EUR", "a" * 64))

    def test_result_actions_remain_reachable_at_200_percent_and_minimum_width(self):
        self.window.window.destroy()
        self.root.tk.call("tk", "scaling", 2.0)
        apply_theme(self.root)
        self.window = SignalsWindow(SimpleNamespace(root=self.root, controller=self.fixture_controller))
        self.window.window.geometry("680x520")
        self.window.notebook.select(self.window.result_page)
        self.root.update()
        viewport = self.window.result_page
        viewport.canvas.yview_moveto(1.0)
        self.root.update()
        bottom = viewport.canvas.winfo_rooty() + viewport.canvas.winfo_height()
        self.assertLessEqual(self.window.watch_button.winfo_rooty() +
                             self.window.watch_button.winfo_height(), bottom + 2)

    def test_explicit_scientific_card_link_can_start_without_an_import(self):
        self.window._query_ready({"query_profile_hash": "a" * 64, "concept_hash": "b" * 64,
                                  "concept_id": "concept-1", "phrase": "ДНК память"})
        self.window._choose_science()
        self.assertEqual(self.fixture_controller.calls[-1][0], "pilot_signal_scientific_runs")
        self.fixture_controller.complete("pilot_signal_scientific_runs", [{
            "run_id": "run-one", "query": "ДНК память", "created_at": "2026-09-19T12:00:00Z", "imported": False}])
        dialogs = [item for item in self.window.window.winfo_children() if isinstance(item, tk.Toplevel)]
        self.assertEqual(len(dialogs), 1)
        dialog = dialogs[0]
        pickers = [item for item in dialog.winfo_children() if isinstance(item, ttk.Combobox)]
        pickers[0].current(0)
        pickers[0].event_generate("<<ComboboxSelected>>")
        self.assertEqual(self.fixture_controller.calls[-1][0], "pilot_signal_scientific_cards")
        self.fixture_controller.complete("pilot_signal_scientific_cards", [{
            "candidate_id": "candidate-1", "label": "ДНК память", "category": "weak_signal_candidate",
            "definition": "Хранение данных в молекулах"}])
        pickers[1].current(0)
        confirm = next(item for item in dialog.winfo_children() if isinstance(item, ttk.Button))
        confirm.invoke()
        self.window._start()
        method, _, kwargs = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_start_signals")
        self.assertEqual(kwargs["base_result_run_id"], "run-one")
        self.assertEqual(kwargs["scientific_links"], ({"concept_id": "concept-1",
                                                      "candidate_id": "candidate-1"},))
