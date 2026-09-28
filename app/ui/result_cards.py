"""Bounded native result cards; opening sections never rebuilds this view."""

from tkinter import ttk
from urllib.parse import urlsplit

from app.ui.display import px
from app.ui.pilot_passport import GATE_NAMES


class ResultCards:
    PAGE_SIZE = 10

    def __init__(self, panel, parent):
        self.panel = panel
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", padx=16)
        self.items = ttk.Frame(self.frame)
        self.items.pack(fill="x")
        self.pager = ttk.Frame(self.frame)
        self.previous = ttk.Button(self.pager, text="Назад", command=lambda: self.move(-1))
        self.previous.pack(side="left")
        self.position = ttk.Label(self.pager)
        self.position.pack(side="left", padx=8)
        self.following = ttk.Button(self.pager, text="Далее", command=lambda: self.move(1))
        self.following.pack(side="left")
        self.cards, self.assessments, self.labels, self.expanded = {}, {}, [], set()
        self.offset = 0
        self.frame.bind("<Configure>", self.resize)

    def clear(self):
        for child in self.items.winfo_children():
            child.destroy()
        self.labels.clear()
        self.expanded.clear()
        self.cards.clear()
        self.pager.pack_forget()

    def render(self, cards, assessments):
        self.clear()
        self.cards, self.assessments = dict(cards), assessments
        self.offset = 0
        self.draw()

    def move(self, delta):
        self.offset = max(0, min(self.offset + delta * self.PAGE_SIZE,
                                max(0, (len(self.cards) - 1) // self.PAGE_SIZE * self.PAGE_SIZE)))
        self.draw()

    def label(self, parent, text, *, style="TLabel"):
        label = ttk.Label(parent, text=text, style=style, wraplength=500, width=1, justify="left")
        label.pack(fill="x", pady=4)
        self.labels.append(label)
        return label

    def select(self, identifier):
        tree = self.panel.tree if identifier in self.panel.tree.get_children() else self.panel.other_tree
        self.panel.result_tabs.select(0 if tree is self.panel.tree else 1)
        tree.selection_set(identifier)
        tree.focus(identifier)
        self.panel._selection()

    def draw(self):
        from app.ui.pilot_panel import category_label
        top_ranks = {identifier: rank for rank, identifier in enumerate(self.panel.tree.get_children(), 1)}
        for child in self.items.winfo_children():
            child.destroy()
        self.labels.clear()
        self._wrap_width = None
        for identifier in list(self.cards)[self.offset:self.offset + self.PAGE_SIZE]:
            card = self.cards[identifier]
            assessment = self.assessments.get(identifier, {})
            frame = ttk.Frame(self.items, padding=16, style="Card.TFrame")
            frame.pack(fill="x", pady=(8, 4))
            self.label(frame, card["candidate"]["label"], style="CardTitle.TLabel")
            self.label(frame, card["candidate"]["definition"])
            position = f"TOP {top_ranks[identifier]}" if identifier in top_ranks else "Вне TOP"
            self.label(frame, position + " · " + category_label(card), style="Muted.TLabel")
            supported = [claim["text"] for claim in card.get("claims", [])
                         if claim["role"] in {"case", "advantage"} and claim["support"] == "supported"]
            reason = supported[0] if supported else "Основания требуют проверки по источникам."
            self.label(frame, "Основание: " + (reason[:280] + "…" if len(reason) > 280 else reason))
            urls = list(dict.fromkeys(item["source_url"] for item in card.get("evidence", []) if item.get("source_url")))
            self.label(frame, "Источники" if urls else "Источники не указаны", style="Muted.TLabel")
            for number, url in enumerate(urls[:3], 1):
                link = self.label(frame, f"{number}. {urlsplit(url).netloc}", style="Muted.TLabel")
                link.configure(cursor="hand2", takefocus=True)
                def open_source(_=None, url=url):
                    self.panel.app.controller.call("pilot-card-source", "open_url", lambda _: None, self.panel.error, url)
                for event in ("<Button-1>", "<Return>", "<space>"):
                    link.bind(event, open_source)
            details = ttk.Frame(frame)
            self.label(details, "Исследований за последние три полных года: " + str(assessment.get("recent_studies", "не проверено")))
            for warning in card.get("limitations", []):
                self.label(details, warning, style="Muted.TLabel")
            for gate in assessment.get("gate_failures", []):
                self.label(details, GATE_NAMES.get(gate, gate), style="Muted.TLabel")
            def passport(identifier=identifier):
                self.select(identifier)
                self.panel.passport()
            ttk.Button(details, text="Полный паспорт и доказательства", command=passport).pack(anchor="w", pady=8)
            button = ttk.Button(frame, text="Показатели и ограничения  ›")
            button.pack(anchor="w", pady=(8, 0))
            def toggle(identifier=identifier, details=details, button=button):
                self.select(identifier)
                if identifier in self.expanded:
                    self.expanded.remove(identifier)
                    details.pack_forget()
                    button.configure(text="Показатели и ограничения  ›")
                else:
                    self.expanded.add(identifier)
                    details.pack(fill="x", before=button)
                    button.configure(text="Скрыть подробности  ‹")
            button.configure(command=toggle)
            if identifier in self.expanded:
                details.pack(fill="x", before=button)
                button.configure(text="Скрыть подробности  ‹")
        if len(self.cards) > self.PAGE_SIZE:
            self.pager.pack(fill="x", pady=8)
            self.previous.state(["disabled"] if not self.offset else ["!disabled"])
            self.following.state(["disabled"] if self.offset + self.PAGE_SIZE >= len(self.cards) else ["!disabled"])
            self.position.configure(text=f"{self.offset + 1}–{min(len(self.cards), self.offset + self.PAGE_SIZE)} из {len(self.cards)}")
        self.resize()

    def resize(self, _=None):
        width = max(px(self.panel.app.root, 120), self.frame.winfo_width() - px(self.panel.app.root, 40))
        if getattr(self, "_wrap_width", None) == width:
            return
        self._wrap_width = width
        for label in self.labels:
            label.configure(wraplength=width)
