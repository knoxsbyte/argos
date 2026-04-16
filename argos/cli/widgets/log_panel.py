"""
LogPanel widget — scrolling, colour-coded event log.

Log levels and colours:
  INFO     → Cyan   #00FFFF
  WARNING  → Yellow #FFD700
  ERROR    → Red    #FF4444
  SUCCESS  → Green  #00FF88
  DEBUG    → Dim    #888888

Max 500 lines; auto-scrolls to bottom unless user is scrolling up.
Supports basic search filtering via search() method.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.widget import Widget
from textual.widgets import RichLog, Static

from argos.cli.theme import CYAN, YELLOW, RED, GREEN, DIM, SILVER, WHITE


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MAX_LINES = 500

LEVEL_COLORS = {
    "INFO": CYAN,
    "WARNING": YELLOW,
    "WARN": YELLOW,
    "ERROR": RED,
    "CRITICAL": RED,
    "SUCCESS": GREEN,
    "DONE": GREEN,
    "DEBUG": DIM,
}

LEVEL_ICONS = {
    "INFO": "ℹ",
    "WARNING": "⚠",
    "WARN": "⚠",
    "ERROR": "✖",
    "CRITICAL": "✖",
    "SUCCESS": "✔",
    "DONE": "✔",
    "DEBUG": "·",
}


def _level_color(level: str) -> str:
    return LEVEL_COLORS.get(level.upper(), WHITE)


def _level_icon(level: str) -> str:
    return LEVEL_ICONS.get(level.upper(), " ")


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class LogPanel(Widget):
    """
    Scrolling log panel backed by a RichLog widget.

    Usage::

        log = LogPanel()
        log.add_entry("INFO", "Robot G1-Alpha started task wipe_surface")
        log.add_entry("WARNING", "Battery below 20% on G1-Beta")
        log.add_entry("ERROR", "Connection lost to G1-Beta")
        log.add_entry("SUCCESS", "Task sweep_zone_A completed")
    """

    DEFAULT_CSS = """
    LogPanel {
        background: #16213E;
        border: solid #C0C0C0;
        height: 1fr;
    }

    LogPanel RichLog {
        background: #16213E;
        color: #E0E0E0;
        scrollbar-color: #C0C0C0;
        scrollbar-background: #0F3460;
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: Deque[tuple] = deque(maxlen=MAX_LINES)
        self._filter: str = ""
        self._rich_log: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        rl = RichLog(highlight=False, markup=False, wrap=False, max_lines=MAX_LINES)
        rl.id = "inner-log"
        yield rl

    def on_mount(self) -> None:
        self._rich_log = self.query_one("#inner-log", RichLog)
        # Seed with a few startup messages
        self.add_entry("INFO", "ARGOS TUI started")
        self.add_entry("INFO", "Awaiting robot connections…")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, level: str, message: str, source: str = "") -> None:
        """Add a log entry. Thread-safe via call_from_thread if needed."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = (ts, level.upper(), source, message)
        self._entries.append(entry)

        if self._filter and self._filter.lower() not in message.lower():
            return

        self._write_entry(entry)

    def add_robot_event(self, robot_name: str, level: str, message: str) -> None:
        """Convenience: log an event attributed to a named robot."""
        self.add_entry(level, message, source=robot_name)

    def clear(self) -> None:
        """Clear all log entries."""
        self._entries.clear()
        if self._rich_log:
            self._rich_log.clear()

    def search(self, query: str) -> None:
        """
        Filter displayed entries to those matching *query*.
        Pass empty string to clear filter.
        """
        self._filter = query
        if self._rich_log is None:
            return
        self._rich_log.clear()
        for entry in self._entries:
            ts, level, source, message = entry
            if not query or query.lower() in message.lower():
                self._write_entry(entry)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_entry(self, entry: tuple) -> None:
        if self._rich_log is None:
            return
        ts, level, source, message = entry
        color = _level_color(level)
        icon = _level_icon(level)

        text = Text()
        text.append(ts, style=DIM)
        text.append(" ")
        text.append(f"{icon} {level:<8}", style=f"bold {color}")
        if source:
            text.append(f"[{source}] ", style=SILVER)
        text.append(message, style=color if level in ("ERROR", "CRITICAL", "SUCCESS", "DONE") else WHITE)

        self._rich_log.write(text)
