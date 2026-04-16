"""
RobotCard widget — displays one G1 robot's live status.

Visual layout (7 rows):
    ┌─ G1-Alpha ●─────────────────┐
    │ Status:  CLEANING           │
    │ Task:    wipe_surface       │
    │ Battery: ████████░░ 82%     │
    │ Zone: A   Pos: (2.1, 3.4)  │
    │ Uptime: 01:23:45            │
    └─────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from argos.cli.theme import (
    CYAN, SILVER, GREEN, YELLOW, RED, DIM, WHITE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """Snapshot of a single robot's state."""

    name: str = "G1-Unknown"
    status: str = "offline"          # active / cleaning / idle / error / offline
    task: str = "—"
    battery: float = 0.0             # 0–100
    zone: str = "?"
    pos_x: float = 0.0
    pos_y: float = 0.0
    uptime_seconds: int = 0
    error_msg: str = ""

    # ---------- helpers ----------

    @property
    def status_color(self) -> str:
        s = self.status.lower()
        if s in ("active", "cleaning", "ok"):
            return GREEN
        if s in ("idle", "warning"):
            return YELLOW
        if s in ("error", "critical"):
            return RED
        return DIM  # offline

    @property
    def status_dot(self) -> str:
        s = self.status.lower()
        if s in ("active", "cleaning", "ok"):
            return "●"
        if s in ("idle", "warning"):
            return "◉"
        if s in ("error", "critical"):
            return "✖"
        return "○"

    @property
    def battery_bar(self) -> Text:
        """Render a 10-char filled/empty progress bar with colour."""
        pct = max(0.0, min(100.0, self.battery))
        filled = round(pct / 10)
        empty = 10 - filled

        if pct >= 50:
            bar_color = GREEN
        elif pct >= 20:
            bar_color = YELLOW
        else:
            bar_color = RED

        bar_str = "█" * filled + "░" * empty
        t = Text()
        t.append(bar_str, style=f"bold {bar_color}")
        t.append(f" {pct:.0f}%", style=WHITE)
        return t

    @property
    def uptime_str(self) -> str:
        h = self.uptime_seconds // 3600
        m = (self.uptime_seconds % 3600) // 60
        s = self.uptime_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class RobotCard(Widget):
    """Textual widget that renders a single robot status card."""

    DEFAULT_CSS = """
    RobotCard {
        height: 7;
        border: solid #C0C0C0;
        background: #0F3460;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    RobotCard:hover { border: solid #00FFFF; }
    RobotCard.selected { border: double #00FFFF; }
    """

    # Reactive state — updating triggers re-render
    state: reactive[RobotState] = reactive(RobotState, layout=True)

    def __init__(self, state: RobotState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="rc-line-name")
        yield Static("", id="rc-line-status")
        yield Static("", id="rc-line-task")
        yield Static("", id="rc-line-battery")
        yield Static("", id="rc-line-pos")
        yield Static("", id="rc-line-uptime")

    def on_mount(self) -> None:
        self._refresh_all()

    # ------------------------------------------------------------------
    # Reactive watcher — called whenever self.state changes
    # ------------------------------------------------------------------

    def watch_state(self, state: RobotState) -> None:
        self._refresh_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        s = self.state

        # --- name line ---
        name_text = Text()
        name_text.append(f" {s.name} ", style=f"bold {CYAN}")
        name_text.append(s.status_dot, style=s.status_color)
        try:
            self.query_one("#rc-line-name", Static).update(name_text)
        except Exception:
            pass

        # --- status line ---
        status_text = Text()
        status_text.append(" Status: ", style=SILVER)
        status_text.append(s.status.upper(), style=f"bold {s.status_color}")
        if s.error_msg:
            status_text.append(f"  ({s.error_msg})", style=RED)
        try:
            self.query_one("#rc-line-status", Static).update(status_text)
        except Exception:
            pass

        # --- task line ---
        task_text = Text()
        task_text.append(" Task:   ", style=SILVER)
        task_text.append(s.task, style=WHITE)
        try:
            self.query_one("#rc-line-task", Static).update(task_text)
        except Exception:
            pass

        # --- battery line ---
        bat_text = Text()
        bat_text.append(" Battery: ", style=SILVER)
        bat_text.append_text(s.battery_bar)
        try:
            self.query_one("#rc-line-battery", Static).update(bat_text)
        except Exception:
            pass

        # --- position / zone line ---
        pos_text = Text()
        pos_text.append(" Zone: ", style=SILVER)
        pos_text.append(s.zone, style=CYAN)
        pos_text.append("   Pos: ", style=SILVER)
        pos_text.append(f"({s.pos_x:.1f}, {s.pos_y:.1f})", style=WHITE)
        try:
            self.query_one("#rc-line-pos", Static).update(pos_text)
        except Exception:
            pass

        # --- uptime line ---
        uptime_text = Text()
        uptime_text.append(" Uptime: ", style=SILVER)
        uptime_text.append(s.uptime_str, style=DIM)
        try:
            self.query_one("#rc-line-uptime", Static).update(uptime_text)
        except Exception:
            pass

    def update_state(self, new_state: RobotState) -> None:
        """Public API for external callers to push new state."""
        self.state = new_state
