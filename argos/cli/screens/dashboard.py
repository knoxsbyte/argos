"""
DashboardScreen — main 2×2 grid view.

Layout:
    ┌─ ARGOS ────────────────── v0.1.0 ── N robots online ──────────┐
    │ ┌─ Fleet Status ──────────┐  ┌─ Swarm Map ───────────────────┐ │
    │ │  RobotCard G1-Alpha     │  │  ASCII map with robot labels  │ │
    │ │  RobotCard G1-Beta      │  │                               │ │
    │ └─────────────────────────┘  └───────────────────────────────┘ │
    │ ┌─ Task Queue ────────────┐  ┌─ Event Log ───────────────────┐ │
    │ │  DataTable of tasks     │  │  LogPanel scrolling log       │ │
    │ └─────────────────────────┘  └───────────────────────────────┘ │
    │ [q] Quit  [t] Tasks  [r] Training  [s] Sim  [?] Help          │
    └────────────────────────────────────────────────────────────────┘

Keyboard bindings (local):
    r  — refresh robot list
    c  — clear log
    /  — focus search in log
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
)

from argos.cli.theme import CYAN, SILVER, GREEN, YELLOW, RED, DIM, WHITE, ARGOS_CSS
from argos.cli.widgets.robot_card import RobotCard, RobotState
from argos.cli.widgets.swarm_map import SwarmMap, MapState, _default_map_state
from argos.cli.widgets.log_panel import LogPanel


# ---------------------------------------------------------------------------
# Fake task data (replaced by real coordinator when available)
# ---------------------------------------------------------------------------

_DEMO_TASKS = [
    ("T-001", "Sweep Zone A",       "sweep",   "G1-Alpha", "ACTIVE",  "85%",  "00:02:10"),
    ("T-002", "Wipe Surface B",     "wipe",    "G1-Beta",  "ACTIVE",  "42%",  "00:04:30"),
    ("T-003", "Vacuum Corridor",    "vacuum",  "—",        "PENDING", "0%",   "—"),
    ("T-004", "Mop Zone C",         "mop",     "—",        "PENDING", "0%",   "—"),
    ("T-005", "Sanitise Kitchen",   "sanitise","G1-Alpha", "DONE",    "100%", "—"),
]

_TASK_STATUS_COLORS = {
    "ACTIVE":   CYAN,
    "PENDING":  SILVER,
    "DONE":     GREEN,
    "FAILED":   RED,
    "PAUSED":   YELLOW,
}


# ---------------------------------------------------------------------------
# Dashboard screen
# ---------------------------------------------------------------------------

class DashboardScreen(Screen):
    """Main ARGOS dashboard with 2×2 panel layout."""

    BINDINGS = [
        Binding("r", "refresh_robots", "Refresh"),
        Binding("c", "clear_log",      "Clear Log"),
        Binding("slash", "search_log", "Search Log"),
        Binding("escape", "app.focus_next", "Unfocus", show=False),
    ]

    # Demo robot states — replaced by live data in production
    _demo_states: List[RobotState] = [
        RobotState(
            name="G1-Alpha",
            status="cleaning",
            task="sweep_zone_a",
            battery=82.0,
            zone="A",
            pos_x=2.1,
            pos_y=3.4,
            uptime_seconds=5025,
        ),
        RobotState(
            name="G1-Beta",
            status="idle",
            task="—",
            battery=55.0,
            zone="B",
            pos_x=6.8,
            pos_y=1.2,
            uptime_seconds=3600,
        ),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="dashboard-grid"):
            # ── Fleet Status (top-left) ──────────────────────────────────
            with ScrollableContainer(id="fleet-panel"):
                yield Label(" Fleet Status", classes="panel-title")
                for state in self._demo_states:
                    yield RobotCard(state=state, id=f"card-{state.name.lower().replace('-', '_')}")

            # ── Swarm Map (top-right) ────────────────────────────────────
            with Container(id="map-panel"):
                yield Label(" Swarm Map", classes="panel-title")
                yield SwarmMap(map_state=_default_map_state(), id="swarm-map")

            # ── Task Queue (bottom-left) ─────────────────────────────────
            with Container(id="task-panel"):
                yield Label(" Task Queue", classes="panel-title")
                yield DataTable(id="task-table", zebra_stripes=True)

            # ── Event Log (bottom-right) ─────────────────────────────────
            with Container(id="log-panel"):
                yield Label(" Event Log", classes="panel-title")
                yield LogPanel(id="event-log")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._setup_task_table()
        self._populate_tasks()
        self._start_sim_loop()

    def _setup_task_table(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("ID", "Name", "Type", "Robot", "Status", "Progress", "ETA")
        table.cursor_type = "row"

    def _populate_tasks(self) -> None:
        table = self.query_one("#task-table", DataTable)
        for row in _DEMO_TASKS:
            task_id, name, ttype, robot, status, progress, eta = row
            color = _TASK_STATUS_COLORS.get(status, WHITE)
            colored_status = Text(status, style=f"bold {color}")
            table.add_row(task_id, name, ttype, robot, colored_status, progress, eta)

    # ------------------------------------------------------------------
    # Background simulation — updates robot cards and map periodically
    # ------------------------------------------------------------------

    @work(exclusive=False)
    async def _start_sim_loop(self) -> None:
        """Lightweight sim: tick robot states every 2 s."""
        import math, random

        t = 0
        log: LogPanel = self.query_one("#event-log", LogPanel)
        map_widget: SwarmMap = self.query_one("#swarm-map", SwarmMap)

        while True:
            await asyncio.sleep(2.0)
            t += 1

            # Update robot positions (sinusoidal wander)
            for i, state in enumerate(self._demo_states):
                state.uptime_seconds += 2
                state.battery = max(0.0, state.battery - random.uniform(0.0, 0.3))

                # Wander within zone grid
                state.pos_x = round(state.pos_x + random.uniform(-0.2, 0.2), 1)
                state.pos_y = round(state.pos_y + random.uniform(-0.2, 0.2), 1)

                # Map col/row (scale to 40×20 grid — rough mapping)
                col = 2 + (state.pos_x / 10.0) * 16 + (i * 20)
                row = 2 + (state.pos_y / 10.0) * 7

                try:
                    card_id = f"#card-{state.name.lower().replace('-', '_')}"
                    self.query_one(card_id, RobotCard).update_state(state)
                except Exception:
                    pass

            # Refresh map
            ms = _default_map_state()
            for i, state in enumerate(self._demo_states):
                col = 2 + (state.pos_x / 10.0) * 16 + (i * 20)
                row = 2 + (state.pos_y / 10.0) * 7
                ms.robots[i].col = max(2, min(ms.cols - 3, col))
                ms.robots[i].row = max(1, min(ms.rows - 2, row))

            # Increase zone cleaning progress slowly
            for zone in ms.zones:
                zone.cleaned_pct = min(100.0, zone.cleaned_pct + random.uniform(0, 0.5))

            try:
                map_widget.update_map(ms)
            except Exception:
                pass

            # Occasional log events
            if t % 3 == 0:
                msgs = [
                    ("INFO",    "G1-Alpha", "Cleaning path segment updated"),
                    ("INFO",    "G1-Beta",  "Idle — awaiting task assignment"),
                    ("WARNING", "G1-Beta",  f"Battery at {self._demo_states[1].battery:.0f}%"),
                    ("SUCCESS", "G1-Alpha", "Segment sweep_a3 completed"),
                ]
                import random as _r
                ts, src, msg = _r.choice(msgs)
                try:
                    log.add_entry(ts, msg, source=src)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_refresh_robots(self) -> None:
        log = self.query_one("#event-log", LogPanel)
        log.add_entry("INFO", "Manual refresh triggered")

    def action_clear_log(self) -> None:
        self.query_one("#event-log", LogPanel).clear()

    def action_search_log(self) -> None:
        # In a full implementation this would open a search input overlay
        self.query_one("#event-log", LogPanel).search("")
