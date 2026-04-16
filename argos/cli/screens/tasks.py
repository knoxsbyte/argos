"""
TasksScreen — task management screen.

Layout:
    ┌─ Task Management ───────────────────────────────────────────────┐
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │ ID    Name              Type    Robot    Status  Prog ETA│   │
    │  │ T-001 Sweep Zone A      sweep   G1-Alpha ACTIVE  85% 2m  │   │
    │  │ T-002 Wipe Surface B    wipe    G1-Beta  ACTIVE  42% 4m  │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  > Add task (natural language): [___________________________]   │
    │                                                                 │
    │  [a] Add  [d] Delete  [Enter] View Detail  [r] Refresh         │
    └─────────────────────────────────────────────────────────────────┘

Status colours: PENDING=silver, ACTIVE=cyan, DONE=green, FAILED=red, PAUSED=yellow
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
)

from argos.cli.theme import CYAN, SILVER, GREEN, YELLOW, RED, DIM, WHITE


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    name: str
    task_type: str
    assigned_robot: str
    status: str            # PENDING / ACTIVE / DONE / FAILED / PAUSED
    progress: float        # 0–100
    eta: str               # human-readable

    @property
    def status_color(self) -> str:
        return {
            "ACTIVE":  CYAN,
            "PENDING": SILVER,
            "DONE":    GREEN,
            "FAILED":  RED,
            "PAUSED":  YELLOW,
        }.get(self.status, WHITE)

    def as_row(self) -> tuple:
        status_text = Text(self.status, style=f"bold {self.status_color}")
        progress_text = Text(f"{self.progress:.0f}%", style=CYAN if self.status == "ACTIVE" else SILVER)
        return (
            self.id,
            self.name,
            self.task_type,
            self.assigned_robot,
            status_text,
            progress_text,
            self.eta,
        )


_INITIAL_TASKS: List[Task] = [
    Task("T-001", "Sweep Zone A",       "sweep",    "G1-Alpha", "ACTIVE",  85.0, "00:02:10"),
    Task("T-002", "Wipe Surface B",     "wipe",     "G1-Beta",  "ACTIVE",  42.0, "00:04:30"),
    Task("T-003", "Vacuum Corridor",    "vacuum",   "—",        "PENDING",  0.0, "—"),
    Task("T-004", "Mop Zone C",         "mop",      "—",        "PENDING",  0.0, "—"),
    Task("T-005", "Sanitise Kitchen",   "sanitise", "G1-Alpha", "DONE",   100.0, "—"),
    Task("T-006", "Dust Shelves",       "dust",     "—",        "FAILED",   22.0, "—"),
]

# Simple NL → task type mapping used by the "add task" parser
_NL_TYPE_MAP = {
    "sweep":    "sweep",
    "vacuum":   "vacuum",
    "mop":      "mop",
    "wipe":     "wipe",
    "clean":    "sweep",
    "sanitise": "sanitise",
    "sanitize": "sanitise",
    "dust":     "dust",
    "scrub":    "scrub",
    "polish":   "polish",
}


def _parse_nl_task(goal: str) -> Task:
    """Parse a natural-language goal string into a Task."""
    words = goal.lower().split()
    task_type = "generic"
    for word in words:
        if word in _NL_TYPE_MAP:
            task_type = _NL_TYPE_MAP[word]
            break

    task_id = f"T-{str(uuid.uuid4())[:6].upper()}"
    name = goal[:40] if len(goal) > 40 else goal
    name = name.capitalize()

    return Task(
        id=task_id,
        name=name,
        task_type=task_type,
        assigned_robot="—",
        status="PENDING",
        progress=0.0,
        eta="—",
    )


# ---------------------------------------------------------------------------
# Detail overlay (simple modal)
# ---------------------------------------------------------------------------

class TaskDetailOverlay(Static):
    """Small overlay showing full task details."""

    DEFAULT_CSS = """
    TaskDetailOverlay {
        background: #16213E;
        border: double #00FFFF;
        padding: 1 2;
        width: 50;
        height: 12;
        offset: 10 5;
        layer: overlay;
    }
    """

    def __init__(self, task: Task, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task = task

    def render(self) -> Text:
        t = self._task
        text = Text()
        text.append("Task Detail\n", style=f"bold {CYAN}")
        text.append("─" * 40 + "\n", style=SILVER)
        text.append("ID:       ", style=SILVER); text.append(t.id + "\n", style=WHITE)
        text.append("Name:     ", style=SILVER); text.append(t.name + "\n", style=WHITE)
        text.append("Type:     ", style=SILVER); text.append(t.task_type + "\n", style=WHITE)
        text.append("Robot:    ", style=SILVER); text.append(t.assigned_robot + "\n", style=WHITE)
        text.append("Status:   ", style=SILVER)
        text.append(t.status + "\n", style=f"bold {t.status_color}")
        text.append("Progress: ", style=SILVER); text.append(f"{t.progress:.0f}%\n", style=CYAN)
        text.append("ETA:      ", style=SILVER); text.append(t.eta + "\n", style=WHITE)
        text.append("\n[Escape] to close", style=DIM)
        return text


# ---------------------------------------------------------------------------
# Tasks screen
# ---------------------------------------------------------------------------

class TasksScreen(Screen):
    """Task management screen with DataTable and NL input."""

    BINDINGS = [
        Binding("a",      "add_task",    "Add Task"),
        Binding("d",      "delete_task", "Delete"),
        Binding("enter",  "view_detail", "Detail", show=True),
        Binding("r",      "refresh",     "Refresh"),
        Binding("escape", "close_detail","Close Detail", show=False),
    ]

    _tasks: List[Task] = list(_INITIAL_TASKS)
    _row_keys: Dict[str, str] = {}   # task_id → DataTable row key
    _detail_open: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="tasks-container"):
            yield Label(" Task Management", classes="panel-title")

            yield DataTable(id="task-table", zebra_stripes=True, cursor_type="row")

            # Summary bar
            yield Static("", id="task-summary")

            # Input row
            with Horizontal(id="task-input-row"):
                yield Input(
                    placeholder="Add task (e.g. 'sweep the kitchen floor'…)",
                    id="task-input",
                )
                yield Button("Add", id="btn-add", variant="primary")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._setup_table()
        self._populate_table()
        self._update_summary()

    def _setup_table(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("ID", "Name", "Type", "Robot", "Status", "Progress", "ETA")

    def _populate_table(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.clear()
        self._row_keys = {}
        for task in self._tasks:
            key = table.add_row(*task.as_row(), key=task.id)
            self._row_keys[task.id] = task.id

    def _update_summary(self) -> None:
        active  = sum(1 for t in self._tasks if t.status == "ACTIVE")
        pending = sum(1 for t in self._tasks if t.status == "PENDING")
        done    = sum(1 for t in self._tasks if t.status == "DONE")
        failed  = sum(1 for t in self._tasks if t.status == "FAILED")

        text = Text()
        text.append(f"  {len(self._tasks)} tasks  ", style=SILVER)
        text.append(f"● {active} active  ",  style=f"bold {CYAN}")
        text.append(f"● {pending} pending  ", style=SILVER)
        text.append(f"● {done} done  ",       style=f"bold {GREEN}")
        text.append(f"● {failed} failed",     style=f"bold {RED}")

        try:
            self.query_one("#task-summary", Static).update(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Button / input handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#btn-add")
    def handle_add_button(self) -> None:
        self._do_add_task()

    @on(Input.Submitted, "#task-input")
    def handle_input_submit(self, event: Input.Submitted) -> None:
        self._do_add_task()

    def _do_add_task(self) -> None:
        inp = self.query_one("#task-input", Input)
        goal = inp.value.strip()
        if not goal:
            return
        task = _parse_nl_task(goal)
        self._tasks.append(task)
        table = self.query_one("#task-table", DataTable)
        table.add_row(*task.as_row(), key=task.id)
        self._row_keys[task.id] = task.id
        inp.value = ""
        self._update_summary()
        self.notify(f"Task added: {task.name}", severity="information")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_add_task(self) -> None:
        self.query_one("#task-input", Input).focus()

    def action_delete_task(self) -> None:
        table = self.query_one("#task-table", DataTable)
        if table.cursor_row < 0 or table.row_count == 0:
            return

        # Get row key at cursor
        row_key, _ = table.coordinate_to_cell_key(
            table.cursor_coordinate
        )
        task_id = str(row_key.value) if row_key.value else None
        if not task_id:
            return

        # Remove from list and table
        self._tasks = [t for t in self._tasks if t.id != task_id]
        try:
            table.remove_row(task_id)
        except Exception:
            pass
        self._update_summary()
        self.notify(f"Task {task_id} deleted", severity="warning")

    def action_view_detail(self) -> None:
        table = self.query_one("#task-table", DataTable)
        if table.row_count == 0:
            return
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            task_id = str(row_key.value) if row_key.value else None
            if not task_id:
                return
            task = next((t for t in self._tasks if t.id == task_id), None)
            if task is None:
                return
            if self._detail_open:
                self.action_close_detail()
            overlay = TaskDetailOverlay(task, id="detail-overlay")
            self.mount(overlay)
            self._detail_open = True
        except Exception:
            pass

    def action_close_detail(self) -> None:
        try:
            self.query_one("#detail-overlay", TaskDetailOverlay).remove()
            self._detail_open = False
        except Exception:
            pass

    def action_refresh(self) -> None:
        self._populate_table()
        self._update_summary()
        self.notify("Task list refreshed", severity="information")
