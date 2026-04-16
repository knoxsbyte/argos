"""
ArgosApp — main Textual application class.

Screens:
  DashboardScreen  — default, shows fleet + map + task queue + log
  TasksScreen      — full task management
  TrainingScreen   — training pipeline monitor

Global bindings:
  q  quit
  d  dashboard
  t  tasks
  r  training / robots
  ?  help (placeholder)
"""

from __future__ import annotations

import asyncio
from typing import ClassVar, List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from argos.cli.theme import ARGOS_CSS
from argos.cli.screens.dashboard import DashboardScreen
from argos.cli.screens.training import TrainingScreen
from argos.cli.screens.tasks import TasksScreen


class ArgosApp(App):
    """ARGOS Textual TUI application."""

    CSS = ARGOS_CSS

    TITLE = "ARGOS — Autonomous Robot Group Operations System"
    SUB_TITLE = "v0.1.0"

    # Global key bindings available on every screen
    BINDINGS: ClassVar[List[Binding]] = [
        Binding("q",         "quit",      "Quit",      priority=True),
        Binding("d",         "dashboard", "Dashboard"),
        Binding("t",         "tasks",     "Tasks"),
        Binding("r",         "training",  "Training"),
        Binding("question_mark", "help_screen", "Help", show=True),
    ]

    # Screen registry — Textual will instantiate lazily
    SCREENS = {
        "dashboard": DashboardScreen,
        "tasks":     TasksScreen,
        "training":  TrainingScreen,
    }

    def on_mount(self) -> None:
        """Show dashboard on startup and begin background polling."""
        self.push_screen("dashboard")

    # ------------------------------------------------------------------
    # Screen actions
    # ------------------------------------------------------------------

    def action_dashboard(self) -> None:
        """Switch to (or reveal) the dashboard screen."""
        # Pop all screens until we're at the root, then push dashboard
        while len(self.screen_stack) > 1:
            self.pop_screen()
        if not isinstance(self.screen, DashboardScreen):
            self.push_screen("dashboard")

    def action_tasks(self) -> None:
        """Switch to the task management screen."""
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen("tasks")

    def action_training(self) -> None:
        """Switch to the training pipeline screen."""
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.push_screen("training")

    def action_help_screen(self) -> None:
        """Show a brief help notification."""
        self.notify(
            "Bindings: [q] Quit  [d] Dashboard  [t] Tasks  [r] Training  "
            "[?] Help\n"
            "Dashboard: [r] Refresh  [c] Clear log\n"
            "Tasks: [a] Add  [d] Delete  [Enter] Detail\n"
            "Training: [s] Start  [p] Pause  [x] Cancel  [Deploy]",
            title="ARGOS Help",
            severity="information",
            timeout=8,
        )
