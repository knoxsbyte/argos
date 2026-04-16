"""
TrainingScreen — training pipeline monitor.

Layout:
    ┌─ Training Pipeline ─────────────────────────────────────────────┐
    │  Model: policy_v2_finetune.pt                                   │
    │                                                                 │
    │  Video Ingestion  ████████████████████░░░░  78%                 │
    │  Preprocessing    █████████████████████████ 100%               │
    │  Training Epochs  ██████░░░░░░░░░░░░░░░░░░  24% (epoch 6/25)   │
    │  Evaluation       ░░░░░░░░░░░░░░░░░░░░░░░░   0%                │
    │                                                                 │
    │  ┌─ Metrics ──────────────────────────────────────────────────┐ │
    │  │  Loss          0.0342    Success Rate  87.3%               │ │
    │  │  Epoch         6/25      ETA           00:14:22            │ │
    │  │  LR            0.0001    GPU Mem       14.2 GB / 24 GB     │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                 │
    │  ┌─ Training Log ─────────────────────────────────────────────┐ │
    │  │  [scrolling stdout output]                                 │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                 │
    │   [Start]  [Pause]  [Cancel]  [Deploy]                         │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    RichLog,
    Static,
)

from argos.cli.theme import CYAN, SILVER, GREEN, YELLOW, RED, DIM, WHITE


# ---------------------------------------------------------------------------
# Training state model
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    model_name: str = "policy_v2_finetune.pt"
    phase: str = "idle"           # idle / ingesting / preprocessing / training / evaluating / done / error

    ingest_pct: float = 0.0
    preprocess_pct: float = 0.0
    train_pct: float = 0.0
    eval_pct: float = 0.0

    epoch: int = 0
    total_epochs: int = 25
    loss: float = 1.0
    success_rate: float = 0.0
    lr: float = 1e-4
    gpu_mem_gb: float = 0.0
    gpu_total_gb: float = 24.0
    eta_seconds: int = 0

    running: bool = False
    paused: bool = False


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class TrainingScreen(Screen):
    """Training pipeline monitor screen."""

    BINDINGS = [
        Binding("s", "start_training",  "Start"),
        Binding("p", "pause_training",  "Pause"),
        Binding("x", "cancel_training", "Cancel"),
        Binding("d", "deploy_model",    "Deploy"),
        Binding("escape", "app.pop_screen", "Back", show=False),
    ]

    _state: TrainingState = TrainingState()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="training-outer"):
            # ── Model name ──────────────────────────────────────────────
            yield Static(
                Text.assemble(
                    ("  Model: ", SILVER),
                    (self._state.model_name, f"bold {CYAN}"),
                ),
                id="model-label",
            )

            # ── Progress bars ────────────────────────────────────────────
            with Container(id="progress-section"):
                yield Label("  Progress", classes="panel-title")

                yield Static(" Video Ingestion   ", id="lbl-ingest",  classes="training-progress-label")
                yield ProgressBar(total=100, show_eta=False, id="pb-ingest")

                yield Static(" Preprocessing     ", id="lbl-preprocess", classes="training-progress-label")
                yield ProgressBar(total=100, show_eta=False, id="pb-preprocess")

                yield Static(" Training Epochs   ", id="lbl-train", classes="training-progress-label")
                yield ProgressBar(total=100, show_eta=False, id="pb-train")

                yield Static(" Evaluation        ", id="lbl-eval",  classes="training-progress-label")
                yield ProgressBar(total=100, show_eta=False, id="pb-eval")

            # ── Metrics table ────────────────────────────────────────────
            with Container(id="metrics-section"):
                yield Label("  Metrics", classes="panel-title")
                yield DataTable(id="metrics-table", show_cursor=False)

            # ── Training log ─────────────────────────────────────────────
            with Container(id="train-log-section"):
                yield Label("  Training Log", classes="panel-title")
                yield RichLog(
                    highlight=False,
                    markup=False,
                    wrap=False,
                    max_lines=200,
                    id="train-log",
                )

            # ── Action buttons ───────────────────────────────────────────
            with Horizontal(id="train-buttons"):
                yield Button("Start",   id="btn-start",  variant="primary")
                yield Button("Pause",   id="btn-pause",  variant="warning")
                yield Button("Cancel",  id="btn-cancel", variant="error")
                yield Button("Deploy",  id="btn-deploy")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._setup_metrics_table()
        self._update_all_widgets()

    def _setup_metrics_table(self) -> None:
        table = self.query_one("#metrics-table", DataTable)
        table.add_columns("Metric", "Value", "Metric", "Value")
        self._populate_metrics()

    def _populate_metrics(self) -> None:
        s = self._state
        eta_str = _fmt_eta(s.eta_seconds)
        table = self.query_one("#metrics-table", DataTable)
        table.clear()
        table.add_row(
            _metric("Loss"),         _value(f"{s.loss:.4f}"),
            _metric("Success Rate"), _value(f"{s.success_rate:.1f}%"),
        )
        table.add_row(
            _metric("Epoch"),        _value(f"{s.epoch}/{s.total_epochs}"),
            _metric("ETA"),          _value(eta_str),
        )
        table.add_row(
            _metric("Learning Rate"), _value(f"{s.lr:.2e}"),
            _metric("GPU Mem"),       _value(f"{s.gpu_mem_gb:.1f} / {s.gpu_total_gb:.0f} GB"),
        )

    def _update_all_widgets(self) -> None:
        s = self._state
        try:
            self.query_one("#pb-ingest",     ProgressBar).update(progress=s.ingest_pct)
            self.query_one("#pb-preprocess", ProgressBar).update(progress=s.preprocess_pct)
            self.query_one("#pb-train",      ProgressBar).update(progress=s.train_pct)
            self.query_one("#pb-eval",       ProgressBar).update(progress=s.eval_pct)
            self._populate_metrics()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#btn-start")
    def handle_start(self) -> None:
        if not self._state.running:
            self._state.running = True
            self._state.paused = False
            self._state.phase = "ingesting"
            self._log("INFO", "Training pipeline started")
            self._run_training_sim()

    @on(Button.Pressed, "#btn-pause")
    def handle_pause(self) -> None:
        self._state.paused = not self._state.paused
        label = "Resume" if self._state.paused else "Pause"
        self.query_one("#btn-pause", Button).label = label
        status = "paused" if self._state.paused else "resumed"
        self._log("WARNING", f"Training {status}")

    @on(Button.Pressed, "#btn-cancel")
    def handle_cancel(self) -> None:
        self._state.running = False
        self._state.paused = False
        self._state.phase = "idle"
        self._log("ERROR", "Training cancelled by user")
        self._update_all_widgets()

    @on(Button.Pressed, "#btn-deploy")
    def handle_deploy(self) -> None:
        if self._state.phase == "done":
            self._log("SUCCESS", f"Deploying {self._state.model_name} to fleet…")
        else:
            self._log("WARNING", "Model not ready for deployment — complete training first")

    # ------------------------------------------------------------------
    # Actions (keyboard)
    # ------------------------------------------------------------------

    def action_start_training(self) -> None:
        self.handle_start()

    def action_pause_training(self) -> None:
        self.handle_pause()

    def action_cancel_training(self) -> None:
        self.handle_cancel()

    def action_deploy_model(self) -> None:
        self.handle_deploy()

    # ------------------------------------------------------------------
    # Background training simulation
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _run_training_sim(self) -> None:
        """Simulate a training pipeline with realistic progression."""
        s = self._state

        # Phase 1 — video ingestion
        s.phase = "ingesting"
        s.gpu_mem_gb = 2.0
        while s.ingest_pct < 100.0 and s.running:
            if not s.paused:
                s.ingest_pct = min(100.0, s.ingest_pct + random.uniform(1.5, 4.0))
                s.gpu_mem_gb = min(s.gpu_total_gb, s.gpu_mem_gb + 0.1)
                self._update_all_widgets()
                self._log("INFO", f"Ingesting video frames… {s.ingest_pct:.0f}%")
            await asyncio.sleep(0.3)

        if not s.running:
            return

        # Phase 2 — preprocessing
        s.phase = "preprocessing"
        while s.preprocess_pct < 100.0 and s.running:
            if not s.paused:
                s.preprocess_pct = min(100.0, s.preprocess_pct + random.uniform(2.0, 5.0))
                self._update_all_widgets()
                self._log("INFO", f"Preprocessing data… {s.preprocess_pct:.0f}%")
            await asyncio.sleep(0.25)

        if not s.running:
            return

        # Phase 3 — training epochs
        s.phase = "training"
        s.gpu_mem_gb = 14.2
        for epoch in range(1, s.total_epochs + 1):
            if not s.running:
                return
            while s.paused:
                await asyncio.sleep(0.5)
            s.epoch = epoch
            s.loss = max(0.001, s.loss * random.uniform(0.88, 0.97))
            s.success_rate = min(99.9, s.success_rate + random.uniform(1.5, 4.0))
            s.train_pct = (epoch / s.total_epochs) * 100
            s.eta_seconds = int((s.total_epochs - epoch) * 1.8)
            self._update_all_widgets()
            self._log(
                "INFO",
                f"Epoch {epoch}/{s.total_epochs}  loss={s.loss:.4f}  "
                f"success_rate={s.success_rate:.1f}%",
            )
            await asyncio.sleep(0.4)

        if not s.running:
            return

        # Phase 4 — evaluation
        s.phase = "evaluating"
        while s.eval_pct < 100.0 and s.running:
            if not s.paused:
                s.eval_pct = min(100.0, s.eval_pct + random.uniform(3.0, 8.0))
                self._update_all_widgets()
                self._log("INFO", f"Evaluating model… {s.eval_pct:.0f}%")
            await asyncio.sleep(0.3)

        s.phase = "done"
        s.running = False
        self._log("SUCCESS", f"Training complete! Final success_rate={s.success_rate:.1f}%  loss={s.loss:.4f}")
        self._update_all_widgets()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        from argos.cli.theme import CYAN, YELLOW, RED, GREEN, DIM, WHITE

        _COLORS = {
            "INFO": CYAN, "WARNING": YELLOW, "ERROR": RED,
            "SUCCESS": GREEN, "DEBUG": DIM,
        }
        _ICONS = {
            "INFO": "ℹ", "WARNING": "⚠", "ERROR": "✖",
            "SUCCESS": "✔", "DEBUG": "·",
        }
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        color = _COLORS.get(level, WHITE)
        icon = _ICONS.get(level, " ")

        text = Text()
        text.append(ts, style=DIM)
        text.append(f" {icon} {level:<8}", style=f"bold {color}")
        text.append(message, style=color if level in ("ERROR", "SUCCESS") else WHITE)

        try:
            self.query_one("#train-log", RichLog).write(text)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric(label: str) -> Text:
    return Text(label, style=SILVER)


def _value(val: str) -> Text:
    return Text(val, style=f"bold {CYAN}")


def _fmt_eta(seconds: int) -> str:
    if seconds <= 0:
        return "—"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
