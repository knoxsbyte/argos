"""
SwarmMap widget — 2D ASCII room map with robot positions and zone shading.

Unicode block characters used:
  ░  (U+2591) light shade  → uncleaned zone
  ▒  (U+2592) medium shade → partially cleaned
  █  (U+2588) full block   → wall / obstacle
  ·  (U+00B7) middle dot   → cleaned floor
  space                    → outside room boundary

Robot labels shown as [A], [B], … in cyan, overlaid on zone tiles.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from argos.cli.theme import CYAN, SILVER, GREEN, YELLOW, RED, DIM, WHITE, PANEL


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """Rectangular cleaning zone."""
    id: str
    label: str
    x: int          # grid column (top-left)
    y: int          # grid row (top-left)
    width: int
    height: int
    cleaned_pct: float = 0.0   # 0–100


@dataclass
class RobotPos:
    """Robot position on the map grid."""
    name: str          # full name e.g. "G1-Alpha"
    label: str         # single letter e.g. "A"
    col: float         # column (float for sub-cell precision)
    row: float         # row
    status: str = "idle"


@dataclass
class MapState:
    """Complete map state snapshot."""
    rows: int = 20
    cols: int = 40
    zones: List[Zone] = field(default_factory=list)
    robots: List[RobotPos] = field(default_factory=list)
    walls: List[Tuple[int, int]] = field(default_factory=list)  # (col, row)
    room_name: str = "Room A"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

_UNCLEANED = "░"
_PARTIAL = "▒"
_CLEANED = "·"
_WALL = "█"
_OUTSIDE = " "


def _render_map(state: MapState) -> Text:
    """Build a Rich Text grid from MapState."""
    rows = state.rows
    cols = state.cols

    # Build base char/color grid
    grid: List[List[Tuple[str, str]]] = []
    for r in range(rows):
        row_cells: List[Tuple[str, str]] = [(_OUTSIDE, DIM)] * cols
        grid.append(row_cells)

    # Paint zones
    for zone in state.zones:
        for r in range(zone.y, min(zone.y + zone.height, rows)):
            for c in range(zone.x, min(zone.x + zone.width, cols)):
                if zone.cleaned_pct >= 90:
                    ch, color = _CLEANED, GREEN
                elif zone.cleaned_pct >= 40:
                    ch, color = _PARTIAL, YELLOW
                else:
                    ch, color = _UNCLEANED, SILVER
                grid[r][c] = (ch, color)

    # Paint walls
    for (wc, wr) in state.walls:
        if 0 <= wr < rows and 0 <= wc < cols:
            grid[wr][wc] = (_WALL, DIM)

    # Draw room border
    for c in range(cols):
        grid[0][c] = (_WALL, DIM)
        grid[rows - 1][c] = (_WALL, DIM)
    for r in range(rows):
        grid[r][0] = (_WALL, DIM)
        grid[r][cols - 1] = (_WALL, DIM)

    # Overlay robot labels — each robot occupies 3 cols: [X]
    robot_cells: Dict[Tuple[int, int], RobotPos] = {}
    for robot in state.robots:
        rc = int(round(robot.col))
        rr = int(round(robot.row))
        # bracket left
        if 0 <= rr < rows and 0 <= rc - 1 < cols:
            robot_cells[(rr, rc - 1)] = robot
        if 0 <= rr < rows and 0 <= rc < cols:
            robot_cells[(rr, rc)] = robot
        if 0 <= rr < rows and 0 <= rc + 1 < cols:
            robot_cells[(rr, rc + 1)] = robot

    # Build Text object row by row
    text = Text()
    for r in range(rows):
        for c in range(cols):
            key = (r, c)
            if key in robot_cells:
                robot = robot_cells[key]
                rc_center = int(round(robot.col))
                rr_center = int(round(robot.row))
                if c == rc_center - 1:
                    text.append("[", style=f"bold {CYAN}")
                elif c == rc_center:
                    text.append(robot.label, style=f"bold {CYAN}")
                else:
                    text.append("]", style=f"bold {CYAN}")
            else:
                ch, color = grid[r][c]
                text.append(ch, style=color)
        text.append("\n")

    return text


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class SwarmMap(Widget):
    """Textual widget that renders an ASCII swarm map."""

    DEFAULT_CSS = """
    SwarmMap {
        background: #0F3460;
        border: solid #C0C0C0;
        padding: 0 1;
        height: 1fr;
        overflow: auto;
    }
    """

    map_state: reactive[MapState] = reactive(MapState, layout=True)

    def __init__(self, map_state: Optional[MapState] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.map_state = map_state or _default_map_state()

    def compose(self) -> ComposeResult:
        yield Static("", id="map-static")

    def on_mount(self) -> None:
        self._refresh_map()

    def watch_map_state(self, state: MapState) -> None:
        self._refresh_map()

    def _refresh_map(self) -> None:
        text = _render_map(self.map_state)
        try:
            self.query_one("#map-static", Static).update(text)
        except Exception:
            pass

    def update_map(self, new_state: MapState) -> None:
        """Public API: push a new map snapshot."""
        self.map_state = new_state

    def move_robot(self, name: str, col: float, row: float) -> None:
        """Convenience: update a single robot's position."""
        state = self.map_state
        for robot in state.robots:
            if robot.name == name:
                robot.col = col
                robot.row = row
                break
        # Force reactive refresh by reassigning
        self.map_state = state
        self._refresh_map()


# ---------------------------------------------------------------------------
# Default demo map
# ---------------------------------------------------------------------------

def _default_map_state() -> MapState:
    zones = [
        Zone("A", "A", 1, 1, 18, 8, cleaned_pct=75.0),
        Zone("B", "B", 21, 1, 18, 8, cleaned_pct=30.0),
        Zone("C", "C", 1, 11, 18, 8, cleaned_pct=10.0),
        Zone("D", "D", 21, 11, 18, 8, cleaned_pct=0.0),
    ]
    robots = [
        RobotPos("G1-Alpha", "A", col=9, row=5, status="cleaning"),
        RobotPos("G1-Beta", "B", col=30, row=5, status="idle"),
    ]
    # Internal wall segment
    walls = [(20, r) for r in range(1, 19)]
    walls += [(c, 10) for c in range(1, 40)]
    return MapState(rows=20, cols=40, zones=zones, robots=robots, walls=walls)
