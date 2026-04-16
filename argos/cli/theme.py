"""
ARGOS TUI Theme — color constants and Textual CSS.

Color palette:
  Cyan   #00FFFF  — active, highlight, primary accent
  Silver #C0C0C0  — borders, labels, secondary text
  Navy   #1A1A2E  — application background
  Panel  #16213E  — panel/card background
  Deep   #0F3460  — secondary panel background
  Green  #00FF88  — ok / success / task complete
  Yellow #FFD700  — warning / idle
  Red    #FF4444  — error / critical
  White  #E0E0E0  — primary text
  Dim    #888888  — dim / offline / disabled
"""

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
CYAN = "#00FFFF"
SILVER = "#C0C0C0"
NAVY = "#1A1A2E"
PANEL = "#16213E"
DEEP = "#0F3460"

GREEN = "#00FF88"
YELLOW = "#FFD700"
RED = "#FF4444"
WHITE = "#E0E0E0"
DIM = "#888888"

STATUS_COLORS = {
    "active": GREEN,
    "cleaning": GREEN,
    "ok": GREEN,
    "idle": YELLOW,
    "warning": YELLOW,
    "error": RED,
    "critical": RED,
    "offline": DIM,
    "pending": SILVER,
    "done": GREEN,
    "failed": RED,
}

# ---------------------------------------------------------------------------
# Textual CSS
# ---------------------------------------------------------------------------
ARGOS_CSS = """
/* ── Application shell ────────────────────────────────────────────── */
Screen {
    background: #1A1A2E;
    color: #E0E0E0;
}

/* ── Header / footer ─────────────────────────────────────────────── */
Header {
    background: #0F3460;
    color: #00FFFF;
    text-style: bold;
    height: 3;
}

Footer {
    background: #0F3460;
    color: #C0C0C0;
    height: 1;
}

/* ── Generic panel / card chrome ─────────────────────────────────── */
.panel {
    background: #16213E;
    border: solid #C0C0C0;
    padding: 0 1;
    height: 1fr;
}

.panel-title {
    color: #00FFFF;
    text-style: bold;
    padding: 0 1;
}

/* ── Dashboard grid ──────────────────────────────────────────────── */
#dashboard-grid {
    layout: grid;
    grid-size: 2 2;
    grid-gutter: 1;
    padding: 1;
    height: 1fr;
}

#fleet-panel {
    background: #16213E;
    border: solid #C0C0C0;
    padding: 0 1;
    height: 1fr;
    overflow-y: auto;
}

#map-panel {
    background: #16213E;
    border: solid #C0C0C0;
    padding: 0 1;
    height: 1fr;
}

#task-panel {
    background: #16213E;
    border: solid #C0C0C0;
    padding: 0 1;
    height: 1fr;
    overflow-y: auto;
}

#log-panel {
    background: #16213E;
    border: solid #C0C0C0;
    padding: 0 1;
    height: 1fr;
    overflow-y: auto;
}

/* ── Robot card ──────────────────────────────────────────────────── */
RobotCard {
    background: #0F3460;
    border: solid #C0C0C0;
    padding: 0 1;
    height: 7;
    margin: 0 0 1 0;
}

RobotCard:hover {
    border: solid #00FFFF;
}

RobotCard.selected {
    border: double #00FFFF;
}

.robot-name {
    color: #00FFFF;
    text-style: bold;
}

.robot-status-active {
    color: #00FF88;
    text-style: bold;
}

.robot-status-idle {
    color: #FFD700;
}

.robot-status-error {
    color: #FF4444;
    text-style: bold;
}

.robot-status-offline {
    color: #888888;
}

.battery-bar {
    color: #00FF88;
}

.battery-low {
    color: #FF4444;
}

.battery-mid {
    color: #FFD700;
}

.robot-label {
    color: #C0C0C0;
}

/* ── Swarm map ───────────────────────────────────────────────────── */
SwarmMap {
    background: #0F3460;
    border: solid #C0C0C0;
    padding: 1;
    height: 1fr;
}

.map-robot-label {
    color: #00FFFF;
    text-style: bold;
}

.map-cleaned {
    color: #00FF88;
}

.map-zone {
    color: #C0C0C0;
}

/* ── Log panel ───────────────────────────────────────────────────── */
LogPanel {
    background: #16213E;
    border: solid #C0C0C0;
    height: 1fr;
}

.log-info {
    color: #00FFFF;
}

.log-warning {
    color: #FFD700;
}

.log-error {
    color: #FF4444;
}

.log-success {
    color: #00FF88;
}

.log-timestamp {
    color: #888888;
}

/* ── Task table ──────────────────────────────────────────────────── */
DataTable {
    background: #16213E;
    color: #E0E0E0;
    border: solid #C0C0C0;
    height: 1fr;
}

DataTable > .datatable--header {
    background: #0F3460;
    color: #00FFFF;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #0F3460;
    color: #00FFFF;
}

DataTable > .datatable--hover {
    background: #0F3460;
}

/* ── Training screen ─────────────────────────────────────────────── */
#training-grid {
    layout: grid;
    grid-size: 1;
    grid-gutter: 1;
    padding: 1;
    height: 1fr;
}

.training-progress-label {
    color: #C0C0C0;
}

ProgressBar {
    color: #00FFFF;
    background: #0F3460;
}

ProgressBar > .bar--bar {
    color: #00FFFF;
}

ProgressBar > .bar--complete {
    color: #00FF88;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
Button {
    background: #0F3460;
    color: #00FFFF;
    border: solid #C0C0C0;
    margin: 0 1;
}

Button:hover {
    background: #16213E;
    border: solid #00FFFF;
}

Button.-primary {
    background: #00FFFF;
    color: #1A1A2E;
    text-style: bold;
}

Button.-error {
    background: #FF4444;
    color: #E0E0E0;
    text-style: bold;
}

Button.-warning {
    background: #FFD700;
    color: #1A1A2E;
}

/* ── Input fields ────────────────────────────────────────────────── */
Input {
    background: #0F3460;
    color: #E0E0E0;
    border: solid #C0C0C0;
}

Input:focus {
    border: solid #00FFFF;
}

/* ── Task screen ─────────────────────────────────────────────────── */
#tasks-container {
    layout: vertical;
    padding: 1;
    height: 1fr;
}

#task-input-row {
    layout: horizontal;
    height: 3;
    margin-top: 1;
}

#task-input {
    width: 1fr;
    margin-right: 1;
}

/* ── Metrics table ───────────────────────────────────────────────── */
.metric-label {
    color: #C0C0C0;
    width: 20;
}

.metric-value {
    color: #00FFFF;
    text-style: bold;
}

/* ── Banner / title area ─────────────────────────────────────────── */
.banner {
    color: #00FFFF;
    text-style: bold;
    text-align: center;
}

.subtitle {
    color: #C0C0C0;
    text-align: center;
}

/* ── Status bar inside dashboard ─────────────────────────────────── */
#status-bar {
    background: #0F3460;
    color: #C0C0C0;
    height: 1;
    padding: 0 1;
}

/* ── Scrollable containers ───────────────────────────────────────── */
ScrollableContainer {
    background: #16213E;
    height: 1fr;
}

/* ── Static text helpers ─────────────────────────────────────────── */
.dim {
    color: #888888;
}

.cyan {
    color: #00FFFF;
}

.silver {
    color: #C0C0C0;
}

.green {
    color: #00FF88;
}

.yellow {
    color: #FFD700;
}

.red {
    color: #FF4444;
}

.bold {
    text-style: bold;
}
"""
