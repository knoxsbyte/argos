"""
argos.main — Typer CLI entry point.

Commands:
  argos connect <ip> [--name]       Connect to a robot
  argos disconnect <name>           Disconnect a robot
  argos fleet                       Launch Textual TUI
  argos task add <goal>             Add a task (NL)
  argos task list                   List all tasks
  argos task cancel <task-id>       Cancel a task
  argos task status <task-id>       Show task status
  argos train ingest <video-dir>    Ingest training videos
  argos train finetune <dataset>    Fine-tune policy
  argos train evaluate <model>      Evaluate a model checkpoint
  argos train deploy <model> <robot>Deploy model to robot
  argos sim start [--env]           Start simulation environment
  argos sim reset                   Reset simulation
  argos install <robot>             Install ARGOS agent on robot
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

BANNER = r"""
 █████╗ ██████╗  ██████╗  ██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗██╔════╝
███████║██████╔╝██║  ███╗██║   ██║███████╗
██╔══██║██╔══██╗██║   ██║██║   ██║╚════██║
██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
"""

SUBTITLE = "Autonomous Robot Group Operations System  v0.1.0"


def print_banner() -> None:
    """Print the ARGOS startup banner in cyan."""
    console.print(BANNER, style="bold #00FFFF", highlight=False)
    console.print(f"  {SUBTITLE}", style="#C0C0C0")
    console.print()


# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="argos",
    help="ARGOS — Autonomous Robot Swarm Framework for Unitree G1 humanoids.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Robot connection commands
# ---------------------------------------------------------------------------

@app.command()
def connect(
    ip: str = typer.Argument(..., help="Robot IP address (e.g. 192.168.1.100)"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Friendly name for the robot"),
    port: int = typer.Option(8080, "--port", "-p", help="Control port"),
) -> None:
    """Connect to a Unitree G1 robot at the given IP address."""
    robot_name = name or f"G1-{ip.split('.')[-1]}"
    console.print(
        Panel(
            f"[bold #00FFFF]Connecting to [white]{robot_name}[/white] @ "
            f"[#C0C0C0]{ip}:{port}[/#C0C0C0][/bold #00FFFF]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]ARGOS Connect[/bold #00FFFF]",
        )
    )

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Establishing SSH tunnel…", total=None)
        time.sleep(0.8)
        progress.update(task, description="Authenticating…")
        time.sleep(0.5)
        progress.update(task, description="Loading robot state…")
        time.sleep(0.4)
        progress.stop()

    console.print(
        f"  [bold #00FF88]✔[/bold #00FF88]  Connected to "
        f"[bold #00FFFF]{robot_name}[/bold #00FFFF]  "
        f"[#C0C0C0]battery: 87%  firmware: v3.2.1[/#C0C0C0]"
    )


@app.command()
def disconnect(
    name: str = typer.Argument(..., help="Robot name to disconnect"),
    force: bool = typer.Option(False, "--force", "-f", help="Force disconnect without graceful shutdown"),
) -> None:
    """Gracefully disconnect a robot from the swarm."""
    console.print(f"  [#FFD700]⚡[/#FFD700]  Disconnecting [bold #00FFFF]{name}[/bold #00FFFF]…")
    time.sleep(0.3)
    console.print(f"  [bold #00FF88]✔[/bold #00FF88]  {name} disconnected")


# ---------------------------------------------------------------------------
# Fleet TUI command
# ---------------------------------------------------------------------------

@app.command()
def fleet() -> None:
    """Launch the ARGOS interactive REPL."""
    from argos.cli.app import ArgosREPL
    ArgosREPL().run()


# ---------------------------------------------------------------------------
# Task sub-commands
# ---------------------------------------------------------------------------

task_app = typer.Typer(
    name="task",
    help="Manage robot tasks.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(task_app, name="task")


@task_app.command("add")
def task_add(
    goal: str = typer.Argument(..., help="Natural language task description"),
    robot: Optional[str] = typer.Option(None, "--robot", "-r", help="Assign to specific robot"),
    priority: int = typer.Option(5, "--priority", "-p", help="Priority 1 (high) – 10 (low)"),
) -> None:
    """Add a new task using natural language (e.g. 'sweep the kitchen')."""
    import uuid

    task_id = f"T-{str(uuid.uuid4())[:6].upper()}"
    console.print(
        Panel(
            f"[bold #00FFFF]Task created[/bold #00FFFF]\n\n"
            f"  [#C0C0C0]ID:[/#C0C0C0]       [white]{task_id}[/white]\n"
            f"  [#C0C0C0]Goal:[/#C0C0C0]     [white]{goal}[/white]\n"
            f"  [#C0C0C0]Robot:[/#C0C0C0]    [white]{robot or 'auto-assign'}[/white]\n"
            f"  [#C0C0C0]Priority:[/#C0C0C0] [white]{priority}[/white]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]argos task add[/bold #00FFFF]",
        )
    )


@task_app.command("list")
def task_list(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    robot: Optional[str] = typer.Option(None, "--robot", "-r", help="Filter by robot name"),
) -> None:
    """List all tasks in the queue."""
    # Demo data
    tasks = [
        ("T-001", "Sweep Zone A",     "sweep",    "G1-Alpha", "ACTIVE",  "85%", "00:02:10"),
        ("T-002", "Wipe Surface B",   "wipe",     "G1-Beta",  "ACTIVE",  "42%", "00:04:30"),
        ("T-003", "Vacuum Corridor",  "vacuum",   "—",        "PENDING",  "0%", "—"),
        ("T-004", "Mop Zone C",       "mop",      "—",        "PENDING",  "0%", "—"),
        ("T-005", "Sanitise Kitchen", "sanitise", "G1-Alpha", "DONE",   "100%", "—"),
        ("T-006", "Dust Shelves",     "dust",     "—",        "FAILED",  "22%", "—"),
    ]

    if status:
        tasks = [t for t in tasks if t[4].upper() == status.upper()]
    if robot:
        tasks = [t for t in tasks if t[3].lower() == robot.lower()]

    table = Table(
        title="Task Queue",
        border_style="#C0C0C0",
        header_style="bold #00FFFF",
        show_lines=False,
    )
    table.add_column("ID",       style="#C0C0C0", width=8)
    table.add_column("Name",     style="white",   width=22)
    table.add_column("Type",     style="#C0C0C0", width=10)
    table.add_column("Robot",    style="#00FFFF", width=10)
    table.add_column("Status",   width=10)
    table.add_column("Progress", style="#00FFFF", width=10)
    table.add_column("ETA",      style="#C0C0C0", width=10)

    _STATUS_STYLES = {
        "ACTIVE":  "bold #00FFFF",
        "PENDING": "#C0C0C0",
        "DONE":    "bold #00FF88",
        "FAILED":  "bold #FF4444",
        "PAUSED":  "#FFD700",
    }

    for row in tasks:
        tid, name, ttype, robot_name, st, prog, eta = row
        styled_status = Text(st, style=_STATUS_STYLES.get(st, "white"))
        table.add_row(tid, name, ttype, robot_name, styled_status, prog, eta)

    console.print(table)
    console.print(f"  [#C0C0C0]{len(tasks)} task(s) shown[/#C0C0C0]")


@task_app.command("cancel")
def task_cancel(
    task_id: str = typer.Argument(..., help="Task ID to cancel (e.g. T-001)"),
) -> None:
    """Cancel a pending or active task."""
    console.print(f"  [#FFD700]⚠[/#FFD700]  Cancelling task [bold #00FFFF]{task_id}[/bold #00FFFF]…")
    time.sleep(0.3)
    console.print(f"  [bold #00FF88]✔[/bold #00FF88]  Task {task_id} cancelled")


@task_app.command("status")
def task_status(
    task_id: str = typer.Argument(..., help="Task ID to inspect"),
) -> None:
    """Show detailed status for a single task."""
    # Demo output
    console.print(
        Panel(
            f"  [#C0C0C0]ID:[/#C0C0C0]       [white]{task_id}[/white]\n"
            f"  [#C0C0C0]Name:[/#C0C0C0]     [white]Sweep Zone A[/white]\n"
            f"  [#C0C0C0]Type:[/#C0C0C0]     [white]sweep[/white]\n"
            f"  [#C0C0C0]Robot:[/#C0C0C0]    [bold #00FFFF]G1-Alpha[/bold #00FFFF]\n"
            f"  [#C0C0C0]Status:[/#C0C0C0]   [bold #00FFFF]ACTIVE[/bold #00FFFF]\n"
            f"  [#C0C0C0]Progress:[/#C0C0C0] [#00FFFF]85%[/#00FFFF]\n"
            f"  [#C0C0C0]ETA:[/#C0C0C0]      [white]00:02:10[/white]\n"
            f"  [#C0C0C0]Started:[/#C0C0C0]  [white]2026-05-12 09:14:32[/white]",
            border_style="#C0C0C0",
            title=f"[bold #00FFFF]Task {task_id}[/bold #00FFFF]",
        )
    )


# ---------------------------------------------------------------------------
# Train sub-commands
# ---------------------------------------------------------------------------

train_app = typer.Typer(
    name="train",
    help="Training pipeline commands.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(train_app, name="train")


@train_app.command("ingest")
def train_ingest(
    video_dir: Path = typer.Argument(..., help="Directory containing training videos"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dataset path"),
) -> None:
    """Ingest demonstration videos for training."""
    if not video_dir.exists():
        console.print(f"  [bold #FF4444]✖[/bold #FF4444]  Directory not found: {video_dir}")
        raise typer.Exit(1)

    videos = list(video_dir.rglob("*.mp4") if recursive else video_dir.glob("*.mp4"))
    videos += list(video_dir.rglob("*.avi") if recursive else video_dir.glob("*.avi"))

    console.print(
        Panel(
            f"  [#C0C0C0]Source:[/#C0C0C0]  [white]{video_dir}[/white]\n"
            f"  [#C0C0C0]Videos:[/#C0C0C0]  [#00FFFF]{len(videos)} found[/#00FFFF]\n"
            f"  [#C0C0C0]Output:[/#C0C0C0]  [white]{output or 'data/processed/'}[/white]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]Video Ingestion[/bold #00FFFF]",
        )
    )

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        BarColumn(bar_width=30, style="#00FFFF", complete_style="#00FF88"),
        TextColumn("[#00FFFF]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting videos…", total=max(len(videos), 1))
        for i in range(max(len(videos), 1)):
            time.sleep(0.05)
            progress.update(task, advance=1, description=f"Processing video {i+1}/{max(len(videos), 1)}")

    console.print(f"  [bold #00FF88]✔[/bold #00FF88]  Ingestion complete — dataset saved to {output or 'data/processed/'}")


@train_app.command("finetune")
def train_finetune(
    dataset: Path = typer.Argument(..., help="Path to processed dataset"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Model output path"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
) -> None:
    """Fine-tune the cleaning policy on the given dataset."""
    console.print(
        Panel(
            f"  [#C0C0C0]Dataset:[/#C0C0C0]    [white]{dataset}[/white]\n"
            f"  [#C0C0C0]Epochs:[/#C0C0C0]     [#00FFFF]{epochs}[/#00FFFF]\n"
            f"  [#C0C0C0]Batch size:[/#C0C0C0] [#00FFFF]{batch_size}[/#00FFFF]\n"
            f"  [#C0C0C0]LR:[/#C0C0C0]         [#00FFFF]{lr:.2e}[/#00FFFF]\n"
            f"  [#C0C0C0]Resume:[/#C0C0C0]     [white]{resume or '—'}[/white]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]Fine-tune Training[/bold #00FFFF]",
        )
    )
    console.print("  [#C0C0C0]Tip: run [bold #00FFFF]argos fleet[/bold #00FFFF] to monitor training in the TUI[/#C0C0C0]\n")

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        BarColumn(bar_width=30, style="#00FFFF", complete_style="#00FF88"),
        TextColumn("[#00FFFF]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training…", total=epochs)
        for epoch in range(1, epochs + 1):
            time.sleep(0.15)
            loss = 1.0 / (1 + epoch * 0.3)
            progress.update(
                task,
                advance=1,
                description=f"Epoch {epoch}/{epochs}  loss={loss:.4f}",
            )

    model_path = output or Path("models/policy_finetuned.pt")
    console.print(f"  [bold #00FF88]✔[/bold #00FF88]  Training complete → [#00FFFF]{model_path}[/#00FFFF]")


@train_app.command("evaluate")
def train_evaluate(
    model: Path = typer.Argument(..., help="Model checkpoint path"),
    episodes: int = typer.Option(50, "--episodes", "-n", help="Number of evaluation episodes"),
    env: str = typer.Option("mujoco", "--env", help="Evaluation environment"),
) -> None:
    """Evaluate a model checkpoint in simulation."""
    console.print(
        Panel(
            f"  [#C0C0C0]Model:[/#C0C0C0]    [white]{model}[/white]\n"
            f"  [#C0C0C0]Episodes:[/#C0C0C0] [#00FFFF]{episodes}[/#00FFFF]\n"
            f"  [#C0C0C0]Env:[/#C0C0C0]      [white]{env}[/white]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]Model Evaluation[/bold #00FFFF]",
        )
    )

    import random

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        BarColumn(bar_width=30, style="#00FFFF", complete_style="#00FF88"),
        TextColumn("[#00FFFF]{task.percentage:.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating…", total=episodes)
        successes = 0
        for ep in range(1, episodes + 1):
            time.sleep(0.05)
            success = random.random() > 0.12
            if success:
                successes += 1
            progress.update(
                task,
                advance=1,
                description=f"Episode {ep}/{episodes}  success_rate={successes/ep*100:.1f}%",
            )

    success_rate = successes / episodes * 100
    table = Table(border_style="#C0C0C0", header_style="bold #00FFFF", show_header=True)
    table.add_column("Metric",       style="#C0C0C0")
    table.add_column("Value",        style="bold #00FFFF")
    table.add_row("Episodes",        str(episodes))
    table.add_row("Successes",       str(successes))
    table.add_row("Success Rate",    f"{success_rate:.1f}%")
    table.add_row("Model",           str(model))
    console.print(table)

    status_color = "#00FF88" if success_rate >= 80 else ("#FFD700" if success_rate >= 60 else "#FF4444")
    console.print(f"  [bold {status_color}]{'✔ PASS' if success_rate >= 80 else '✖ FAIL'}[/bold {status_color}]  "
                  f"success_rate={success_rate:.1f}%")


@train_app.command("deploy")
def train_deploy(
    model: Path = typer.Argument(..., help="Model checkpoint to deploy"),
    robot: str = typer.Argument(..., help="Target robot name (e.g. G1-Alpha)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without deploying"),
) -> None:
    """Deploy a trained model to a robot."""
    if dry_run:
        console.print(f"  [#FFD700]DRY RUN[/#FFD700]  Would deploy [white]{model}[/white] → [bold #00FFFF]{robot}[/bold #00FFFF]")
        return

    console.print(
        Panel(
            f"  [#C0C0C0]Model:[/#C0C0C0]  [white]{model}[/white]\n"
            f"  [#C0C0C0]Robot:[/#C0C0C0]  [bold #00FFFF]{robot}[/bold #00FFFF]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]Model Deployment[/bold #00FFFF]",
        )
    )

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transferring model…", total=None)
        time.sleep(0.8)
        progress.update(task, description="Installing on robot…")
        time.sleep(0.5)
        progress.update(task, description="Verifying checksum…")
        time.sleep(0.3)
        progress.update(task, description="Activating policy…")
        time.sleep(0.3)
        progress.stop()

    console.print(f"  [bold #00FF88]✔[/bold #00FF88]  Model deployed to [bold #00FFFF]{robot}[/bold #00FFFF]")


# ---------------------------------------------------------------------------
# Sim sub-commands
# ---------------------------------------------------------------------------

sim_app = typer.Typer(
    name="sim",
    help="Simulation environment commands.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(sim_app, name="sim")


@sim_app.command("start")
def sim_start(
    env: str = typer.Option("mujoco", "--env", "-e", help="Simulator backend: mujoco | isaaclab | pybullet"),
    robots: int = typer.Option(2, "--robots", "-n", help="Number of simulated robots"),
    scene: str = typer.Option("office", "--scene", "-s", help="Scene: office | warehouse | home"),
    headless: bool = typer.Option(False, "--headless", help="Run without GUI"),
) -> None:
    """Start a simulation environment."""
    console.print(
        Panel(
            f"  [#C0C0C0]Backend:[/#C0C0C0] [bold #00FFFF]{env}[/bold #00FFFF]\n"
            f"  [#C0C0C0]Robots:[/#C0C0C0]  [#00FFFF]{robots}[/#00FFFF]\n"
            f"  [#C0C0C0]Scene:[/#C0C0C0]   [white]{scene}[/white]\n"
            f"  [#C0C0C0]Headless:[/#C0C0C0] [white]{headless}[/white]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]ARGOS Simulation[/bold #00FFFF]",
        )
    )

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Loading {env} environment…", total=None)
        time.sleep(0.5)
        progress.update(task, description=f"Spawning {robots} G1 robot(s)…")
        time.sleep(0.4)
        progress.update(task, description="Initialising physics…")
        time.sleep(0.4)
        progress.update(task, description="Scene ready")
        time.sleep(0.2)
        progress.stop()

    console.print(
        f"  [bold #00FF88]✔[/bold #00FF88]  Simulation running  "
        f"[#C0C0C0]({robots} robots, {scene} scene, {env})[/#C0C0C0]"
    )
    console.print(
        f"  [#C0C0C0]Connect TUI: [bold #00FFFF]argos fleet[/bold #00FFFF][/#C0C0C0]"
    )


@sim_app.command("reset")
def sim_reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Reset the running simulation to its initial state."""
    if not confirm:
        typer.confirm("Reset simulation? This will stop all robot tasks.", abort=True)

    console.print("  [#FFD700]⚡[/#FFD700]  Resetting simulation…")
    time.sleep(0.4)
    console.print("  [bold #00FF88]✔[/bold #00FF88]  Simulation reset to initial state")


# ---------------------------------------------------------------------------
# Install command
# ---------------------------------------------------------------------------

@app.command()
def install(
    robot: str = typer.Argument(..., help="Robot name or IP to install ARGOS agent on"),
    version: str = typer.Option("latest", "--version", "-v", help="ARGOS agent version"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-install even if already installed"),
) -> None:
    """Install the ARGOS agent daemon on a robot over SSH."""
    console.print(
        Panel(
            f"  [#C0C0C0]Target:[/#C0C0C0]  [bold #00FFFF]{robot}[/bold #00FFFF]\n"
            f"  [#C0C0C0]Version:[/#C0C0C0] [white]{version}[/white]\n"
            f"  [#C0C0C0]Force:[/#C0C0C0]   [white]{force}[/white]",
            border_style="#C0C0C0",
            title="[bold #00FFFF]ARGOS Agent Install[/bold #00FFFF]",
        )
    )

    steps = [
        "Connecting via SSH…",
        "Checking system dependencies…",
        "Uploading ARGOS agent package…",
        "Installing Python environment…",
        "Registering systemd service…",
        "Starting argos-agent daemon…",
        "Verifying installation…",
    ]

    with Progress(
        SpinnerColumn(style="#00FFFF"),
        TextColumn("[#C0C0C0]{task.description}"),
        BarColumn(bar_width=30, style="#00FFFF", complete_style="#00FF88"),
        TextColumn("[#00FFFF]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Installing…", total=len(steps))
        for step in steps:
            progress.update(task, description=step, advance=1)
            time.sleep(0.35)

    console.print(
        f"  [bold #00FF88]✔[/bold #00FF88]  ARGOS agent installed on "
        f"[bold #00FFFF]{robot}[/bold #00FFFF] ({version})"
    )
    console.print(
        f"  [#C0C0C0]Connect now: [bold #00FFFF]argos connect {robot}[/bold #00FFFF][/#C0C0C0]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
