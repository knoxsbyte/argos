"""ARGOS interactive REPL — stays in terminal, Rich-based, Claude Code style."""
from __future__ import annotations

import time
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from argos.cli.theme import CYAN, DIM, GREEN, RED, SILVER, YELLOW

console = Console(highlight=False)

BANNER = r"""
 █████╗ ██████╗  ██████╗  ██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗██╔════╝
███████║██████╔╝██║  ███╗██║   ██║███████╗
██╔══██║██╔══██╗██║   ██║██║   ██║╚════██║
██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
"""


def print_banner(robot_count: int = 0) -> None:
    console.print(Text(BANNER, style=f"bold {CYAN}"), end="")
    console.print(f"  Autonomous Robot Group Operations System  [{SILVER}]v0.1.0[/]")
    if robot_count:
        console.print(f"  [{GREEN}]{robot_count} robot{'s' if robot_count != 1 else ''} online[/]")
    else:
        console.print(f"  [{YELLOW}]no robots connected[/]")
    console.print()


def _battery_bar(pct: float) -> str:
    filled = int(pct / 10)
    empty = 10 - filled
    color = GREEN if pct > 40 else (YELLOW if pct > 15 else RED)
    return f"[{color}]{'█' * filled}{'░' * empty}[/] {pct:.0f}%"


def _box():
    from rich.box import SIMPLE_HEAVY
    return SIMPLE_HEAVY


def spinner(message: str, duration: float = 1.2) -> None:
    with Progress(
        SpinnerColumn(style=CYAN),
        TextColumn(f"[{SILVER}]{message}[/]"),
        transient=True,
        console=console,
    ) as p:
        p.add_task("", total=None)
        time.sleep(duration)


def render_fleet_table(robots: list[dict]) -> Table:
    t = Table(show_header=True, header_style=f"bold {CYAN}",
              border_style=SILVER, box=_box(), expand=False, min_width=62)
    t.add_column("Robot", style=f"bold {CYAN}", no_wrap=True)
    t.add_column("Status", no_wrap=True)
    t.add_column("Battery", no_wrap=True)
    t.add_column("Task", style=SILVER)
    t.add_column("Zone", style=DIM)
    for r in robots:
        bat = r.get("battery", 100)
        dot = {"CLEANING": f"[{GREEN}]● CLEANING[/]",
               "IDLE":     f"[{SILVER}]○ IDLE[/]",
               "ERROR":    f"[{RED}]✖ ERROR[/]",
               "OFFLINE":  f"[{DIM}]— OFFLINE[/]"}.get(r.get("status", "IDLE"), f"[{SILVER}]○ IDLE[/]")
        t.add_row(r.get("name", "?"), dot, _battery_bar(bat),
                  r.get("task", "—"), r.get("zone", "—"))
    return t


def render_task_table(tasks: list[dict]) -> Table:
    t = Table(show_header=True, header_style=f"bold {CYAN}",
              border_style=SILVER, box=_box(), expand=False, min_width=72)
    t.add_column("ID", style=DIM, no_wrap=True, width=8)
    t.add_column("Task", style=f"bold {SILVER}")
    t.add_column("Type", style=DIM)
    t.add_column("Robot(s)", style=CYAN)
    t.add_column("Status", no_wrap=True)
    style_map = {
        "PENDING":   f"[{SILVER}]PENDING[/]",
        "ACTIVE":    f"[{CYAN}]▶ ACTIVE[/]",
        "DONE":      f"[{GREEN}]✓ DONE[/]",
        "FAILED":    f"[{RED}]✖ FAILED[/]",
        "CANCELLED": f"[{YELLOW}]⊘ CANCELLED[/]",
    }
    for task in tasks:
        t.add_row(task.get("id", "?")[:8], task.get("name", "?"),
                  task.get("type", "solo"), task.get("robots", "—"),
                  style_map.get(task.get("status", "PENDING"), task.get("status", "")))
    return t


HELP_TEXT = f"""[bold {CYAN}]Commands[/]

  [bold {SILVER}]status[/]                          Show fleet and task overview
  [bold {SILVER}]task add[/] [italic]"<goal>"[/]            Add a cleaning task (natural language)
  [bold {SILVER}]task list[/]                       List all tasks
  [bold {SILVER}]task cancel[/] [italic]<id>[/]             Cancel a task
  [bold {SILVER}]task status[/] [italic]<id>[/]             Show task detail
  [bold {SILVER}]connect[/] [italic]<ip>[/] [[italic]--name NAME[/]]    Connect a Unitree G1 robot
  [bold {SILVER}]disconnect[/] [italic]<name>[/]            Disconnect a robot
  [bold {SILVER}]train[/]                           Show training commands
  [bold {SILVER}]sim[/]                             Launch MuJoCo simulation
  [bold {SILVER}]clear[/]                           Clear screen
  [bold {SILVER}]help[/]                            Show this help
  [bold {SILVER}]exit[/] / [bold {SILVER}]quit[/]                   Exit ARGOS
"""


class ArgosREPL:
    def __init__(self) -> None:
        self._robots: list[dict] = []
        self._tasks: list[dict] = []
        self._running = True
        self._task_counter = 0

    def run(self) -> None:
        console.clear()
        print_banner(len(self._robots))
        console.print(f"  [{DIM}]Type [bold]help[/] for commands · [bold]exit[/] to quit[/]\n")
        while self._running:
            try:
                raw = console.input(f"[bold {CYAN}]argos[/] [{SILVER}]›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                self._cmd_exit()
                break
            if not raw:
                continue
            self._dispatch(raw)

    def _dispatch(self, raw: str) -> None:
        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]
        routes: dict[str, Callable] = {
            "status":     self._cmd_status,
            "fleet":      self._cmd_status,
            "task":       self._cmd_task,
            "connect":    self._cmd_connect,
            "disconnect": self._cmd_disconnect,
            "train":      self._cmd_train,
            "sim":        self._cmd_sim,
            "clear":      self._cmd_clear,
            "help":       self._cmd_help,
            "exit":       self._cmd_exit,
            "quit":       self._cmd_exit,
        }
        fn = routes.get(cmd)
        if fn:
            fn(args)
        else:
            console.print(f"  [{YELLOW}]Unknown command:[/] {cmd}  — type [bold]help[/]\n")

    # ── commands ──────────────────────────────────────────────────────────────

    def _cmd_status(self, args: list[str] = []) -> None:
        console.print()
        if self._robots:
            console.print(Panel(render_fleet_table(self._robots),
                                title=f"[bold {CYAN}]Fleet[/]", border_style=SILVER))
        else:
            console.print(f"  [{YELLOW}]No robots connected.[/]  Use [bold]connect <ip>[/] to add one.\n")
        if self._tasks:
            console.print(Panel(render_task_table(self._tasks),
                                title=f"[bold {CYAN}]Tasks[/]", border_style=SILVER))
        else:
            console.print(f"  [{DIM}]No tasks queued.[/]\n")

    def _cmd_task(self, args: list[str]) -> None:
        if not args:
            console.print(f"  [{YELLOW}]Usage:[/] task add \"<goal>\" | task list | task cancel <id>\n")
            return
        sub = args[0].lower()
        if sub == "add":
            goal = " ".join(args[1:]).strip('"\'')
            if not goal:
                console.print(f"  [{YELLOW}]Usage:[/] task add \"<goal>\"\n")
                return
            self._add_task(goal)
        elif sub == "list":
            if not self._tasks:
                console.print(f"  [{DIM}]No tasks.[/]\n")
            else:
                console.print(Panel(render_task_table(self._tasks),
                                    title=f"[bold {CYAN}]Tasks[/]", border_style=SILVER))
                console.print()
        elif sub == "cancel" and args[1:]:
            tid = args[1]
            for t in self._tasks:
                if t["id"].startswith(tid):
                    t["status"] = "CANCELLED"
                    console.print(f"  [{YELLOW}]⊘[/] Task [{SILVER}]{t['id']}[/] cancelled.\n")
                    return
            console.print(f"  [{RED}]Task not found:[/] {tid}\n")
        elif sub == "status" and args[1:]:
            tid = args[1]
            for t in self._tasks:
                if t["id"].startswith(tid):
                    console.print(Panel(
                        f"[bold {SILVER}]{t['name']}[/]\n"
                        f"Type:    {t.get('type','solo')}\n"
                        f"Robot:   {t.get('robots','—')}\n"
                        f"Status:  {t['status']}",
                        title=f"[bold {CYAN}]Task {t['id']}[/]", border_style=SILVER))
                    console.print()
                    return
            console.print(f"  [{RED}]Task not found:[/] {tid}\n")
        else:
            console.print(f"  [{YELLOW}]Unknown sub-command:[/] {sub}\n")

    def _add_task(self, goal: str) -> None:
        console.print()
        spinner(f"Decomposing: \"{goal}\"")
        subtasks = self._infer_subtasks(goal)
        robot_names = [r["name"] for r in self._robots] if self._robots else ["—"]
        for i, st in enumerate(subtasks):
            self._task_counter += 1
            tid = f"t{self._task_counter:03d}"
            coop = st.get("cooperative", False)
            robots_str = " + ".join(robot_names) if coop else robot_names[i % len(robot_names)]
            self._tasks.append({"id": tid, "name": st["name"], "type": st.get("type", "solo"),
                                 "robots": robots_str, "status": "PENDING"})
            tag = f"[{YELLOW}][cooperative][/]" if coop else ""
            console.print(f"  [{DIM}]↳[/] [{CYAN}]{tid}[/]  {st['name']:<24} [{SILVER}]→ {robots_str}[/]  {tag}")
        console.print(f"\n  [{GREEN}]✓[/] {len(subtasks)} task(s) queued.\n")
        for t in self._tasks:
            if t["status"] == "PENDING":
                t["status"] = "ACTIVE"
                for r in self._robots:
                    if r["name"] in t["robots"]:
                        r["status"] = "CLEANING"
                        r["task"] = t["name"]

    def _cmd_connect(self, args: list[str]) -> None:
        if not args:
            console.print(f"  [{YELLOW}]Usage:[/] connect <ip> [--name NAME]\n")
            return
        ip = args[0]
        name = args[args.index("--name") + 1] if "--name" in args else f"G1-{chr(65 + len(self._robots))}"
        spinner(f"Connecting to {ip} as {name}…")
        self._robots.append({"name": name, "ip": ip, "status": "IDLE",
                              "battery": 100, "task": "—", "zone": "—"})
        console.print(f"  [{GREEN}]✓[/] Connected: [{CYAN}]{name}[/] @ {ip}\n")

    def _cmd_disconnect(self, args: list[str]) -> None:
        if not args:
            console.print(f"  [{YELLOW}]Usage:[/] disconnect <name>\n")
            return
        name = args[0]
        before = len(self._robots)
        self._robots = [r for r in self._robots if r["name"] != name]
        if len(self._robots) < before:
            console.print(f"  [{YELLOW}]⊘[/] Disconnected: {name}\n")
        else:
            console.print(f"  [{RED}]Robot not found:[/] {name}\n")

    def _cmd_train(self, args: list[str] = []) -> None:
        console.print(Panel(
            f"  [{SILVER}]Run these from your shell:[/]\n\n"
            f"  [{CYAN}]argos train ingest[/]   --video-dir DIR\n"
            f"  [{CYAN}]argos train finetune[/] --dataset DIR [--epochs N]\n"
            f"  [{CYAN}]argos train evaluate[/] --model PATH\n"
            f"  [{CYAN}]argos train deploy[/]   --model PATH --robot NAME\n",
            title=f"[bold {CYAN}]Training[/]", border_style=SILVER))
        console.print()

    def _cmd_sim(self, args: list[str] = []) -> None:
        spinner("Launching MuJoCo simulation…", 1.5)
        try:
            from argos.training.sim.mujoco_env import CleaningEnv
            env = CleaningEnv(task_type="sweep_floor", room_layout="simple")
            obs, _ = env.reset()
            console.print(f"  [{GREEN}]✓[/] Simulation ready — room: simple, task: sweep_floor")
            console.print(f"  [{DIM}]  obs keys: {list(obs.keys())}[/]\n")
            env.close()
        except Exception as e:
            console.print(f"  [{RED}]Sim error:[/] {e}\n")

    def _cmd_clear(self, args: list[str] = []) -> None:
        console.clear()
        print_banner(len(self._robots))

    def _cmd_help(self, args: list[str] = []) -> None:
        console.print(Panel(HELP_TEXT, title=f"[bold {CYAN}]ARGOS Help[/]", border_style=SILVER))
        console.print()

    def _cmd_exit(self, args: list[str] = []) -> None:
        console.print(f"\n  [{DIM}]Goodbye.[/]\n")
        self._running = False

    def _infer_subtasks(self, goal: str) -> list[dict]:
        g = goal.lower()
        tasks = []
        if any(w in g for w in ["sweep", "clean", "floor"]):
            tasks.append({"name": "sweep_floor", "type": "solo"})
        if any(w in g for w in ["wipe", "surface", "counter", "kitchen"]):
            tasks.append({"name": "wipe_surface", "type": "solo"})
        if any(w in g for w in ["mop"]):
            tasks.append({"name": "mop_floor", "type": "solo"})
        if any(w in g for w in ["bed", "bedroom", "sheet"]):
            tasks.append({"name": "make_bed", "type": "cooperative", "cooperative": True})
        if any(w in g for w in ["vacuum"]):
            tasks.append({"name": "vacuum_floor", "type": "solo"})
        if any(w in g for w in ["trash", "garbage"]):
            tasks.append({"name": "take_out_trash", "type": "solo"})
        if any(w in g for w in ["organiz", "shelf"]):
            tasks.append({"name": "organize_shelf", "type": "solo"})
        if not tasks:
            tasks.append({"name": "sweep_floor", "type": "solo"})
            tasks.append({"name": "wipe_surface", "type": "solo"})
        return tasks
