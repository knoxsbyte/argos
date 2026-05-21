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
  [bold {SILVER}]task add[/] [italic]"<goal>"[/]            Add a task via natural language (LLM decomposed)
  [bold {SILVER}]task create[/] [italic]<type>[/] [--robot N] [--zone Z] [--target T]
                              Schedule a specific task directly to a robot
  [bold {SILVER}]task build[/]                      Interactive wizard — guided task creation
  [bold {SILVER}]task types[/]                      List all available task types
  [bold {SILVER}]task list[/]                       List all queued/active/done tasks
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
            "battery":    self._cmd_battery,
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
        elif sub == "create":
            self._cmd_task_create(args[1:])
        elif sub == "build":
            self._cmd_task_build()
        elif sub == "types":
            self._cmd_task_types()
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

    # ── task create / build / types ───────────────────────────────────────────

    def _cmd_task_types(self, args: list[str] = []) -> None:
        """Show every task type available in the library."""
        from argos.tasks.library import TaskLibrary
        lib = TaskLibrary.get_instance()

        t = Table(show_header=True, header_style=f"bold {CYAN}",
                  border_style=SILVER, box=_box(), expand=False, min_width=68)
        t.add_column("#",         style=DIM,    width=3,  no_wrap=True)
        t.add_column("Type",      style=f"bold {SILVER}", no_wrap=True)
        t.add_column("Kind",      style=DIM,    no_wrap=True)
        t.add_column("Policy",    style=CYAN)
        t.add_column("Min robots",style=DIM,    no_wrap=True)

        for i, name in enumerate(sorted(lib.list_types()), 1):
            cfg  = lib.get_config(name)
            kind = f"[{YELLOW}]cooperative[/]" if lib.is_cooperative(name) else "solo"
            t.add_row(str(i), name,
                      kind,
                      cfg.get("policy", "—"),
                      str(lib.min_robots(name)))

        console.print(Panel(t, title=f"[bold {CYAN}]Available Task Types[/]",
                            border_style=SILVER))
        console.print(f"  [{DIM}]Use [bold]task create <type>[/] to schedule directly, "
                      f"or [bold]task build[/] for the guided wizard.[/]\n")

    def _cmd_task_create(self, args: list[str]) -> None:
        """
        Direct task scheduling — bypasses LLM decomposition.

        Usage:
          task create <type> [--robot NAME] [--zone ZONE] [--target TARGET]

        Examples:
          task create sweep_floor --robot G1-Alpha --zone A
          task create wipe_surface --robot G1-Alpha --target counter
          task create make_bed --robot G1-Alpha --robot G1-Beta
        """
        if not args or args[0].startswith("--"):
            self._cmd_task_types()
            return

        from argos.tasks.library import TaskLibrary
        lib = TaskLibrary.get_instance()
        task_type = args[0]

        if task_type not in lib.list_types():
            console.print(f"\n  [{RED}]Unknown task type:[/] {task_type}")
            console.print(f"  [{DIM}]Run [bold]task types[/] to see all available types.[/]\n")
            return

        # Parse --robot (can appear multiple times), --zone, --target
        robots_given: list[str] = []
        params: dict = {}
        i = 1
        while i < len(args):
            if args[i] == "--robot" and i + 1 < len(args):
                robots_given.append(args[i + 1]); i += 2
            elif args[i] == "--zone" and i + 1 < len(args):
                params["zone"] = args[i + 1]; i += 2
            elif args[i] == "--target" and i + 1 < len(args):
                params["target"] = args[i + 1]; i += 2
            elif args[i] == "--pos" and i + 1 < len(args):
                params["bed_pos"] = [float(x) for x in args[i + 1].split(",")]; i += 2
            else:
                i += 1

        # Validate / auto-assign robots
        cfg          = lib.get_config(task_type)
        min_r        = lib.min_robots(task_type)
        is_coop      = lib.is_cooperative(task_type)
        known_names  = [r["name"] for r in self._robots]

        if robots_given:
            for rn in robots_given:
                if rn not in known_names:
                    console.print(f"\n  [{RED}]Robot not connected:[/] {rn}  "
                                  f"[{DIM}](connected: {', '.join(known_names) or 'none'})[/]\n")
                    return
            assigned = robots_given
        elif known_names:
            assigned = known_names[:min_r]
        else:
            assigned = ["—"]

        if len(assigned) < min_r:
            console.print(f"\n  [{YELLOW}]Warning:[/] {task_type} requires {min_r} robot(s), "
                          f"only {len(assigned)} assigned.\n")

        # Queue the task
        self._task_counter += 1
        tid       = f"t{self._task_counter:03d}"
        robots_str = " + ".join(assigned)
        self._tasks.append({"id": tid, "name": task_type,
                             "type": "cooperative" if is_coop else "solo",
                             "robots": robots_str, "status": "ACTIVE",
                             "params": params})

        # Update robot state
        for r in self._robots:
            if r["name"] in assigned:
                r["status"] = "CLEANING"
                r["task"]   = task_type

        # Summary panel
        param_lines = "\n".join(f"  [{DIM}]{k}:[/]  {v}" for k, v in params.items()) or \
                      f"  [{DIM}](no extra params)[/]"
        console.print(Panel(
            f"[bold {SILVER}]{task_type}[/]\n"
            f"  [{DIM}]ID:[/]      [{CYAN}]{tid}[/]\n"
            f"  [{DIM}]Robot(s):[/] [{CYAN}]{robots_str}[/]\n"
            f"  [{DIM}]Kind:[/]    {'cooperative' if is_coop else 'solo'}\n"
            f"  [{DIM}]Policy:[/]  {cfg.get('policy','—')}\n"
            f"{param_lines}",
            title=f"[bold {CYAN}]Task Created[/]", border_style=SILVER))
        console.print(f"  [{GREEN}]✓[/] [{CYAN}]{tid}[/] queued → [{CYAN}]{robots_str}[/]\n")

    def _cmd_task_build(self, args: list[str] = []) -> None:
        """Interactive step-by-step task creation wizard."""
        from argos.tasks.library import TaskLibrary
        lib   = TaskLibrary.get_instance()
        types = sorted(lib.list_types())

        console.print()
        console.print(f"[bold {CYAN}]Task Builder[/] [{DIM}]— step-by-step wizard (blank = cancel)[/]\n")

        # ── Step 1: pick task type ────────────────────────────────────────────
        for i, name in enumerate(types, 1):
            coop = f" [{YELLOW}][coop][/]" if lib.is_cooperative(name) else ""
            console.print(f"  [{CYAN}]{i:>2}.[/] [{SILVER}]{name}[/]{coop}")
        console.print()

        raw = console.input(f"  [{CYAN}]Select task type[/] [{DIM}](number or name)[/] [{SILVER}]›[/] ").strip()
        if not raw:
            console.print(f"  [{DIM}]Cancelled.[/]\n"); return

        if raw.isdigit() and 1 <= int(raw) <= len(types):
            task_type = types[int(raw) - 1]
        elif raw in types:
            task_type = raw
        else:
            console.print(f"  [{RED}]Invalid selection.[/]\n"); return

        cfg     = lib.get_config(task_type)
        min_r   = lib.min_robots(task_type)
        is_coop = lib.is_cooperative(task_type)
        console.print(f"\n  [{GREEN}]✓[/] Selected: [{CYAN}]{task_type}[/]  "
                      f"[{DIM}]policy={cfg.get('policy','—')}  "
                      f"min_robots={min_r}[/]\n")

        # ── Step 2: pick robot(s) ─────────────────────────────────────────────
        known = [r["name"] for r in self._robots]
        assigned: list[str] = []

        if not known:
            console.print(f"  [{YELLOW}]No robots connected — task will be queued unassigned.[/]")
            assigned = ["—"]
        else:
            for i, rn in enumerate(known, 1):
                bat = next((r["battery"] for r in self._robots if r["name"] == rn), 0)
                console.print(f"  [{CYAN}]{i}.[/] [{SILVER}]{rn}[/]  [{DIM}]battery {bat:.0f}%[/]")
            console.print()

            need = f"{min_r}" if not is_coop else f"{min_r}+ (cooperative)"
            raw2 = console.input(
                f"  [{CYAN}]Assign robot(s)[/] [{DIM}](comma-separated numbers, need {need})[/] [{SILVER}]›[/] "
            ).strip()
            if not raw2:
                console.print(f"  [{DIM}]Cancelled.[/]\n"); return

            for tok in raw2.split(","):
                tok = tok.strip()
                if tok.isdigit() and 1 <= int(tok) <= len(known):
                    assigned.append(known[int(tok) - 1])
                elif tok in known:
                    assigned.append(tok)

            if not assigned:
                console.print(f"  [{RED}]No valid robots selected.[/]\n"); return
            if len(assigned) < min_r:
                console.print(f"  [{YELLOW}]Warning:[/] {task_type} needs {min_r} robot(s), "
                              f"got {len(assigned)}.")

        console.print(f"\n  [{GREEN}]✓[/] Robot(s): [{CYAN}]{', '.join(assigned)}[/]\n")

        # ── Step 3: params ────────────────────────────────────────────────────
        params: dict = {}
        PARAM_PROMPTS = {
            "sweep_floor":    [("zone", "Zone label (e.g. A, B)")],
            "vacuum_floor":   [("zone", "Zone label")],
            "mop_floor":      [("zone", "Zone label")],
            "wipe_surface":   [("target", "Surface target (e.g. counter, table)")],
            "wipe_window":    [("target", "Window label (e.g. north, living-room)")],
            "pick_up_object": [("object", "Object name to pick up")],
            "sort_items":     [("source", "Source location"), ("destination", "Destination")],
            "take_out_trash": [("bin_location", "Bin location (e.g. kitchen)")],
            "make_bed":       [("bed_pos", "Bed centre x,y (e.g. 2.0,1.5)")],
            "change_sheets":  [("bed_pos", "Bed centre x,y")],
            "move_furniture": [("furniture", "Furniture name"), ("destination", "Move to x,y")],
            "organize_shelf": [("shelf", "Shelf label or location")],
        }

        prompts = PARAM_PROMPTS.get(task_type, [])
        if prompts:
            console.print(f"  [{DIM}]Parameters (press Enter to skip):[/]\n")
            for key, label in prompts:
                val = console.input(f"  [{CYAN}]{label}[/] [{SILVER}]›[/] ").strip()
                if val:
                    if key == "bed_pos" and "," in val:
                        params[key] = [float(x) for x in val.split(",")]
                    else:
                        params[key] = val
            console.print()

        # ── Step 4: confirm ───────────────────────────────────────────────────
        param_str = "  ".join(f"{k}={v}" for k, v in params.items()) or "(none)"
        console.print(Panel(
            f"[bold {SILVER}]{task_type}[/]\n"
            f"  [{DIM}]Robot(s):[/] [{CYAN}]{', '.join(assigned)}[/]\n"
            f"  [{DIM}]Params:[/]   {param_str}",
            title=f"[bold {CYAN}]Confirm Task[/]", border_style=SILVER))

        confirm = console.input(f"  [{CYAN}]Queue this task?[/] [{DIM}](y/n)[/] [{SILVER}]›[/] ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print(f"  [{DIM}]Cancelled.[/]\n"); return

        # Queue it
        self._task_counter += 1
        tid        = f"t{self._task_counter:03d}"
        robots_str = " + ".join(assigned)
        self._tasks.append({"id": tid, "name": task_type,
                             "type": "cooperative" if is_coop else "solo",
                             "robots": robots_str, "status": "ACTIVE",
                             "params": params})
        for r in self._robots:
            if r["name"] in assigned:
                r["status"] = "CLEANING"
                r["task"]   = task_type

        console.print(f"\n  [{GREEN}]✓[/] [{CYAN}]{tid}[/] queued → [{CYAN}]{robots_str}[/]\n")

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

    def _cmd_battery(self, args: list[str] = []) -> None:
        from argos.comm.battery import ChargeState
        console.print()
        if not self._robots:
            console.print(f"  [{YELLOW}]No robots connected.[/]\n")
            return

        t = Table(show_header=True, header_style=f"bold {CYAN}",
                  border_style=SILVER, box=_box(), expand=False, min_width=62)
        t.add_column("Robot", style=f"bold {CYAN}", no_wrap=True)
        t.add_column("Battery", no_wrap=True)
        t.add_column("State", no_wrap=True)
        t.add_column("Est. remaining", style=DIM)
        t.add_column("Dock", style=DIM)

        state_style = {
            "nominal":  f"[{GREEN}]● NOMINAL[/]",
            "low":      f"[{YELLOW}]▲ LOW[/]",
            "critical": f"[{RED}]✖ CRITICAL[/]",
            "charging": f"[{CYAN}]⚡ CHARGING[/]",
            "full":     f"[{GREEN}]✓ FULL[/]",
        }

        for r in self._robots:
            bat = r.get("battery", 100.0)
            if bat >= 98:
                state = "full"
            elif bat < 15:
                state = "critical"
            elif bat < 40:
                state = "low"
            else:
                state = "nominal"

            mins = bat / 0.8 if state not in ("charging", "full") else float("inf")
            mins_str = f"{mins:.0f} min" if mins != float("inf") else "—"
            dock = r.get("dock", "—")

            t.add_row(r["name"], _battery_bar(bat),
                      state_style.get(state, state), mins_str, dock)

        console.print(Panel(t, title=f"[bold {CYAN}]Battery Status[/]",
                            border_style=SILVER))
        console.print()

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


# ── Demo REPL ─────────────────────────────────────────────────────────────────

import threading
import random


class DemoArgosREPL(ArgosREPL):
    """
    Identical to ArgosREPL but pre-loaded with two simulated G1 robots
    and a set of realistic tasks. A background thread slowly drains battery,
    advances task progress, and emits log events so the demo feels live.
    """

    _LOG_EVENTS = [
        ("G1-Alpha", "Waypoint 12/24 reached in zone A"),
        ("G1-Beta",  "Cooperative sync signal sent to G1-Alpha"),
        ("G1-Alpha", "Surface wipe cycle 3/5 complete"),
        ("G1-Beta",  "Battery check — 68% remaining"),
        ("Swarm",    "Auction re-run: vacuum_rug allocated to G1-Alpha"),
        ("G1-Alpha", "Obstacle detected — rerouting"),
        ("G1-Beta",  "Sheet grip confirmed — executing pull"),
        ("G1-Alpha", "Zone A coverage: 78%"),
        ("Swarm",    "PEFA phase: EXECUTE — both robots synchronized"),
        ("G1-Beta",  "Pillow placement complete"),
        ("G1-Alpha", "sweep_floor zone A → DONE"),
        ("Swarm",    "Task t002 progressing — ETA 42s"),
    ]

    def __init__(self) -> None:
        super().__init__()

        # Pre-populate two connected robots
        self._robots = [
            {"name": "G1-Alpha", "ip": "192.168.1.10", "status": "CLEANING",
             "battery": 87.0, "task": "wipe_surface", "zone": "A"},
            {"name": "G1-Beta",  "ip": "192.168.1.11", "status": "CLEANING",
             "battery": 72.0, "task": "make_bed",     "zone": "B"},
        ]

        # Pre-populate a realistic task history
        self._tasks = [
            {"id": "t001", "name": "sweep_floor",  "type": "solo",
             "robots": "G1-Alpha", "status": "DONE"},
            {"id": "t002", "name": "wipe_surface", "type": "solo",
             "robots": "G1-Alpha", "status": "ACTIVE"},
            {"id": "t003", "name": "make_bed",     "type": "cooperative",
             "robots": "G1-Alpha + G1-Beta", "status": "ACTIVE"},
            {"id": "t004", "name": "vacuum_rug",   "type": "solo",
             "robots": "G1-Alpha", "status": "PENDING"},
        ]
        self._task_counter = 4
        self._log: list[str] = []
        self._log_idx = 0
        self._stop_sim = threading.Event()

    def run(self) -> None:
        # Start background simulation thread
        sim = threading.Thread(target=self._simulate, daemon=True)
        sim.start()
        try:
            super().run()
        finally:
            self._stop_sim.set()

    # ── background simulation ─────────────────────────────────────────────────

    def _simulate(self) -> None:
        tick = 0
        while not self._stop_sim.is_set():
            time.sleep(4)
            tick += 1

            # Drain battery slowly
            for r in self._robots:
                if r["status"] == "CLEANING":
                    r["battery"] = max(0.0, r["battery"] - random.uniform(0.3, 0.8))
                    if r["battery"] < 15:
                        r["status"] = "IDLE"
                        r["task"] = "—"

            # Advance wipe_surface → DONE after ~20s, then start vacuum_rug
            if tick == 5:
                for t in self._tasks:
                    if t["id"] == "t002":
                        t["status"] = "DONE"
                for r in self._robots:
                    if r["name"] == "G1-Alpha":
                        r["task"] = "vacuum_rug"
                for t in self._tasks:
                    if t["id"] == "t004":
                        t["status"] = "ACTIVE"

            # Advance make_bed → DONE after ~36s
            if tick == 9:
                for t in self._tasks:
                    if t["id"] == "t003":
                        t["status"] = "DONE"
                for r in self._robots:
                    if r["name"] == "G1-Beta":
                        r["task"] = "—"
                        r["status"] = "IDLE"

            # Cycle log events (shown on next `status` call)
            event_robot, event_msg = self._LOG_EVENTS[self._log_idx % len(self._LOG_EVENTS)]
            self._log.append(f"[{event_robot}] {event_msg}")
            if len(self._log) > 20:
                self._log.pop(0)
            self._log_idx += 1

    # ── override status to also show live log ─────────────────────────────────

    def _cmd_status(self, args: list[str] = []) -> None:
        super()._cmd_status(args)
        if self._log:
            from rich.text import Text
            log_text = "\n".join(
                f"  [{DIM}]·[/] [{SILVER}]{line}[/]"
                for line in self._log[-6:]
            )
            console.print(Panel(log_text, title=f"[bold {CYAN}]Live Log[/]",
                                border_style=SILVER))
            console.print()
