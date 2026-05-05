# ARGOS — Autonomous Robot Group Operations System

> A fully end-to-end framework for coordinating swarms of **Unitree G1 humanoid robots** to autonomously clean rooms. Robots divide the space into zones and clean independently; for tasks that require both hands or multiple agents (making a bed, moving furniture) they coordinate using a multi-robot synchronization protocol. Includes an ML training pipeline for learning from cleaning video footage and a modern silver/cyan terminal UI.

---

## Features

- **Multi-robot swarm coordination** — 2+ Unitree G1 robots divide a room into zones, clean in parallel, and merge for cooperative tasks
- **LLM-powered task planning** — natural language goals decomposed into task DAGs via Claude API
- **Auction-based task allocation** — market-based bidding assigns tasks optimally based on robot position, battery, and load
- **PEFA cooperative protocol** — Propose→Execute→Feedback→Adjust synchronization for bimanual/multi-robot tasks like bed-making
- **Video-based ML training** — ingest cleaning footage → pose estimation → LeRobot HDF5 dataset → LoRA fine-tuning of OpenVLA/Diffusion Policy/ACT
- **Three policy architectures** — OpenVLA-7B (language-conditioned), Diffusion Policy (multimodal manipulation), ACT (dexterous bimanual)
- **MuJoCo simulation** — test policies before deploying to real hardware, 4 room layouts
- **Modern TUI** — silver/cyan Textual dashboard with real-time fleet monitoring, swarm map, task queue, training progress
- **Full mock fallbacks** — every module works without GPU, robot hardware, or optional deps installed

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARGOS CLI (Textual TUI)                  │
│  argos connect │ argos fleet │ argos task │ argos train │ argos sim│
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Swarm Coordinator                          │
│  LLM Planner → Dependency Graph → Auction Allocator → Monitor  │
└──────┬─────────────────────┬──────────────────────┬────────────┘
       │                     │                      │
┌──────▼──────┐    ┌─────────▼───────┐    ┌────────▼──────────┐
│  Robot A    │    │   Robot B       │    │   Robot N ...     │
│  G1 Bridge  │    │   G1 Bridge     │    │   G1 Bridge       │
│  Policy     │    │   Policy        │    │   Policy          │
│  Navigation │    │   Navigation    │    │   Navigation      │
└──────┬──────┘    └─────────┬───────┘    └────────┬──────────┘
       └─────────────────────┴──────────────────────┘
                        CycloneDDS Mesh
                  (Unitree SDK2 native transport)

┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                            │
│  Video Ingest → Preprocess → HDF5 Dataset → LoRA Fine-tune     │
│  OpenVLA (7B) / Diffusion Policy / ACT → Eval → Deploy         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### Core (always required)
- Python 3.10+
- `pip install -e .`

### Robot hardware
- Unitree G1 (EDU / EDU Ultimate recommended for dexterous hands)
- [`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python) — install from Unitree GitHub
- Both the host machine and robots on the same CycloneDDS network

### ML training (GPU recommended)
```bash
pip install torch transformers peft diffusers accelerate
```

### Full perception stack
```bash
pip install opencv-python mediapipe ultralytics open3d h5py
```

### Simulation
```bash
pip install mujoco gymnasium
# or run: bash scripts/setup_sim.sh
```

### Development
```bash
pip install -e ".[dev]"   # includes pytest, black, ruff
```

> **Note:** All hardware and ML dependencies are optional. The framework runs in mock/simulation mode automatically when they are absent.

---

## Installation

```bash
git clone https://github.com/knoxsbyte/argos.git
cd argos
python -m venv .venv && source .venv/bin/activate
pip install -e .
argos --help
```

### Install on a Unitree G1 robot

```bash
# SSH-deploy the ARGOS agent daemon to the robot's Jetson Orin
argos install --robot 192.168.1.10
# or directly:
bash scripts/install_robot.sh 192.168.1.10
```

---

## Quick Start

### 1. Connect your robots

```bash
argos connect 192.168.1.10 --name G1-Alpha
argos connect 192.168.1.11 --name G1-Beta
```

### 2. Launch the fleet dashboard

```bash
argos fleet
```

```
┌─ ARGOS Fleet ────────────────────────────────── v0.1.0 ──────────┐
│ ┌─ Fleet Status ──────────┐  ┌─ Swarm Map ─────────────────────┐ │
│ │ [G1-Alpha] ● CLEANING   │  │  ┌───────────────────────────┐  │ │
│ │   Battery: 87%          │  │  │   [A]         [B]         │  │ │
│ │   Task: wipe_surface    │  │  │    Zone A  │   Zone B     │  │ │
│ │                         │  │  └───────────────────────────┘  │ │
│ │ [G1-Beta]  ● BED-MAKING │  └─────────────────────────────────┘ │
│ │   Battery: 72%          │  ┌─ Event Log ─────────────────────┐ │
│ │   Task: cooperative     │  │ 14:02:31 G1-Alpha → zone done  │ │
│ └─────────────────────────┘  │ 14:02:45 G1-Beta → bed start   │ │
│ ┌─ Task Queue ────────────┐  │ 14:03:01 Auction: allocated    │ │
│ │ [DONE] sweep_floor      │  └─────────────────────────────────┘ │
│ │ [ACTIVE] wipe_surface   │                                      │
│ │ [ACTIVE] make_bed       │  [q]Quit [t]Tasks [r]Train [?]Help  │
│ │ [QUEUE]  vacuum_rug     │                                      │
└────────────────────────────────────────────────────────────────────┘
```

### 3. Add a cleaning task

```bash
argos task add "clean the bedroom"
# → LLM decomposes into: sweep_floor, wipe_surfaces, make_bed
# → Auction allocates zones to G1-Alpha and G1-Beta
# → Robots execute in parallel; PEFA sync on make_bed
```

---

## CLI Reference

```
argos
├── connect <ip> [--name NAME]          Connect to a G1 robot
├── disconnect <name>                   Disconnect a robot
├── fleet                               Open TUI dashboard
│
├── task
│   ├── add "<natural language goal>"   Decompose & assign task
│   ├── list                            Show all tasks
│   ├── cancel <task-id>                Cancel a running task
│   └── status <task-id>                Show task detail
│
├── train
│   ├── ingest --video-dir DIR          Process footage → HDF5 dataset
│   ├── finetune --dataset DIR          LoRA fine-tune policy
│   │             [--epochs N]
│   ├── evaluate --model PATH           Eval in MuJoCo simulation
│   └── deploy --model PATH             Push checkpoint to robot
│               --robot NAME
│
├── sim
│   ├── start [--env mujoco|isaac]      Launch simulation
│   └── reset                           Reset sim state
│
└── install --robot <ip>                SSH-deploy daemon to Jetson Orin
```

---

## Training Pipeline

Train a policy from your own cleaning video footage:

```bash
# 1. Record cleaning demonstrations (MP4/AVI) and place in ./footage/
#    Name files to hint the task: "sweep_kitchen_01.mp4", "make_bed_01.mp4"
#    Optionally add a metadata.json sidecar for explicit labels.

# 2. Ingest — extract frames, estimate poses, label actions
argos train ingest --video-dir ./footage/
# Output: data/processed/dataset.h5  (LeRobot HDF5 format)

# 3. Fine-tune — LoRA fine-tune OpenVLA-7B on your dataset
argos train finetune --dataset ./data/processed/ --epochs 10
# Output: data/models/checkpoint_epoch_N/

# 4. Evaluate in simulation
argos train evaluate --model ./data/models/best.ckpt

# 5. Deploy to robot
argos train deploy --model ./data/models/best.ckpt --robot G1-Alpha
```

### Supported policy architectures

| Policy | Best for | Params | Training time (RTX 4090) |
|---|---|---|---|
| **OpenVLA** | Language-conditioned tasks (`sort_items`, `organize_shelf`) | 7B (LoRA) | ~3h / 10 epochs |
| **Diffusion Policy** | Coverage tasks (`sweep_floor`, `mop_floor`) | ~80M | ~1h / 10 epochs |
| **ACT** | Dexterous bimanual (`make_bed`, `wipe_surface`) | ~80M | ~45min / 10 epochs |

All support 4-bit quantization via bitsandbytes for 8GB VRAM cards.

---

## Swarm Coordination

### Task allocation (auction-based MRTA)

Each robot bids on every available task. Bid cost is computed from:
- **Distance** — Euclidean distance from robot's current position to task location
- **Battery penalty** — extra cost if battery < 20%
- **Load penalty** — robots with queued tasks bid higher

The lowest-cost robot (or team) wins each task. For cooperative tasks requiring `min_robots ≥ 2`, the cheapest team is selected together.

### Cooperative task protocol (PEFA)

For multi-robot tasks like `make_bed` or `move_furniture`:

1. **Propose** — lead robot computes action plan (grip positions, timing)
2. **Execute** — all robots act simultaneously via `asyncio.gather()`
3. **Feedback** — collect success signals from each robot
4. **Adjust** — if partial failure, adjust plan and retry (max 3 attempts)

### LLM task planning

Natural language goals are decomposed by Claude into a directed acyclic task graph:

```
"clean the bedroom"
        ↓
┌─────────────────────────────────────┐
│  sweep_floor (zone A)  ──────────┐  │
│  sweep_floor (zone B)  ──────────┤  │
│                                  ↓  │
│  wipe_surfaces ──────────→  make_bed │
└─────────────────────────────────────┘
```

Tasks with no pending predecessors are immediately available for auction.

---

## Module Reference

```
argos/
├── comm/           Robot communication (Unitree SDK2, ROS2, registry)
├── swarm/          Coordination (LLM planner, MRTA, PEFA, TaskDAG)
├── tasks/          Task library (12 cleaning tasks, solo + cooperative)
├── navigation/     Path planning (boustrophedon, zones, A* obstacle avoidance)
├── perception/     Scene understanding (YOLO, dirt detection, LiDAR mapping)
├── policy/         Policy inference (OpenVLA, Diffusion Policy, ACT, router)
├── training/       ML pipeline (ingest, preprocess, HDF5, LoRA, eval, MuJoCo sim)
└── cli/            TUI (Textual, silver/cyan theme, dashboard, training screen)
```

### Task library

| Task | Type | Policy | Min robots |
|---|---|---|---|
| `sweep_floor` | solo | Diffusion Policy | 1 |
| `vacuum_floor` | solo | Diffusion Policy | 1 |
| `mop_floor` | solo | Diffusion Policy | 1 |
| `wipe_surface` | solo | ACT | 1 |
| `wipe_window` | solo | ACT | 1 |
| `pick_up_object` | solo | ACT | 1 |
| `sort_items` | solo | OpenVLA | 1 |
| `take_out_trash` | solo | ACT | 1 |
| `make_bed` | cooperative | ACT | 2 |
| `change_sheets` | cooperative | ACT | 2 |
| `move_furniture` | cooperative | Diffusion Policy | 2 |
| `organize_shelf` | solo | OpenVLA | 1 |

---

## Configuration

### Robot config — `configs/robots/g1.yaml`

```yaml
model: unitree_g1_edu_ultimate
dof: 29
communication:
  protocol: cyclonedds
  control_freq: 50          # Hz
sensors:
  camera: intel_realsense_d435
  lidar: livox_mid360
capabilities:
  locomotion_speed: 2.0     # m/s
  payload_kg: 3.0
  dexterous_hands: true
policy:
  default: openvla
  fallback: act
```

### Task config — `configs/tasks/cleaning.yaml`

Add new tasks by extending this file:

```yaml
tasks:
  my_custom_task:
    type: solo              # or cooperative
    policy: act
    min_robots: 1
    duration_estimate: 120  # seconds
    required_tools: [sponge]
    success_criteria:
      coverage_threshold: 0.90
```

---

## Research Foundations

ARGOS is built on current (2024–2025) state-of-the-art research:

| Component | Method | Paper/Source |
|---|---|---|
| Policy learning | Diffusion Policy | [Columbia, IJRR 2024](https://diffusion-policy.cs.columbia.edu/) |
| Bimanual control | ACT | [Zhao et al. 2023](https://tonyzhaozh.github.io/aloha/) |
| Language-conditioned | OpenVLA-7B | [Kim et al. arXiv:2406.09246](https://openvla.github.io/) |
| Fast tokenization | FAST tokenizer | [Physical Intelligence 2024](https://www.pi.website/) |
| Task allocation | Auction-based MRTA | [Zlot & Stentz, CMU 2006](https://www.ri.cmu.edu/) |
| Cooperative tasks | COHERENT PEFA | [arXiv:2409.15146](https://arxiv.org/abs/2409.15146) |
| LLM planning | RobotFleet pattern | [arXiv:2510.10379](https://arxiv.org/abs/2510.10379) |
| Video transfer | H2R augmentation | [arXiv:2505.11920](https://arxiv.org/abs/2505.11920) |
| Robot SDK | Unitree SDK2 | [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) |

---

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check argos/

# Format
black argos/ tests/

# Test without any hardware or GPU
pytest tests/ -v   # all mocks auto-activate
```

### Adding a new task

1. Add an entry to `configs/tasks/cleaning.yaml`
2. Subclass `BaseTask` in `argos/tasks/solo.py` or `argos/tasks/cooperative.py`
3. Register in `TaskLibrary.create()` in `argos/tasks/library.py`
4. Add to `PolicyRouter.TASK_POLICY_MAP` in `argos/policy/router.py`

### Adding a new policy

1. Subclass `BasePolicy` in `argos/policy/`
2. Implement `load()`, `predict()`, `reset()`
3. Register in `PolicyRouter` and update `TASK_POLICY_MAP`

---

## Roadmap

- [ ] Real-time video streaming in TUI
- [ ] Isaac Lab high-fidelity training environment
- [ ] Multi-floor coordination with elevator navigation
- [ ] Battery management & charging dock integration
- [ ] Web dashboard (Textual web mode)
- [ ] Support for additional robot embodiments (Agility Digit, Fourier GR-1)

---

## License

MIT
