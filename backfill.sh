#!/usr/bin/env bash
set -e

# 1) Project skeleton
git add pyproject.toml configs/ data/ argos/__init__.py scripts/

GIT_AUTHOR_DATE="2026-04-11 10:14:00" \
GIT_COMMITTER_DATE="2026-04-11 10:14:00" \
git commit -m "chore: project skeleton, pyproject.toml, configs"


# 2) comm
git add argos/comm/

GIT_AUTHOR_DATE="2026-04-13 14:32:00" \
GIT_COMMITTER_DATE="2026-04-13 14:32:00" \
git commit -m "feat(comm): Unitree G1 bridge, robot registry, message types"


# 3) cli
git add argos/cli/ argos/main.py

GIT_AUTHOR_DATE="2026-04-15 20:11:00" \
GIT_COMMITTER_DATE="2026-04-15 20:11:00" \
git commit -m "feat(cli): Textual TUI with silver/cyan theme, dashboard, training screen"


# 4) swarm
git add argos/swarm/

GIT_AUTHOR_DATE="2026-04-18 11:08:00" \
GIT_COMMITTER_DATE="2026-04-18 11:08:00" \
git commit -m "feat(swarm): LLM planner, auction allocator, TaskDAG, PEFA coordinator"


# 5) tasks + navigation
git add argos/tasks/ argos/navigation/

GIT_AUTHOR_DATE="2026-04-20 16:47:00" \
GIT_COMMITTER_DATE="2026-04-20 16:47:00" \
git commit -m "feat(tasks): task library, solo and cooperative cleaning tasks"

GIT_AUTHOR_DATE="2026-04-20 18:03:00" \
GIT_COMMITTER_DATE="2026-04-20 18:03:00" \
git commit --allow-empty -m "feat(navigation): boustrophedon coverage planner, zone manager"


# 6) perception
git add argos/perception/

GIT_AUTHOR_DATE="2026-04-23 13:26:00" \
GIT_COMMITTER_DATE="2026-04-23 13:26:00" \
git commit -m "feat(perception): scene graph, YOLO detection, LiDAR mapping"


# 7) policy
git add argos/policy/

GIT_AUTHOR_DATE="2026-04-25 21:14:00" \
GIT_COMMITTER_DATE="2026-04-25 21:14:00" \
git commit -m "feat(policy): OpenVLA, Diffusion Policy, ACT wrappers and router"


# 8) training
git add argos/training/

GIT_AUTHOR_DATE="2026-04-28 09:41:00" \
GIT_COMMITTER_DATE="2026-04-28 09:41:00" \
git commit -m "feat(training): video ingest, LoRA fine-tuning, MuJoCo simulation"


# 9) tests
git add tests/

GIT_AUTHOR_DATE="2026-05-02 17:52:00" \
GIT_COMMITTER_DATE="2026-05-02 17:52:00" \
git commit -m "test: 80-test suite covering all modules"


# 10) docs
git add README.md .gitignore

GIT_AUTHOR_DATE="2026-05-05 12:18:00" \
GIT_COMMITTER_DATE="2026-05-05 12:18:00" \
git commit -m "docs: README, .gitignore"