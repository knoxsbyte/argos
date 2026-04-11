#!/usr/bin/env bash
# =============================================================================
# ARGOS — Simulation Environment Setup Script
# Installs MuJoCo 3.x, downloads Unitree G1 MJCF models, and optionally
# installs NVIDIA Isaac Lab.
#
# Usage:
#   bash setup_sim.sh [--isaac] [--conda ENV_NAME] [--venv PATH] [--dry-run]
#
# Flags:
#   --isaac            Also install NVIDIA Isaac Lab (requires CUDA + pip)
#   --conda ENV_NAME   Install into an existing conda environment
#   --venv PATH        Install into a Python venv at PATH (created if absent)
#   --dry-run          Print commands without executing them
# =============================================================================

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────
INSTALL_ISAAC=false
CONDA_ENV=""
VENV_PATH=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --isaac)           INSTALL_ISAAC=true; shift ;;
    --conda)           CONDA_ENV="$2"; shift 2 ;;
    --venv)            VENV_PATH="$2"; shift 2 ;;
    --dry-run)         DRY_RUN=true; shift ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

$DRY_RUN && warn "DRY RUN — commands will be printed but not executed."

# ── Helper: run or echo ────────────────────────────────────────────────────────
run() {
  if $DRY_RUN; then
    echo -e "  ${YELLOW}[dry-run]${NC} $*"
  else
    "$@"
  fi
}

# ── Detect OS ─────────────────────────────────────────────────────────────────
UNAME="$(uname -s)"
case "$UNAME" in
  Linux*)   OS=linux ;;
  Darwin*)  OS=macos ;;
  *)        die "Unsupported OS: ${UNAME}" ;;
esac
info "Detected OS: ${OS}"

# ── Detect architecture ───────────────────────────────────────────────────────
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  ARCH_TAG=x86_64 ;;
  aarch64|arm64) ARCH_TAG=aarch64 ;;
  *) die "Unsupported architecture: ${ARCH}" ;;
esac
info "Architecture: ${ARCH_TAG}"

# ── Resolve Python binary ──────────────────────────────────────────────────────
if [[ -n "$CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || die "conda not found in PATH."
  PY="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
  PIP="${CONDA_BASE}/envs/${CONDA_ENV}/bin/pip"
  [[ -f "$PY" ]] || die "Conda env '${CONDA_ENV}' not found. Create it first: conda create -n ${CONDA_ENV} python=3.11"
  info "Using conda env: ${CONDA_ENV}"
elif [[ -n "$VENV_PATH" ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    info "Creating venv at ${VENV_PATH}…"
    run python3 -m venv "$VENV_PATH"
  fi
  PY="${VENV_PATH}/bin/python"
  PIP="${VENV_PATH}/bin/pip"
  info "Using venv: ${VENV_PATH}"
else
  PY="$(command -v python3 || command -v python)" || die "python3 not found in PATH."
  PIP="$(command -v pip3 || command -v pip)"      || die "pip not found in PATH."
  info "Using system Python: $($PY --version 2>&1)"
fi

# ── Check Python version ───────────────────────────────────────────────────────
PY_VER_NUM=$("$PY" -c "import sys; print('%d%02d' % (sys.version_info.major, sys.version_info.minor))")
[[ "$PY_VER_NUM" -ge 310 ]] || die "Python 3.10+ required (found $($PY --version))."
success "Python version OK: $($PY --version)"

# ── Directories ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${REPO_ROOT}/data/models"
MJCF_DIR="${MODELS_DIR}/mjcf"
run mkdir -p "$MJCF_DIR"

# ── Step 1: Install MuJoCo ─────────────────────────────────────────────────────
info "Step 1 — Installing MuJoCo 3.x Python bindings…"
run "$PIP" install --upgrade "mujoco>=3.1"

# Verify
if ! $DRY_RUN; then
  "$PY" -c "import mujoco; print('mujoco', mujoco.__version__)" || die "mujoco import failed."
  success "MuJoCo installed: $($PY -c 'import mujoco; print(mujoco.__version__)')"
else
  success "(dry-run) MuJoCo install skipped."
fi

# ── Step 2: Install gymnasium and supporting sim packages ─────────────────────
info "Step 2 — Installing gymnasium and simulation packages…"
run "$PIP" install --upgrade \
  "gymnasium>=1.0" \
  "stable-baselines3>=2.3" \
  "imageio>=2.34" \
  "imageio-ffmpeg>=0.5" \
  "glfw>=2.6"      # headless rendering support

if ! $DRY_RUN; then
  "$PY" -c "import gymnasium; print('gymnasium', gymnasium.__version__)" || warn "gymnasium import check failed."
fi
success "Gymnasium and simulation packages installed."

# ── Step 3: Download Unitree G1 MJCF model ────────────────────────────────────
info "Step 3 — Downloading Unitree G1 MJCF model…"

G1_MJCF_REPO="https://github.com/unitreerobotics/unitree_mujoco"
G1_MJCF_FALLBACK="https://github.com/google-deepmind/mujoco_menagerie"

if command -v git &>/dev/null; then
  if [[ -d "${MJCF_DIR}/unitree_mujoco/.git" ]]; then
    info "Updating existing unitree_mujoco clone…"
    run git -C "${MJCF_DIR}/unitree_mujoco" pull --ff-only
  else
    info "Cloning ${G1_MJCF_REPO}…"
    run git clone --depth 1 "${G1_MJCF_REPO}" "${MJCF_DIR}/unitree_mujoco"
  fi

  # Also grab MuJoCo Menagerie (contains additional humanoid references)
  if [[ -d "${MJCF_DIR}/mujoco_menagerie/.git" ]]; then
    info "Updating existing mujoco_menagerie clone…"
    run git -C "${MJCF_DIR}/mujoco_menagerie" pull --ff-only
  else
    info "Cloning MuJoCo Menagerie (reference models)…"
    run git clone --depth 1 "${G1_MJCF_FALLBACK}" "${MJCF_DIR}/mujoco_menagerie"
  fi
else
  warn "git not found — skipping MJCF model download. Install git and re-run."
fi

# Validate MJCF exists
if ! $DRY_RUN; then
  G1_XML=$(find "${MJCF_DIR}" -name "g1*.xml" 2>/dev/null | head -1)
  if [[ -n "$G1_XML" ]]; then
    success "G1 MJCF model found: ${G1_XML}"
    # Quick parse check
    "$PY" -c "
import mujoco, sys
try:
    m = mujoco.MjModel.from_xml_path('${G1_XML}')
    print(f'Model OK — {m.nq} DoF, {m.nbody} bodies')
except Exception as e:
    print(f'Model parse warning: {e}', file=sys.stderr)
" || warn "MJCF parse check produced warnings (non-fatal)."
  else
    warn "G1 MJCF XML not found in ${MJCF_DIR} — may need manual download."
  fi
fi

# ── Step 4: Configure MUJOCO_MENAGERIE_PATH ───────────────────────────────────
info "Step 4 — Writing sim environment config…"
SIM_ENV_FILE="${REPO_ROOT}/.env.sim"
if ! $DRY_RUN; then
  cat > "$SIM_ENV_FILE" << EOF
# ARGOS simulation environment — auto-generated by setup_sim.sh
export MUJOCO_MENAGERIE_PATH=${MJCF_DIR}/mujoco_menagerie
export UNITREE_MUJOCO_PATH=${MJCF_DIR}/unitree_mujoco
export ARGOS_SIM_BACKEND=mujoco
export MUJOCO_GL=egl        # use 'glfw' for windowed rendering
EOF
  success "Sim config written to ${SIM_ENV_FILE}"
  info "To activate: source ${SIM_ENV_FILE}"
else
  echo -e "  ${YELLOW}[dry-run]${NC} Would write ${SIM_ENV_FILE}"
fi

# ── Step 5 (optional): Install Isaac Lab ──────────────────────────────────────
if $INSTALL_ISAAC; then
  info "Step 5 — Installing NVIDIA Isaac Lab (this may take 10–30 min)…"

  # Check CUDA
  if ! command -v nvcc &>/dev/null && ! nvidia-smi &>/dev/null 2>&1; then
    die "CUDA / NVIDIA driver not detected. Isaac Lab requires a CUDA-capable GPU."
  fi

  CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1 || echo "unknown")
  info "CUDA version: ${CUDA_VER}"

  # Clone Isaac Lab
  ISAAC_DIR="${MODELS_DIR}/IsaacLab"
  if [[ -d "${ISAAC_DIR}/.git" ]]; then
    info "Updating existing Isaac Lab clone…"
    run git -C "${ISAAC_DIR}" pull --ff-only
  else
    run git clone --depth 1 "https://github.com/isaac-sim/IsaacLab.git" "${ISAAC_DIR}"
  fi

  info "Running Isaac Lab installer…"
  run bash "${ISAAC_DIR}/isaaclab.sh" --install

  # Install Isaac Sim Python bindings via pip (Isaac Sim 4.x pip wheel)
  run "$PIP" install \
    isaacsim-rl \
    isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    --extra-index-url https://pypi.nvidia.com

  success "Isaac Lab installation complete."
  info "To launch Isaac Lab: cd ${ISAAC_DIR} && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py"
else
  info "Step 5 — Skipping Isaac Lab (pass --isaac to enable)."
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
success "=== ARGOS simulation environment setup complete ==="
info "Source the sim config before running simulations:"
info "  source ${REPO_ROOT}/.env.sim"
info ""
info "Quick smoke test:"
info "  python3 -c \"import mujoco, gymnasium; print('Sim stack OK')\""
