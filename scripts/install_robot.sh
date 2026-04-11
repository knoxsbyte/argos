#!/usr/bin/env bash
# =============================================================================
# ARGOS — Robot Installation Script
# Deploys the argos package and policy daemon onto a Unitree G1 Jetson Orin.
#
# Usage:
#   ROBOT_IP=192.168.123.161 bash install_robot.sh [--dry-run] [--skip-service]
#
# Environment variables:
#   ROBOT_IP          Required. IP address of the target robot.
#   ROBOT_USER        SSH user (default: unitree)
#   ROBOT_PORT        SSH port (default: 22)
#   ARGOS_VERSION     pip version specifier (default: installs from local source)
#   PYTHON_BIN        Python binary on remote (default: python3)
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_SERVICE=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY_RUN=true ;;
    --skip-service) SKIP_SERVICE=true ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ── Required configuration ─────────────────────────────────────────────────────
: "${ROBOT_IP:?'ROBOT_IP environment variable is required. E.g. ROBOT_IP=192.168.123.161'}"
ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_PORT="${ROBOT_PORT:-22}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${ROBOT_PORT}"
SSH_TARGET="${ROBOT_USER}@${ROBOT_IP}"

info "Target robot : ${SSH_TARGET} (port ${ROBOT_PORT})"
info "Repo root    : ${REPO_ROOT}"
$DRY_RUN && warn "DRY RUN — no remote commands will be executed."

# ── Helper: run remote command (or echo in dry-run) ────────────────────────────
remote() {
  local cmd="$1"
  if $DRY_RUN; then
    echo -e "  ${YELLOW}[dry-run]${NC} ssh ${SSH_TARGET} \"${cmd}\""
  else
    # shellcheck disable=SC2029
    ssh $SSH_OPTS "${SSH_TARGET}" "${cmd}"
  fi
}

# ── Helper: copy local → remote ───────────────────────────────────────────────
push() {
  local src="$1" dst="$2"
  if $DRY_RUN; then
    echo -e "  ${YELLOW}[dry-run]${NC} scp ${src} ${SSH_TARGET}:${dst}"
  else
    scp -P "${ROBOT_PORT}" -o StrictHostKeyChecking=no -r "${src}" "${SSH_TARGET}:${dst}"
  fi
}

# ── Step 1: connectivity check ────────────────────────────────────────────────
info "Step 1/6 — Checking SSH connectivity to ${ROBOT_IP}…"
if ! $DRY_RUN; then
  ssh $SSH_OPTS "${SSH_TARGET}" "echo 'pong'" > /dev/null 2>&1 || \
    die "Cannot SSH to ${SSH_TARGET}. Check ROBOT_IP, ROBOT_USER, and that your SSH key is authorised."
fi
success "SSH connection OK."

# ── Step 2: prerequisite checks ──────────────────────────────────────────────
info "Step 2/6 — Checking prerequisites (Python 3.10+, pip, git)…"

remote "
set -e
# Python version
PY_VER=\$(${PYTHON_BIN} -c 'import sys; print(\"%d%02d\" % (sys.version_info.major, sys.version_info.minor))')
if [ \"\$PY_VER\" -lt 310 ]; then
  echo 'ERROR: Python 3.10 or newer is required (found '\$(${PYTHON_BIN} --version)')' >&2
  exit 1
fi
echo \"Python OK: \$(${PYTHON_BIN} --version)\"

# pip
${PYTHON_BIN} -m pip --version > /dev/null 2>&1 || {
  echo 'pip not found — installing via ensurepip…'
  ${PYTHON_BIN} -m ensurepip --upgrade
}
echo \"pip OK: \$(${PYTHON_BIN} -m pip --version | cut -d' ' -f1-2)\"

# git (optional, used for source installs)
git --version > /dev/null 2>&1 && echo \"git OK\" || echo 'git not found (non-fatal)'
"
success "Prerequisites satisfied."

# ── Step 3: create remote directory structure ─────────────────────────────────
info "Step 3/6 — Creating remote directories…"
remote "mkdir -p /opt/argos/{configs,data/models,logs}"
success "Remote directories ready."

# ── Step 4: push source and install package ───────────────────────────────────
info "Step 4/6 — Deploying argos package to robot…"

if [[ -n "${ARGOS_VERSION:-}" ]]; then
  info "Installing from PyPI: argos==${ARGOS_VERSION}"
  remote "${PYTHON_BIN} -m pip install --upgrade argos==${ARGOS_VERSION}"
else
  info "Building sdist from local source and pushing…"
  # Build wheel locally (requires hatchling)
  (cd "${REPO_ROOT}" && python3 -m build --wheel --outdir /tmp/argos_dist/ 2>&1 | tail -5) || \
    die "Local build failed. Run: pip install build"
  WHEEL=$(ls /tmp/argos_dist/argos-*.whl 2>/dev/null | sort -V | tail -1)
  [[ -f "$WHEEL" ]] || die "No wheel found in /tmp/argos_dist/"
  info "Pushing wheel: $(basename "$WHEEL")"
  push "$WHEEL" "/tmp/"
  remote "${PYTHON_BIN} -m pip install --upgrade /tmp/$(basename "$WHEEL")"
fi

# Push configs
push "${REPO_ROOT}/configs/" "/opt/argos/"
success "Package installed on robot."

# ── Step 5: set environment variables ─────────────────────────────────────────
info "Step 5/6 — Configuring environment variables…"
remote "
PROFILE=/etc/profile.d/argos.sh
cat > \"\${PROFILE}\" << 'ENVEOF'
# ARGOS environment — auto-generated by install_robot.sh
export ARGOS_HOME=/opt/argos
export ARGOS_CONFIG_DIR=\${ARGOS_HOME}/configs
export ARGOS_DATA_DIR=\${ARGOS_HOME}/data
export ARGOS_LOG_DIR=\${ARGOS_HOME}/logs
export ARGOS_ROBOT_MODEL=g1
ENVEOF
chmod 644 \"\${PROFILE}\"
echo 'Environment profile written to '\${PROFILE}
"
success "Environment variables configured."

# ── Step 6: install systemd service ──────────────────────────────────────────
if $SKIP_SERVICE; then
  warn "Skipping systemd service installation (--skip-service)."
else
  info "Step 6/6 — Installing argos-policy systemd service…"
  POLICY_DAEMON_BIN=$(remote "${PYTHON_BIN} -c 'import shutil; print(shutil.which(\"argos\") or \"/usr/local/bin/argos\")'")

  remote "
UNIT=/etc/systemd/system/argos-policy.service
cat > \"\${UNIT}\" << 'UNITEOF'
[Unit]
Description=ARGOS Policy Daemon — Unitree G1 inference service
After=network.target
Wants=network.target

[Service]
Type=simple
User=unitree
Group=unitree
EnvironmentFile=-/etc/profile.d/argos.sh
Environment=PYTHONUNBUFFERED=1
WorkingDirectory=/opt/argos
ExecStart=/usr/local/bin/argos policy serve --host 0.0.0.0 --port 9090
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=argos-policy

[Install]
WantedBy=multi-user.target
UNITEOF

systemctl daemon-reload
systemctl enable argos-policy.service
systemctl restart argos-policy.service
echo 'Service status:'
systemctl is-active argos-policy.service || true
"
  success "argos-policy service installed and started."
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
success "=== ARGOS installation complete on ${ROBOT_IP} ==="
info "To check service health:  ssh ${SSH_TARGET} systemctl status argos-policy"
info "To view logs:             ssh ${SSH_TARGET} journalctl -u argos-policy -f"
info "To uninstall:             ssh ${SSH_TARGET} 'pip uninstall -y argos && systemctl disable argos-policy'"
