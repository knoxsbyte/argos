"""
argos.training.sim.mujoco_env — MuJoCo cleaning environment for ARGOS.

Implements a gymnasium-compatible Env for a room with a Unitree G1 robot.
When mujoco is not installed the environment runs in mock mode, returning
synthetic observations and simulating physics via simple heuristics so
the rest of the training pipeline can be exercised on any machine.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import mujoco  # type: ignore[import]
    _MUJOCO_AVAILABLE = True
    logger.info("MuJoCo detected — full physics simulation active.")
except ImportError:
    mujoco = None  # type: ignore[assignment]
    _MUJOCO_AVAILABLE = False
    logger.warning("mujoco not installed — CleaningEnv running in mock mode.")

try:
    import gymnasium as gym  # type: ignore[import]
    from gymnasium import spaces  # type: ignore[import]
    _GYM_AVAILABLE = True
except ImportError:
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — CleaningEnv will not subclass gym.Env.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_DIM = 29          # G1 joint positions
_ACTION_DIM = 29         # joint position targets
_IMG_H = 224
_IMG_W = 224
_N_SUBSTEPS = 5          # physics substeps per env step
_MAX_STEPS_DEFAULT = 1000

# Reward weights
_W_COVERAGE = 2.0
_W_PICKUP = 1.0
_W_COLLISION = -0.5
_W_SUCCESS = 10.0

# G1 standing pose (zeros = neutral for all 29 DoF)
_STANDING_QPOS = np.zeros(_STATE_DIM, dtype=np.float32)

# Dirt particle count for floor coverage simulation
_N_DIRT_PARTICLES = 200


# ---------------------------------------------------------------------------
# MJCF XML generation
# ---------------------------------------------------------------------------

_G1_BODY = """\
  <body name="torso" pos="0 0 0.9">
    <freejoint name="root"/>
    <geom name="torso_geom" type="box" size="0.15 0.1 0.25" rgba="0.3 0.3 0.8 1"/>
    <!-- Left arm -->
    <body name="left_upper_arm" pos="0.0 0.2 0.1">
      <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <joint name="left_shoulder_roll"  type="hinge" axis="1 0 0" range="-1.57 1.57"/>
      <joint name="left_shoulder_yaw"   type="hinge" axis="0 0 1" range="-1.57 1.57"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" rgba="0.3 0.3 0.8 1"/>
      <body name="left_lower_arm" pos="0 0 -0.3">
        <joint name="left_elbow" type="hinge" axis="0 1 0" range="-2.5 0.0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.3 0.3 0.8 1"/>
        <body name="left_hand" pos="0 0 -0.25">
          <joint name="left_wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <joint name="left_wrist_roll"  type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <joint name="left_wrist_yaw"   type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom type="box" size="0.04 0.03 0.02" rgba="0.8 0.8 0.8 1"/>
        </body>
      </body>
    </body>
    <!-- Right arm -->
    <body name="right_upper_arm" pos="0.0 -0.2 0.1">
      <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <joint name="right_shoulder_roll"  type="hinge" axis="1 0 0" range="-1.57 1.57"/>
      <joint name="right_shoulder_yaw"   type="hinge" axis="0 0 1" range="-1.57 1.57"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" rgba="0.3 0.3 0.8 1"/>
      <body name="right_lower_arm" pos="0 0 -0.3">
        <joint name="right_elbow" type="hinge" axis="0 1 0" range="-2.5 0.0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.3 0.3 0.8 1"/>
        <body name="right_hand" pos="0 0 -0.25">
          <joint name="right_wrist_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <joint name="right_wrist_roll"  type="hinge" axis="1 0 0" range="-1.57 1.57"/>
          <joint name="right_wrist_yaw"   type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom type="box" size="0.04 0.03 0.02" rgba="0.8 0.8 0.8 1"/>
        </body>
      </body>
    </body>
    <!-- Torso joints -->
    <joint name="torso_pitch" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
    <joint name="torso_roll"  type="hinge" axis="1 0 0" range="-0.3 0.3"/>
    <joint name="torso_yaw"   type="hinge" axis="0 0 1" range="-1.57 1.57"/>
    <!-- Legs (simplified) -->
    <body name="left_thigh" pos="0 0.1 -0.25">
      <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-2.0 0.5"/>
      <joint name="left_hip_roll"  type="hinge" axis="1 0 0" range="-0.5 0.5"/>
      <joint name="left_hip_yaw"   type="hinge" axis="0 0 1" range="-0.5 0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" rgba="0.3 0.3 0.8 1"/>
      <body name="left_shin" pos="0 0 -0.35">
        <joint name="left_knee" type="hinge" axis="0 1 0" range="-0.1 2.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.33" size="0.04" rgba="0.3 0.3 0.8 1"/>
        <body name="left_foot" pos="0 0 -0.33">
          <joint name="left_ankle_pitch" type="hinge" axis="0 1 0" range="-0.7 0.7"/>
          <joint name="left_ankle_roll"  type="hinge" axis="1 0 0" range="-0.3 0.3"/>
          <geom type="box" size="0.1 0.05 0.02" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
    </body>
    <body name="right_thigh" pos="0 -0.1 -0.25">
      <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-2.0 0.5"/>
      <joint name="right_hip_roll"  type="hinge" axis="1 0 0" range="-0.5 0.5"/>
      <joint name="right_hip_yaw"   type="hinge" axis="0 0 1" range="-0.5 0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" rgba="0.3 0.3 0.8 1"/>
      <body name="right_shin" pos="0 0 -0.35">
        <joint name="right_knee" type="hinge" axis="0 1 0" range="-0.1 2.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.33" size="0.04" rgba="0.3 0.3 0.8 1"/>
        <body name="right_foot" pos="0 0 -0.33">
          <joint name="right_ankle_pitch" type="hinge" axis="0 1 0" range="-0.7 0.7"/>
          <joint name="right_ankle_roll"  type="hinge" axis="1 0 0" range="-0.3 0.3"/>
          <geom type="box" size="0.1 0.05 0.02" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
    </body>
  </body>
"""

_ROOM_FURNITURE: dict[str, str] = {
    "simple": "",
    "bedroom": """\
    <body name="bed" pos="2 0 0.4">
      <geom type="box" size="1.0 0.6 0.3" rgba="0.6 0.4 0.2 1" contype="1" conaffinity="1"/>
    </body>
    <body name="nightstand" pos="1 0.8 0.4">
      <geom type="box" size="0.2 0.2 0.3" rgba="0.5 0.35 0.15 1" contype="1" conaffinity="1"/>
    </body>""",
    "kitchen": """\
    <body name="counter" pos="2.5 0 0.5">
      <geom type="box" size="0.5 1.0 0.4" rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    </body>
    <body name="sink" pos="2.5 0.5 0.9">
      <geom type="box" size="0.3 0.25 0.05" rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>
    </body>""",
    "living_room": """\
    <body name="couch" pos="2 0 0.4">
      <geom type="box" size="0.8 0.4 0.35" rgba="0.4 0.2 0.1 1" contype="1" conaffinity="1"/>
    </body>
    <body name="coffee_table" pos="1 0 0.25">
      <geom type="box" size="0.4 0.3 0.2" rgba="0.5 0.35 0.1 1" contype="1" conaffinity="1"/>
    </body>""",
}


# ---------------------------------------------------------------------------
# CleaningEnv
# ---------------------------------------------------------------------------


_GymBase = gym.Env if _GYM_AVAILABLE else object


class CleaningEnv(_GymBase):  # type: ignore[misc]
    """MuJoCo environment simulating a room with a Unitree G1 robot.

    Compatible with the gymnasium Env interface.  Falls back to a mock
    physics simulation when MuJoCo is not installed.

    Parameters
    ----------
    task_type:
        One of sweep_floor, vacuum_floor, mop_floor, wipe_surface,
        make_bed, dust_surfaces, empty_trash, tidy_clutter,
        sanitise_surface, generic_cleaning.
    room_layout:
        One of simple, bedroom, kitchen, living_room.
    render_mode:
        "human" for on-screen rendering, "rgb_array" to return images,
        None to disable rendering.
    """

    ROOM_LAYOUTS: dict[str, str] = {
        "simple":      "room_simple.xml",
        "bedroom":     "room_bedroom.xml",
        "kitchen":     "room_kitchen.xml",
        "living_room": "room_living.xml",
    }

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task_type: str = "sweep_floor",
        room_layout: str = "simple",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.task_type = task_type
        self.room_layout = room_layout
        self.render_mode = render_mode

        self._step_count: int = 0
        self._max_steps: int = _MAX_STEPS_DEFAULT
        self._mujoco_available: bool = False
        self._coverage: float = 0.0
        self._objects_removed: int = 0
        self._n_objects: int = 5
        self._collision: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

        # Dirt particles: (N, 2) positions in [0, 1] normalised floor coords
        self._dirt_positions: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._dirt_cleaned: np.ndarray = np.empty(0, dtype=bool)

        # MuJoCo state
        self._mj_model = None
        self._mj_data = None
        self._renderer = None

        if _MUJOCO_AVAILABLE:
            try:
                self._setup_mujoco(room_layout)
                self._mujoco_available = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("MuJoCo setup failed (%s) — using mock physics.", exc)

        if _GYM_AVAILABLE:
            self._setup_spaces()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset environment to a randomised initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._coverage = 0.0
        self._objects_removed = 0
        self._collision = False

        # Scatter dirt particles on floor
        self._dirt_positions = self._rng.uniform(0.0, 1.0, (_N_DIRT_PARTICLES, 2)).astype(np.float32)
        self._dirt_cleaned = np.zeros(_N_DIRT_PARTICLES, dtype=bool)

        # Randomise object count
        self._n_objects = int(self._rng.integers(2, 8))
        self._objects_removed = 0

        if self._mujoco_available and self._mj_model is not None:
            self._reset_mujoco()

        return self.get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Apply action, step physics, compute reward, check termination."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._step_count += 1

        if self._mujoco_available and self._mj_data is not None:
            self._step_mujoco(action)
        else:
            self._step_mock(action)

        reward = self._compute_reward()
        terminated = self.is_success()
        truncated = self._step_count >= self._max_steps

        info: dict = {
            "success": terminated,
            "coverage": round(self._coverage, 3),
            "objects_removed": self._objects_removed,
            "collision": self._collision,
            "step": self._step_count,
        }

        return self.get_observation(), reward, terminated, truncated, info

    def get_observation(self) -> dict:
        """Return the current observation dict."""
        h, w = _IMG_H, _IMG_W

        if self._mujoco_available and self._renderer is not None:
            rgb, depth = self._render_mujoco()
        else:
            rgb = self._mock_render(h, w)
            depth = self._rng.uniform(0.5, 5.0, (h, w)).astype(np.float32)

        robot_state = self._get_robot_state()
        instruction = self._task_instruction()

        return {
            "rgb": rgb,
            "depth": depth,
            "robot_state": robot_state,
            "language_instruction": instruction,
        }

    def is_success(self) -> bool:
        """Task is complete when coverage > 95% and most objects are removed."""
        coverage_done = self._coverage >= 0.95
        objects_done = self._objects_removed >= max(self._n_objects - 1, 0)
        return coverage_done and objects_done

    def render(self) -> np.ndarray | None:
        """Render the current state."""
        if self.render_mode == "rgb_array":
            obs = self.get_observation()
            return obs["rgb"]
        return None

    def close(self) -> None:
        """Release MuJoCo renderer and resources."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001
                pass
            self._renderer = None
        if self._mj_model is not None:
            self._mj_model = None
            self._mj_data = None

    # ------------------------------------------------------------------
    # Gymnasium spaces (set up only when gymnasium is available)
    # ------------------------------------------------------------------

    def _setup_spaces(self) -> None:
        """Define observation and action spaces."""
        self._observation_space = spaces.Dict({
            "rgb": spaces.Box(
                low=0, high=255, shape=(_IMG_H, _IMG_W, 3), dtype=np.uint8
            ),
            "depth": spaces.Box(
                low=0.0, high=10.0, shape=(_IMG_H, _IMG_W), dtype=np.float32
            ),
            "robot_state": spaces.Box(
                low=-math.pi, high=math.pi, shape=(_STATE_DIM,), dtype=np.float32
            ),
            "language_instruction": spaces.Text(max_length=256),
        })
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32
        )

    @property
    def observation_space(self):
        if _GYM_AVAILABLE:
            return self._observation_space
        return {"rgb": (_IMG_H, _IMG_W, 3), "depth": (_IMG_H, _IMG_W),
                "robot_state": (_STATE_DIM,)}

    @property
    def action_space(self):
        if _GYM_AVAILABLE:
            return self._action_space
        return {"shape": (_ACTION_DIM,), "low": -1.0, "high": 1.0}

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """Shaped reward function."""
        reward = 0.0

        # Coverage fraction increase
        prev_coverage = getattr(self, "_prev_coverage", 0.0)
        delta_coverage = max(0.0, self._coverage - prev_coverage)
        reward += _W_COVERAGE * delta_coverage
        self._prev_coverage = self._coverage

        # Object pickup
        prev_removed = getattr(self, "_prev_objects_removed", 0)
        new_pickups = max(0, self._objects_removed - prev_removed)
        reward += _W_PICKUP * new_pickups
        self._prev_objects_removed = self._objects_removed

        # Collision penalty
        if self._collision:
            reward += _W_COLLISION
            self._collision = False

        # Task completion bonus
        if self.is_success():
            reward += _W_SUCCESS

        return float(reward)

    # ------------------------------------------------------------------
    # MuJoCo setup and stepping
    # ------------------------------------------------------------------

    def _setup_mujoco(self, room_layout: str) -> None:
        """Load MJCF XML and create MuJoCo model/data."""
        xml = self._generate_xml(room_layout)
        self._mj_model = mujoco.MjModel.from_xml_string(xml)
        self._mj_data = mujoco.MjData(self._mj_model)

        if self.render_mode in ("human", "rgb_array"):
            try:
                self._renderer = mujoco.Renderer(
                    self._mj_model, height=_IMG_H, width=_IMG_W
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MuJoCo renderer init failed: %s", exc)

    def _reset_mujoco(self) -> None:
        """Reset MuJoCo data to standing pose."""
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        # Set initial joint positions to standing pose
        n_joints = min(len(_STANDING_QPOS), self._mj_data.qpos.shape[0])
        self._mj_data.qpos[:n_joints] = _STANDING_QPOS[:n_joints]
        # Small random perturbation
        noise = self._rng.normal(0, 0.01, self._mj_data.qpos.shape[0])
        self._mj_data.qpos += noise
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def _step_mujoco(self, action: np.ndarray) -> None:
        """Apply action and step MuJoCo physics."""
        n_act = min(len(action), self._mj_data.ctrl.shape[0])
        self._mj_data.ctrl[:n_act] = action[:n_act]
        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self._mj_model, self._mj_data)
        self._update_coverage_mujoco()

    def _update_coverage_mujoco(self) -> None:
        """Update coverage estimate based on end-effector position."""
        # Project end-effector XY to floor coverage grid
        try:
            # Left hand position
            lh_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_hand")
            rh_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")
            lh_pos = self._mj_data.xpos[lh_id][:2]  # XY
            rh_pos = self._mj_data.xpos[rh_id][:2]
            for pos in [lh_pos, rh_pos]:
                self._clean_dirt_near(pos, radius=0.3)
        except Exception:  # noqa: BLE001
            self._update_coverage_mock(np.zeros(_ACTION_DIM))

        cleaned_frac = float(self._dirt_cleaned.sum()) / max(len(self._dirt_cleaned), 1)
        self._coverage = min(1.0, cleaned_frac)

    def _render_mujoco(self) -> tuple[np.ndarray, np.ndarray]:
        """Render RGB and depth from MuJoCo."""
        try:
            self._renderer.update_scene(self._mj_data)
            rgb = self._renderer.render()
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
            return rgb.astype(np.uint8), depth.astype(np.float32)
        except Exception:  # noqa: BLE001
            return self._mock_render(_IMG_H, _IMG_W), np.zeros((_IMG_H, _IMG_W), dtype=np.float32)

    def _generate_xml(self, room_layout: str) -> str:
        """Generate MuJoCo MJCF XML for room + G1 robot."""
        furniture = _ROOM_FURNITURE.get(room_layout, "")
        xml = f"""<mujoco model="argos_cleaning_{room_layout}">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="Newton"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom condim="3" friction="0.8 0.1 0.1"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="5 5" specular="0.3" shininess="0.1"/>
    <material name="wall_mat" rgba="0.9 0.9 0.9 1"/>
  </asset>
  <worldbody>
    <!-- Floor -->
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="floor_mat"
          contype="1" conaffinity="1"/>
    <!-- Walls -->
    <geom name="wall_north" type="box" size="5 0.1 1.5" pos="0 5 1.5" material="wall_mat"
          contype="1" conaffinity="1"/>
    <geom name="wall_south" type="box" size="5 0.1 1.5" pos="0 -5 1.5" material="wall_mat"
          contype="1" conaffinity="1"/>
    <geom name="wall_east"  type="box" size="0.1 5 1.5" pos="5 0 1.5" material="wall_mat"
          contype="1" conaffinity="1"/>
    <geom name="wall_west"  type="box" size="0.1 5 1.5" pos="-5 0 1.5" material="wall_mat"
          contype="1" conaffinity="1"/>
    <!-- Ceiling camera -->
    <camera name="top_camera" pos="0 0 4" quat="0.707 0 0 -0.707" fovy="60"/>
    <!-- Furniture -->
{furniture}
    <!-- G1 Robot -->
{_G1_BODY}
  </worldbody>
  <actuator>
    <!-- One position actuator per joint (29 total) -->
    <position joint="left_shoulder_pitch"  kp="150" gear="1"/>
    <position joint="left_shoulder_roll"   kp="150" gear="1"/>
    <position joint="left_shoulder_yaw"    kp="100" gear="1"/>
    <position joint="left_elbow"           kp="200" gear="1"/>
    <position joint="left_wrist_pitch"     kp="50"  gear="1"/>
    <position joint="left_wrist_roll"      kp="50"  gear="1"/>
    <position joint="left_wrist_yaw"       kp="50"  gear="1"/>
    <position joint="right_shoulder_pitch" kp="150" gear="1"/>
    <position joint="right_shoulder_roll"  kp="150" gear="1"/>
    <position joint="right_shoulder_yaw"   kp="100" gear="1"/>
    <position joint="right_elbow"          kp="200" gear="1"/>
    <position joint="right_wrist_pitch"    kp="50"  gear="1"/>
    <position joint="right_wrist_roll"     kp="50"  gear="1"/>
    <position joint="right_wrist_yaw"      kp="50"  gear="1"/>
    <position joint="torso_pitch"          kp="300" gear="1"/>
    <position joint="torso_roll"           kp="300" gear="1"/>
    <position joint="torso_yaw"            kp="200" gear="1"/>
    <position joint="left_hip_pitch"       kp="400" gear="1"/>
    <position joint="left_hip_roll"        kp="200" gear="1"/>
    <position joint="left_hip_yaw"         kp="150" gear="1"/>
    <position joint="left_knee"            kp="400" gear="1"/>
    <position joint="left_ankle_pitch"     kp="200" gear="1"/>
    <position joint="left_ankle_roll"      kp="100" gear="1"/>
    <position joint="right_hip_pitch"      kp="400" gear="1"/>
    <position joint="right_hip_roll"       kp="200" gear="1"/>
    <position joint="right_hip_yaw"        kp="150" gear="1"/>
    <position joint="right_knee"           kp="400" gear="1"/>
    <position joint="right_ankle_pitch"    kp="200" gear="1"/>
    <position joint="right_ankle_roll"     kp="100" gear="1"/>
  </actuator>
</mujoco>"""
        return xml

    # ------------------------------------------------------------------
    # Mock physics
    # ------------------------------------------------------------------

    def _step_mock(self, action: np.ndarray) -> None:
        """Simulate a physics step without MuJoCo."""
        self._update_coverage_mock(action)

        # Simulate occasional object pickup
        action_mag = float(np.linalg.norm(action))
        if action_mag > 0.5 and self._objects_removed < self._n_objects:
            pickup_prob = 0.005 * action_mag
            if self._rng.random() < pickup_prob:
                self._objects_removed += 1

        # Simulate rare collision
        if self._rng.random() < 0.005:
            self._collision = True

    def _update_coverage_mock(self, action: np.ndarray) -> None:
        """Update dirt coverage in mock mode based on action magnitude."""
        action_mag = float(np.linalg.norm(action))
        # Map action to a pseudo end-effector position
        norm_pos = np.array([
            0.5 + 0.4 * math.sin(self._step_count * 0.1),
            0.5 + 0.4 * math.cos(self._step_count * 0.07),
        ], dtype=np.float32)
        self._clean_dirt_near(norm_pos, radius=0.15 * (1.0 + action_mag))
        cleaned_frac = float(self._dirt_cleaned.sum()) / max(len(self._dirt_cleaned), 1)
        self._coverage = min(1.0, cleaned_frac)

    def _clean_dirt_near(self, pos: np.ndarray, radius: float) -> None:
        """Mark dirt particles within radius of pos as cleaned."""
        if len(self._dirt_positions) == 0:
            return
        # Normalise pos to [0, 1] if it looks like world coords (|pos| > 2)
        if np.max(np.abs(pos)) > 2.0:
            pos = (pos + 5.0) / 10.0  # world [-5, 5] → [0, 1]
        dists = np.linalg.norm(self._dirt_positions - pos, axis=1)
        self._dirt_cleaned |= (dists < radius)

    # ------------------------------------------------------------------
    # Robot state helpers
    # ------------------------------------------------------------------

    def _get_robot_state(self) -> np.ndarray:
        """Return current joint positions as (state_dim,) float32."""
        if self._mujoco_available and self._mj_data is not None:
            qpos = self._mj_data.qpos
            state = np.zeros(_STATE_DIM, dtype=np.float32)
            n = min(_STATE_DIM, len(qpos))
            state[:n] = qpos[:n]
            return state
        return np.zeros(_STATE_DIM, dtype=np.float32)

    def _task_instruction(self) -> str:
        """Return language instruction for current task."""
        _INSTRUCTIONS: dict[str, str] = {
            "sweep_floor":      "Sweep the floor clean.",
            "vacuum_floor":     "Vacuum the floor.",
            "mop_floor":        "Mop the floor.",
            "wipe_surface":     "Wipe the surface clean.",
            "make_bed":         "Make the bed neatly.",
            "dust_surfaces":    "Dust all surfaces.",
            "empty_trash":      "Empty the trash bin.",
            "tidy_clutter":     "Tidy up the clutter.",
            "sanitise_surface": "Sanitise the surface.",
            "generic_cleaning": "Clean the room.",
        }
        return _INSTRUCTIONS.get(self.task_type, "Perform the cleaning task.")

    def _mock_render(self, h: int, w: int) -> np.ndarray:
        """Return a synthetic RGB frame for mock mode."""
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        # Simple floor checkerboard
        tile = 20
        for i in range(0, h, tile * 2):
            for j in range(0, w, tile * 2):
                img[i:i + tile, j:j + tile] = [180, 180, 180]
                img[i + tile:i + 2 * tile, j + tile:j + 2 * tile] = [180, 180, 180]
        # Coverage indicator: green tint proportional to coverage
        green_val = int(self._coverage * 255)
        img[h - 10:, :, 1] = green_val
        return img
