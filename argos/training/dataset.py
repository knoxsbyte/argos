"""
argos.training.dataset — LeRobot-format HDF5 dataset builder for ARGOS.

Writes episode data (images, robot state, actions, language instructions)
into an HDF5 file following the LeRobot dataset schema so that the resulting
file can be consumed directly by the LeRobot training framework or any custom
PyTorch Dataset that understands the schema.

LeRobot HDF5 structure:
  /data/
    episode_0/
      observation.images.top/    (T, H, W, 3) uint8
      observation.state/         (T, state_dim) float32
      action/                    (T, action_dim) float32
    episode_1/
      ...
  /meta/
    episode_lengths/             (N,) int64
    task_types/                  variable-length string dataset (N,)
    fps                          float attribute
    robot_model                  string attribute
    state_dim                    int attribute
    action_dim                   int attribute
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from argos.training.ingest import Episode

logger = logging.getLogger(__name__)

try:
    import h5py  # type: ignore[import]
    _H5PY_AVAILABLE = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    _H5PY_AVAILABLE = False
    logger.warning("h5py not installed. LeRobotDatasetBuilder running in mock mode.")

_ROBOT_MODEL = "unitree_g1"


# ---------------------------------------------------------------------------
# Helper: variable-length string dtype for h5py
# ---------------------------------------------------------------------------

def _vlen_str_dtype() -> "h5py.special_dtype":
    return h5py.special_dtype(vlen=str)


# ---------------------------------------------------------------------------
# LeRobotDatasetBuilder
# ---------------------------------------------------------------------------


class LeRobotDatasetBuilder:
    """Builds an HDF5 dataset in LeRobot format from Episode objects.

    When h5py is not installed, build() writes a JSON manifest instead so
    the rest of the pipeline can be exercised without the library.
    """

    def __init__(
        self,
        output_path: Path,
        state_dim: int = 29,
        action_dim: int = 29,
    ) -> None:
        self.output_path = Path(output_path)
        self.state_dim = state_dim
        self.action_dim = action_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        episodes: list["Episode"],
        actions_per_episode: list[np.ndarray],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Write all episodes to HDF5 (or JSON fallback).

        Parameters
        ----------
        episodes:
            List of Episode objects from VideoIngestor.
        actions_per_episode:
            Parallel list; actions_per_episode[i] is a (T_i, action_dim)
            float32 array produced by ActionLabeler.label_segment().
        progress_callback:
            Optional callback(current_episode_idx, total_episodes).

        Returns
        -------
        Path to the written file.
        """
        if len(episodes) != len(actions_per_episode):
            raise ValueError(
                f"episodes ({len(episodes)}) and actions_per_episode "
                f"({len(actions_per_episode)}) must have the same length."
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not _H5PY_AVAILABLE:
            return self._build_json_fallback(episodes, actions_per_episode, progress_callback)

        fps = episodes[0].fps if episodes else 15.0

        with h5py.File(self.output_path, "w") as f:
            data_grp = f.create_group("data")
            meta_grp = f.create_group("meta")

            episode_lengths: list[int] = []
            task_types: list[str] = []

            for i, (ep, actions) in enumerate(zip(episodes, actions_per_episode)):
                self.add_episode(f, episode_idx=i, episode=ep, actions=actions)
                episode_lengths.append(len(ep.frames))
                task_types.append(ep.metadata.get("task_type", "generic_cleaning"))

                if progress_callback is not None:
                    progress_callback(i + 1, len(episodes))

                logger.debug("Wrote episode %d (%d frames).", i, len(ep.frames))

            # /meta datasets
            meta_grp.create_dataset(
                "episode_lengths",
                data=np.array(episode_lengths, dtype=np.int64),
            )
            dt = _vlen_str_dtype()
            tt_ds = meta_grp.create_dataset(
                "task_types",
                (len(task_types),),
                dtype=dt,
            )
            for i, t in enumerate(task_types):
                tt_ds[i] = t

            # /meta attributes
            meta_grp.attrs["fps"] = fps
            meta_grp.attrs["robot_model"] = _ROBOT_MODEL
            meta_grp.attrs["state_dim"] = self.state_dim
            meta_grp.attrs["action_dim"] = self.action_dim
            meta_grp.attrs["num_episodes"] = len(episodes)
            meta_grp.attrs["total_frames"] = int(sum(episode_lengths))

        logger.info("Dataset written to %s (%d episodes).", self.output_path, len(episodes))
        return self.output_path

    def add_episode(
        self,
        h5_file: "h5py.File",
        episode_idx: int,
        episode: "Episode",
        actions: np.ndarray,
    ) -> None:
        """Write a single episode into an already-open HDF5 file.

        Creates the group ``/data/episode_{episode_idx}`` with:
          - ``observation.images.top`` : (T, H, W, 3) uint8
          - ``observation.state``      : (T, state_dim) float32
          - ``action``                 : (T, action_dim) float32
        and string attribute ``language_instruction``.
        """
        T = len(episode.frames)
        if T == 0:
            logger.warning("Episode %d has 0 frames — skipping.", episode_idx)
            return

        grp = h5_file["data"].create_group(f"episode_{episode_idx}")

        # Images: (T, H, W, 3) uint8
        sample_rgb = episode.frames[0].rgb
        H, W = sample_rgb.shape[:2]
        img_ds = grp.create_dataset(
            "observation.images.top",
            shape=(T, H, W, 3),
            dtype=np.uint8,
            chunks=(1, H, W, 3),
            compression="gzip",
            compression_opts=4,
        )
        for t, frame in enumerate(episode.frames):
            img_ds[t] = frame.rgb

        # State: (T, state_dim) float32 — zero-initialised (no ground-truth state)
        state_ds = grp.create_dataset(
            "observation.state",
            shape=(T, self.state_dim),
            dtype=np.float32,
        )
        state_ds[:] = 0.0  # placeholder; real state from robot telemetry if available

        # Actions: (T, action_dim) float32
        if actions.shape[0] != T:
            # Pad or truncate to match frame count
            if actions.shape[0] < T:
                pad = np.zeros((T - actions.shape[0], self.action_dim), dtype=np.float32)
                actions = np.concatenate([actions, pad], axis=0)
            else:
                actions = actions[:T]
        if actions.shape[1] != self.action_dim:
            # Pad or truncate action dim
            if actions.shape[1] < self.action_dim:
                pad = np.zeros((T, self.action_dim - actions.shape[1]), dtype=np.float32)
                actions = np.concatenate([actions, pad], axis=1)
            else:
                actions = actions[:, :self.action_dim]

        act_ds = grp.create_dataset(
            "action",
            data=actions.astype(np.float32),
        )

        # Attributes
        grp.attrs["language_instruction"] = episode.metadata.get(
            "language_instruction", ""
        )
        grp.attrs["task_type"] = episode.metadata.get("task_type", "generic_cleaning")
        grp.attrs["episode_id"] = episode.episode_id
        grp.attrs["fps"] = episode.fps
        grp.attrs["duration"] = episode.duration

    def validate(self, dataset_path: Path | None = None) -> dict:
        """Check dataset integrity.

        Checks:
        - All episodes have matching frame / action lengths.
        - No NaN or Inf in action arrays.
        - Images are valid uint8 in range [0, 255].

        Returns a report dict with keys ``valid``, ``num_episodes``,
        ``errors``, ``warnings``.
        """
        path = Path(dataset_path) if dataset_path else self.output_path
        report: dict = {"valid": True, "num_episodes": 0, "errors": [], "warnings": []}

        if not _H5PY_AVAILABLE:
            report["warnings"].append("h5py not available — skipping HDF5 validation.")
            return report

        if not path.exists():
            report["valid"] = False
            report["errors"].append(f"File not found: {path}")
            return report

        try:
            with h5py.File(path, "r") as f:
                if "data" not in f or "meta" not in f:
                    report["valid"] = False
                    report["errors"].append("Missing /data or /meta groups.")
                    return report

                meta = f["meta"]
                num_episodes: int = int(meta.attrs.get("num_episodes", 0))
                report["num_episodes"] = num_episodes
                episode_lengths = meta["episode_lengths"][:] if "episode_lengths" in meta else []

                for i in range(num_episodes):
                    ep_key = f"episode_{i}"
                    if ep_key not in f["data"]:
                        report["errors"].append(f"Missing group: /data/{ep_key}")
                        report["valid"] = False
                        continue

                    ep_grp = f["data"][ep_key]

                    # Check required datasets
                    for ds_name in ["observation.images.top", "observation.state", "action"]:
                        if ds_name not in ep_grp:
                            report["errors"].append(f"/data/{ep_key}/{ds_name} missing.")
                            report["valid"] = False

                    if "action" in ep_grp:
                        act = ep_grp["action"][:]
                        if np.any(np.isnan(act)):
                            report["errors"].append(f"/data/{ep_key}/action contains NaN.")
                            report["valid"] = False
                        if np.any(np.isinf(act)):
                            report["errors"].append(f"/data/{ep_key}/action contains Inf.")
                            report["valid"] = False

                    if "observation.images.top" in ep_grp:
                        img_ds = ep_grp["observation.images.top"]
                        if img_ds.dtype != np.uint8:
                            report["warnings"].append(
                                f"/data/{ep_key}/observation.images.top dtype is {img_ds.dtype}, expected uint8."
                            )

                    if i < len(episode_lengths):
                        declared_T = int(episode_lengths[i])
                        if "action" in ep_grp:
                            actual_T = ep_grp["action"].shape[0]
                            if actual_T != declared_T:
                                report["warnings"].append(
                                    f"/data/{ep_key}: declared length {declared_T} != action length {actual_T}."
                                )

        except Exception as exc:  # noqa: BLE001
            report["valid"] = False
            report["errors"].append(f"HDF5 read error: {exc}")

        return report

    def get_stats(self, dataset_path: Path | None = None) -> dict:
        """Return dataset statistics.

        Returns dict with keys:
        ``num_episodes``, ``total_frames``, ``task_distribution``,
        ``fps``, ``robot_model``, ``state_dim``, ``action_dim``.
        """
        path = Path(dataset_path) if dataset_path else self.output_path

        if not _H5PY_AVAILABLE:
            return {"error": "h5py not available"}

        if not path.exists():
            return {"error": f"File not found: {path}"}

        stats: dict = {}
        try:
            with h5py.File(path, "r") as f:
                meta = f["meta"]
                stats["num_episodes"] = int(meta.attrs.get("num_episodes", 0))
                stats["total_frames"] = int(meta.attrs.get("total_frames", 0))
                stats["fps"] = float(meta.attrs.get("fps", 0.0))
                stats["robot_model"] = str(meta.attrs.get("robot_model", ""))
                stats["state_dim"] = int(meta.attrs.get("state_dim", 0))
                stats["action_dim"] = int(meta.attrs.get("action_dim", 0))

                # Task distribution
                task_dist: dict[str, int] = {}
                if "task_types" in meta:
                    for tt in meta["task_types"]:
                        tt_str = tt.decode() if isinstance(tt, bytes) else str(tt)
                        task_dist[tt_str] = task_dist.get(tt_str, 0) + 1
                stats["task_distribution"] = task_dist

        except Exception as exc:  # noqa: BLE001
            stats["error"] = str(exc)

        return stats

    @staticmethod
    def load_episode(dataset_path: Path, episode_idx: int) -> dict:
        """Load a single episode from HDF5 for inspection.

        Returns dict with keys:
        ``images``, ``state``, ``action``, ``language_instruction``,
        ``task_type``, ``fps``, ``duration``.
        """
        if not _H5PY_AVAILABLE:
            raise ImportError("h5py is required to load episodes.")

        path = Path(dataset_path)
        with h5py.File(path, "r") as f:
            ep_key = f"episode_{episode_idx}"
            if ep_key not in f.get("data", {}):
                raise KeyError(f"Episode {episode_idx} not found in {path}")

            grp = f["data"][ep_key]
            result: dict = {}

            if "observation.images.top" in grp:
                result["images"] = grp["observation.images.top"][:]
            if "observation.state" in grp:
                result["state"] = grp["observation.state"][:]
            if "action" in grp:
                result["action"] = grp["action"][:]

            for attr_key in ["language_instruction", "task_type", "fps", "duration", "episode_id"]:
                if attr_key in grp.attrs:
                    val = grp.attrs[attr_key]
                    result[attr_key] = val.decode() if isinstance(val, bytes) else val

        return result

    # ------------------------------------------------------------------
    # JSON fallback (no h5py)
    # ------------------------------------------------------------------

    def _build_json_fallback(
        self,
        episodes: list["Episode"],
        actions_per_episode: list[np.ndarray],
        progress_callback: Callable[[int, int], None] | None,
    ) -> Path:
        """Write a JSON manifest when h5py is unavailable."""
        manifest_path = self.output_path.with_suffix(".json")
        manifest: dict = {
            "robot_model": _ROBOT_MODEL,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_episodes": len(episodes),
            "episodes": [],
        }

        for i, (ep, actions) in enumerate(zip(episodes, actions_per_episode)):
            manifest["episodes"].append({
                "episode_id": ep.episode_id,
                "video_path": ep.video_path,
                "num_frames": len(ep.frames),
                "fps": ep.fps,
                "task_type": ep.metadata.get("task_type", "generic_cleaning"),
                "language_instruction": ep.metadata.get("language_instruction", ""),
                "action_shape": list(actions.shape),
            })
            if progress_callback is not None:
                progress_callback(i + 1, len(episodes))

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as mf:
            json.dump(manifest, mf, indent=2)

        logger.info("h5py unavailable — wrote JSON manifest to %s.", manifest_path)
        return manifest_path
