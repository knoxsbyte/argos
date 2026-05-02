"""Tests for training/ pipeline."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from argos.training.ingest import VideoIngestor, Episode, VideoFrame
from argos.training.preprocess import PoseEstimator, ActionLabeler
from argos.training.dataset import LeRobotDatasetBuilder
from argos.training.finetune import LoRAFinetuner, FinetuneConfig
from argos.training.evaluate import PolicyEvaluator, EvalConfig
from argos.training.sim.mujoco_env import CleaningEnv


def _make_episode(n_frames: int = 30, task_type: str = "sweep_floor") -> Episode:
    frames = [
        VideoFrame(
            frame_idx=i,
            timestamp=i / 15.0,
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            depth=None,
        )
        for i in range(n_frames)
    ]
    return Episode(
        episode_id="ep-0",
        video_path="/fake/video.mp4",
        frames=frames,
        fps=15.0,
        duration=n_frames / 15.0,
        metadata={"task_type": task_type, "language_instruction": "sweep the floor"},
    )


# ── VideoIngestor ─────────────────────────────────────────────────────────────

def test_ingestor_validate_episode():
    ingestor = VideoIngestor()
    ep = _make_episode(30)
    ok, msg = ingestor.validate_episode(ep)
    assert ok


def test_ingestor_infer_task_type():
    ingestor = VideoIngestor()
    from pathlib import Path
    assert ingestor._infer_task_type(Path("sweep_demo_01.mp4")) == "sweep_floor"
    assert ingestor._infer_task_type(Path("bed_making_room.mp4")) == "make_bed"


def test_ingestor_mock_directory(tmp_path):
    # In mock mode (no cv2), ingest_directory returns empty list gracefully
    ingestor = VideoIngestor()
    # Should not raise even with no video files
    episodes = ingestor.ingest_directory(tmp_path)
    assert isinstance(episodes, list)


# ── PoseEstimator ─────────────────────────────────────────────────────────────

def test_pose_estimator_returns_dict():
    estimator = PoseEstimator()
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = estimator.estimate(frame)
    assert isinstance(result, dict)
    assert "wrist_velocity_left" in result
    assert "wrist_velocity_right" in result


def test_pose_estimator_sequence():
    estimator = PoseEstimator()
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 10
    results = estimator.estimate_sequence(frames)
    assert len(results) == 10


# ── ActionLabeler ─────────────────────────────────────────────────────────────

def test_action_labeler_produces_actions():
    labeler = ActionLabeler()
    ep = _make_episode(30)
    poses = [{"wrist_velocity_left": np.zeros(3), "wrist_velocity_right": np.zeros(3),
               "left_hand": None, "right_hand": None, "pose": None}] * 30
    actions = labeler.label_segment(ep.frames, poses, "sweep_floor")
    assert actions.shape == (30, 29)
    assert actions.dtype == np.float32


# ── LeRobotDatasetBuilder ──────────────────────────────────────────────────────

def test_dataset_builder_get_stats(tmp_path):
    builder = LeRobotDatasetBuilder(tmp_path / "test.h5")
    ep = _make_episode(20)
    actions = np.zeros((20, 29), dtype=np.float32)
    path = builder.build([ep], [actions])
    assert path is not None

    stats = builder.get_stats(path)
    assert "num_episodes" in stats or stats is not None


def test_dataset_builder_validate(tmp_path):
    builder = LeRobotDatasetBuilder(tmp_path / "val.h5")
    ep = _make_episode(20)
    actions = np.zeros((20, 29), dtype=np.float32)
    path = builder.build([ep], [actions])
    report = builder.validate(path)
    assert isinstance(report, dict)


# ── LoRAFinetuner (mock mode) ─────────────────────────────────────────────────

def test_finetune_estimate_time(tmp_path):
    cfg = FinetuneConfig(num_epochs=2, batch_size=2)
    finetuner = LoRAFinetuner(cfg, tmp_path / "out")
    ep = _make_episode(40)
    actions = np.zeros((40, 29), dtype=np.float32)
    builder = LeRobotDatasetBuilder(tmp_path / "ds.h5")
    ds_path = builder.build([ep, ep], [actions, actions])

    estimate = finetuner.estimate_training_time(ds_path)
    assert isinstance(estimate, dict)
    assert "total_samples" in estimate or "estimated_hours" in estimate


def test_finetune_mock_train(tmp_path):
    import threading
    cfg = FinetuneConfig(num_epochs=1, batch_size=2)
    finetuner = LoRAFinetuner(cfg, tmp_path / "out2")

    ep = _make_episode(20)
    actions = np.zeros((20, 29), dtype=np.float32)
    builder = LeRobotDatasetBuilder(tmp_path / "ds2.h5")
    ds_path = builder.build([ep], [actions])

    stop_event = threading.Event()
    losses = []
    result = finetuner.train(
        ds_path,
        progress_callback=lambda epoch, step, loss, metrics: losses.append(loss),
        stop_event=stop_event,
    )
    assert result is not None   # path to checkpoint


# ── CleaningEnv (MuJoCo mock mode) ────────────────────────────────────────────

def test_cleaning_env_reset():
    env = CleaningEnv(task_type="sweep_floor", room_layout="simple")
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "robot_state" in obs


def test_cleaning_env_step():
    env = CleaningEnv(task_type="sweep_floor", room_layout="simple")
    env.reset()
    action = np.zeros(29)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)


def test_cleaning_env_multiple_layouts():
    for layout in ["simple", "bedroom", "kitchen"]:
        env = CleaningEnv(task_type="sweep_floor", room_layout=layout)
        obs, _ = env.reset()
        assert obs is not None
        env.close()


# ── PolicyEvaluator ───────────────────────────────────────────────────────────

def test_evaluator_runs(tmp_path):
    from argos.policy.base import MockPolicy, PolicyConfig
    cfg = PolicyConfig(model_name="mock")
    policy = MockPolicy(cfg)
    policy.load()

    env = CleaningEnv(task_type="sweep_floor")
    eval_cfg = EvalConfig(num_episodes=2, max_steps_per_episode=20,
                           task_types=["sweep_floor"])
    evaluator = PolicyEvaluator(policy, env, eval_cfg)
    results = evaluator.evaluate()
    assert len(results) >= 1
    assert 0.0 <= results[0].success_rate <= 1.0

    report = evaluator.generate_report(results)
    assert isinstance(report, str)
    assert len(report) > 0
