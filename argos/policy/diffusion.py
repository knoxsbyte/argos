"""
argos.policy.diffusion — Diffusion Policy inference wrapper.

Implements DDPM/DDIM denoising for visuomotor action generation.
Architecture: CNN image encoder feeding a UNet-based noise prediction network.
Uses an observation sliding window and action chunk buffering for real-time
control at policy inference rates.

Falls back to MockPolicy behaviour if torch is unavailable or loading fails.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

import numpy as np

from argos.comm import Action
from argos.policy.base import (
    BasePolicy,
    MockPolicy,
    PolicyConfig,
    PolicyObservation,
    PolicyOutput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ML imports
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not available — DiffusionPolicy will run in mock mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_H = 96
_IMAGE_W = 96
_IMAGE_C = 3
_STATE_DIM = 29        # G1 joint positions (29 DOF)
_ACTION_DIM = 29       # joint position targets
_DDIM_STEPS = 10       # fast inference; full DDPM uses 100
_DDPM_STEPS = 100      # scheduler total steps
_BETA_START = 1e-4
_BETA_END = 0.02


# ---------------------------------------------------------------------------
# Lightweight UNet noise predictor (used when no checkpoint is provided)
# ---------------------------------------------------------------------------

def _build_unet(obs_dim: int, action_dim: int, pred_horizon: int) -> Any:
    """Build a minimal 1-D UNet for noise prediction over action sequences."""
    if not _TORCH_AVAILABLE:
        return None

    class SinusoidalEmbedding(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, t: "torch.Tensor") -> "torch.Tensor":
            half = self.dim // 2
            freqs = torch.exp(
                -torch.arange(half, dtype=torch.float32, device=t.device)
                * (np.log(10_000) / (half - 1))
            )
            args = t[:, None].float() * freqs[None]
            return torch.cat([args.sin(), args.cos()], dim=-1)

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.Mish(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class NoisePredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            cond_dim = obs_dim
            self.time_emb = SinusoidalEmbedding(128)
            self.time_proj = nn.Linear(128, cond_dim)
            self.down1 = ConvBlock(action_dim + cond_dim, 256)
            self.down2 = ConvBlock(256, 512)
            self.mid = ConvBlock(512, 512)
            self.up2 = ConvBlock(512 + 256, 256)
            self.up1 = ConvBlock(256 + action_dim + cond_dim, action_dim)

        def forward(
            self,
            x: "torch.Tensor",
            t: "torch.Tensor",
            cond: "torch.Tensor",
        ) -> "torch.Tensor":
            # x: (B, action_dim, pred_horizon)
            # cond: (B, obs_dim) -> broadcast across time
            cond_t = self.time_proj(self.time_emb(t))  # (B, obs_dim)
            c = (cond + cond_t).unsqueeze(-1).expand(-1, -1, x.shape[-1])
            h0 = torch.cat([x, c], dim=1)  # (B, action_dim+obs_dim, T)
            d1 = self.down1(h0)
            d2 = self.down2(d1)
            m = self.mid(d2)
            u2 = self.up2(torch.cat([m, d2], dim=1))
            u1 = self.up1(torch.cat([u2, d1, h0], dim=1))
            return u1

    return NoisePredictor()


def _build_cnn_encoder(obs_horizon: int) -> Any:
    """Build a lightweight CNN image encoder."""
    if not _TORCH_AVAILABLE:
        return None

    import torch.nn as nn  # type: ignore[import]

    class CNNEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(obs_horizon * _IMAGE_C, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            # Compute output size with a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, obs_horizon * _IMAGE_C, _IMAGE_H, _IMAGE_W)
                out_size = self.net(dummy).shape[1]
            self.proj = nn.Linear(out_size, 512)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.proj(self.net(x))

    return CNNEncoder()


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


class DiffusionPolicy(BasePolicy):
    """Visuomotor control via DDPM/DDIM denoising over action sequences.

    Maintains a sliding window of observations and a buffer of predicted
    actions. When the action buffer empties, runs a full DDIM denoising
    pass to generate the next chunk.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(config)
        self._obs_horizon: int = 2
        self._action_horizon: int = 8
        self._pred_horizon: int = 16
        self._obs_queue: collections.deque[PolicyObservation] = collections.deque(
            maxlen=self._obs_horizon
        )
        self._pending_actions: list[np.ndarray] = []

        self._mock: MockPolicy | None = None
        self._encoder: Any = None
        self._unet: Any = None
        self._device: str = config.device

        # DDIM schedule (computed on load)
        self._alphas_cumprod: np.ndarray | None = None
        self._ddim_timesteps: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load diffusion model weights and build DDIM schedule.

        Tries to restore weights from config.checkpoint_path; falls back
        to a freshly initialised (random-weight) network when unavailable.
        Sets mock_mode if torch is not importable.
        """
        if not _TORCH_AVAILABLE:
            logger.warning(
                "DiffusionPolicy: torch unavailable. Enabling mock mode."
            )
            self._activate_mock()
            return

        try:
            self._load_model()
            self._build_ddim_schedule()
            self.is_loaded = True
            logger.info(
                "DiffusionPolicy: loaded on device '%s' "
                "(obs_horizon=%d, pred_horizon=%d, ddim_steps=%d).",
                self._device,
                self._obs_horizon,
                self._pred_horizon,
                _DDIM_STEPS,
            )
        except Exception as exc:
            logger.warning(
                "DiffusionPolicy: loading failed (%s). Enabling mock mode.", exc
            )
            self._activate_mock()

    def _load_model(self) -> None:
        """Internal loader — may raise; handled by load()."""
        import torch  # re-import in guarded scope

        obs_dim = 512 + _STATE_DIM  # CNN features + joint state

        self._encoder = _build_cnn_encoder(self._obs_horizon)
        self._unet = _build_unet(obs_dim, _ACTION_DIM, self._pred_horizon)

        if self.config.checkpoint_path is not None:
            ckpt = torch.load(
                self.config.checkpoint_path, map_location=self._device
            )
            state_dict = ckpt.get("state_dict", ckpt)
            # Separate encoder and unet keys if checkpoint holds both
            enc_sd = {k[len("encoder."):]: v for k, v in state_dict.items()
                      if k.startswith("encoder.")}
            unet_sd = {k[len("unet."):]: v for k, v in state_dict.items()
                       if k.startswith("unet.")}
            if enc_sd:
                self._encoder.load_state_dict(enc_sd, strict=False)
            if unet_sd:
                self._unet.load_state_dict(unet_sd, strict=False)
            logger.info("DiffusionPolicy: loaded weights from '%s'.",
                        self.config.checkpoint_path)

        self._encoder = self._encoder.to(self._device).eval()
        self._unet = self._unet.to(self._device).eval()

    def _build_ddim_schedule(self) -> None:
        """Pre-compute DDPM beta schedule and DDIM timestep subset."""
        betas = np.linspace(_BETA_START, _BETA_END, _DDPM_STEPS, dtype=np.float64)
        alphas = 1.0 - betas
        self._alphas_cumprod = np.cumprod(alphas)

        # Evenly spaced DDIM steps
        step_size = _DDPM_STEPS // _DDIM_STEPS
        self._ddim_timesteps = np.arange(0, _DDPM_STEPS, step_size)[::-1].copy()

    def _activate_mock(self) -> None:
        self._mock = MockPolicy(self.config)
        self._mock.load()
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, obs: PolicyObservation) -> PolicyOutput:
        """Return the next buffered action or run a new denoising pass.

        1. Add obs to the sliding window queue.
        2. If pending_actions is non-empty: pop and return the next action.
        3. Otherwise: encode observations and run DDIM denoising to fill buffer.
        """
        if self._mock is not None:
            return self._mock.predict(obs)

        self._obs_queue.append(obs)

        if self._pending_actions:
            raw_action = self._pending_actions.pop(0)
            return self._wrap_action(raw_action, done=False)

        try:
            obs_encoding = self._encode_observations(list(self._obs_queue))
            action_chunk = self._run_denoising(obs_encoding)
            # Buffer action_horizon actions; execute only that window
            actions = [action_chunk[i] for i in range(self._action_horizon)]
            self._pending_actions = actions[1:]   # queue remainder
            return self._wrap_action(actions[0], done=False)
        except Exception as exc:
            logger.warning("DiffusionPolicy: denoising failed (%s); returning zero action.", exc)
            return _zero_output()

    def _wrap_action(self, raw: np.ndarray, done: bool) -> PolicyOutput:
        """Pack a raw (action_dim,) numpy array into a PolicyOutput."""
        targets = raw.tolist()
        action = Action(
            joint_targets=targets,
            gripper_left=0.0,
            gripper_right=0.0,
            duration_ms=int(1_000.0 / self.config.inference_freq),
        ).clipped()
        return PolicyOutput(
            action=action,
            confidence=0.75,
            done_signal=done,
            metadata={"policy": "diffusion", "buffered": len(self._pending_actions)},
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _encode_observations(self, obs_list: list[PolicyObservation]) -> np.ndarray:
        """CNN-encode stacked image observations and concatenate robot state.

        Returns a flat float32 array of shape (obs_dim,) = (512 + STATE_DIM,).
        Pads with copies of the first observation when the queue is not full.
        """
        import torch  # re-import in guarded scope

        # Pad to obs_horizon if queue is still filling up
        while len(obs_list) < self._obs_horizon:
            obs_list = [obs_list[0]] + obs_list

        # Resize and stack images: (obs_horizon, H, W, C) → (1, obs_horizon*C, H, W)
        frames = []
        for o in obs_list[-self._obs_horizon:]:
            img = _resize_image(o.image, _IMAGE_H, _IMAGE_W)  # (H, W, C)
            frames.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        stacked = np.concatenate(frames, axis=0)  # (obs_horizon*C, H, W)
        img_tensor = torch.from_numpy(stacked).unsqueeze(0).to(self._device)

        with torch.no_grad():
            img_feat = self._encoder(img_tensor)  # (1, 512)

        img_feat_np = img_feat.squeeze(0).cpu().numpy()

        # Use the most recent robot state
        state_np = np.array(
            obs_list[-1].robot_state.joint_positions, dtype=np.float32
        )

        return np.concatenate([img_feat_np, state_np], axis=0)

    # ------------------------------------------------------------------
    # DDIM denoising
    # ------------------------------------------------------------------

    def _run_denoising(self, obs_encoding: np.ndarray) -> np.ndarray:
        """DDIM denoising loop.

        Returns an array of shape (pred_horizon, action_dim) representing the
        denoised action sequence.
        """
        import torch  # re-import in guarded scope

        B = 1
        T = self._pred_horizon
        A = _ACTION_DIM

        cond = torch.from_numpy(obs_encoding).float().unsqueeze(0).to(self._device)

        # Sample initial Gaussian noise
        x = torch.randn(B, A, T, device=self._device)

        alphas_cumprod = self._alphas_cumprod
        assert alphas_cumprod is not None

        with torch.no_grad():
            for step_idx in range(len(self._ddim_timesteps)):
                t_val = int(self._ddim_timesteps[step_idx])
                t_tensor = torch.tensor([t_val], device=self._device, dtype=torch.long)

                # Predict noise
                eps_pred = self._unet(x, t_tensor, cond)  # (B, A, T)

                alpha_t = float(alphas_cumprod[t_val])
                alpha_t_tensor = torch.tensor(alpha_t, device=self._device)

                if step_idx + 1 < len(self._ddim_timesteps):
                    t_prev = int(self._ddim_timesteps[step_idx + 1])
                    alpha_prev = float(alphas_cumprod[t_prev])
                else:
                    alpha_prev = 1.0

                alpha_prev_tensor = torch.tensor(alpha_prev, device=self._device)

                # DDIM update step (deterministic, eta=0)
                x0_pred = (x - (1 - alpha_t_tensor).sqrt() * eps_pred) / alpha_t_tensor.sqrt()
                x0_pred = x0_pred.clamp(-1.0, 1.0)
                x = alpha_prev_tensor.sqrt() * x0_pred + (1 - alpha_prev_tensor).sqrt() * eps_pred

        # x: (1, action_dim, pred_horizon) → (pred_horizon, action_dim)
        actions = x.squeeze(0).permute(1, 0).cpu().numpy()
        return actions.astype(np.float32)

    # ------------------------------------------------------------------
    # Reset / unload
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear sliding window and action buffer between episodes."""
        self._obs_queue.clear()
        self._pending_actions.clear()
        if self._mock is not None:
            self._mock.reset()
        logger.debug("DiffusionPolicy: reset.")

    def unload(self) -> None:
        """Release encoder and UNet from GPU memory."""
        if self._mock is not None:
            self._mock = None
            self.is_loaded = False
            return

        try:
            import torch  # type: ignore[import]
            del self._encoder
            del self._unet
            self._encoder = None
            self._unet = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("DiffusionPolicy: unloaded and freed GPU memory.")
        except Exception as exc:
            logger.warning("DiffusionPolicy: error during unload: %s", exc)
        self.is_loaded = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a HxWx3 uint8 image to (h, w, 3) using nearest-neighbour."""
    src_h, src_w = img.shape[:2]
    if src_h == h and src_w == w:
        return img
    # Use cv2 if available for quality; fall back to numpy slicing
    try:
        import cv2  # type: ignore[import]
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        row_idx = (np.arange(h) * src_h / h).astype(int)
        col_idx = (np.arange(w) * src_w / w).astype(int)
        return img[np.ix_(row_idx, col_idx)]


def _zero_output() -> PolicyOutput:
    """Return a safe zero-action output for error recovery paths."""
    return PolicyOutput(
        action=Action(
            joint_targets=[0.0] * 29,
            gripper_left=0.0,
            gripper_right=0.0,
            duration_ms=100,
        ),
        confidence=0.0,
        done_signal=False,
        metadata={"error": "inference_failed"},
    )
