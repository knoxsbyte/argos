"""
argos.policy.act — ACT (Action Chunking with Transformers) inference wrapper.

Predicts action_chunk_size future joint positions simultaneously via a
transformer encoder-decoder. Uses temporal ensemble (exponential-decay
weighted average) across overlapping chunk predictions to reduce jitter
at chunk boundaries.

Falls back to MockPolicy behaviour if torch is unavailable or loading fails.
"""

from __future__ import annotations

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
    logger.warning("torch not available — ACTPolicy will run in mock mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_H = 224
_IMAGE_W = 224
_IMAGE_C = 3
_RESNET_OUT_DIM = 512    # ResNet-18 final feature dimension
_STATE_DIM = 29          # G1 DOF
_ACTION_DIM = 29         # joint position targets
_TRANSFORMER_DIM = 256
_TRANSFORMER_HEADS = 8
_TRANSFORMER_ENC_LAYERS = 4
_TRANSFORMER_DEC_LAYERS = 7


# ---------------------------------------------------------------------------
# Minimal ACT network
# ---------------------------------------------------------------------------

def _build_act_network(chunk_size: int) -> Any:
    """Build a lightweight ACT transformer (encoder + decoder).

    The architecture follows the original ACT paper:
    - ResNet-18 backbone for image encoding
    - Linear projection for robot state
    - BERT-style encoder over the concatenated tokens
    - GPT-style decoder that autoregressively generates the action chunk
    """
    if not _TORCH_AVAILABLE:
        return None

    import torch  # re-import in guarded scope
    import torch.nn as nn  # type: ignore[import]

    class ResNet18Backbone(nn.Module):
        """Minimal ResNet-18-like CNN producing (B, _RESNET_OUT_DIM) features."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(_IMAGE_C, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, _RESNET_OUT_DIM, 2, stride=2)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        @staticmethod
        def _make_layer(
            in_ch: int, out_ch: int, blocks: int, stride: int = 1
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
            for _ in range(1, blocks):
                layers += [
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.pool(x)
            return x.flatten(1)  # (B, 512)

    class ACTNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d = _TRANSFORMER_DIM

            # Encoders
            self.image_backbone = ResNet18Backbone()
            self.image_proj = nn.Linear(_RESNET_OUT_DIM, d)
            self.state_proj = nn.Linear(_STATE_DIM, d)

            # Transformer encoder (BERT-style, bidirectional)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=_TRANSFORMER_HEADS,
                dim_feedforward=d * 4, dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=_TRANSFORMER_ENC_LAYERS
            )

            # Transformer decoder (GPT-style, causal)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d, nhead=_TRANSFORMER_HEADS,
                dim_feedforward=d * 4, dropout=0.1, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                dec_layer, num_layers=_TRANSFORMER_DEC_LAYERS
            )

            # Learned query embeddings for each action in the chunk
            self.query_embed = nn.Embedding(chunk_size, d)

            # Output head: project from d to action_dim
            self.action_head = nn.Linear(d, _ACTION_DIM)

        def forward(
            self,
            image: "torch.Tensor",       # (B, C, H, W)
            state: "torch.Tensor",       # (B, STATE_DIM)
        ) -> "torch.Tensor":             # (B, chunk_size, ACTION_DIM)
            B = image.shape[0]

            img_feat = self.image_proj(self.image_backbone(image))  # (B, d)
            state_feat = self.state_proj(state)                      # (B, d)

            # Memory: stack image + state as encoder tokens
            memory_in = torch.stack([img_feat, state_feat], dim=1)  # (B, 2, d)
            memory = self.encoder(memory_in)                          # (B, 2, d)

            # Queries: one per chunk timestep
            idx = torch.arange(self.query_embed.num_embeddings, device=image.device)
            queries = self.query_embed(idx).unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

            # Decode
            out = self.decoder(queries, memory)  # (B, K, d)
            return self.action_head(out)          # (B, K, ACTION_DIM)

    return ACTNetwork()


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


class ACTPolicy(BasePolicy):
    """Action Chunking with Transformers for dexterous bimanual manipulation.

    Predicts a chunk of config.action_chunk_size future joint positions in one
    forward pass. Overlapping chunks are combined via exponential-decay temporal
    ensemble to suppress boundary jitter.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(config)
        K = config.action_chunk_size
        self._chunk_buffer: list[np.ndarray] = []  # queued action chunks (overlapping)
        self._chunk_idx: int = 0                   # position within current chunk

        # Temporal ensemble weights: higher weight to more recent predictions
        weights = np.exp(-0.1 * np.arange(K, dtype=np.float64))
        self._ensemble_weights: np.ndarray = (weights / weights.sum()).astype(np.float32)

        self._mock: MockPolicy | None = None
        self._network: Any = None
        self._device: str = config.device

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ACT model weights.

        Restores from config.checkpoint_path if provided; otherwise starts
        with random weights (useful for debugging the control loop).
        Falls back to mock mode if torch is unavailable or loading fails.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("ACTPolicy: torch unavailable. Enabling mock mode.")
            self._activate_mock()
            return

        try:
            self._load_model()
            self.is_loaded = True
            logger.info(
                "ACTPolicy: loaded on device '%s' (chunk_size=%d).",
                self._device,
                self.config.action_chunk_size,
            )
        except Exception as exc:
            logger.warning(
                "ACTPolicy: loading failed (%s). Enabling mock mode.", exc
            )
            self._activate_mock()

    def _load_model(self) -> None:
        import torch  # re-import in guarded scope

        self._network = _build_act_network(self.config.action_chunk_size)

        if self.config.checkpoint_path is not None:
            state_dict = torch.load(
                self.config.checkpoint_path, map_location=self._device
            )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._network.load_state_dict(state_dict, strict=False)
            logger.info("ACTPolicy: loaded weights from '%s'.", self.config.checkpoint_path)

        self._network = self._network.to(self._device).eval()

    def _activate_mock(self) -> None:
        self._mock = MockPolicy(self.config)
        self._mock.load()
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, obs: PolicyObservation) -> PolicyOutput:
        """Run ACT inference with temporal ensemble.

        1. Encode image (ResNet-18 backbone).
        2. Encode robot state (linear projection).
        3. Forward through transformer → action_chunk (K, action_dim).
        4. Apply temporal ensemble across overlapping chunk predictions.
        5. Return the action for the current timestep.
        """
        if self._mock is not None:
            return self._mock.predict(obs)

        try:
            return self._run_inference(obs)
        except Exception as exc:
            logger.warning("ACTPolicy: inference failed (%s); returning zero action.", exc)
            return _zero_output()

    def _run_inference(self, obs: PolicyObservation) -> PolicyOutput:
        import torch  # re-import in guarded scope

        # Prepare image tensor: (1, C, H, W) float32 in [0, 1]
        img = _resize_image(obs.image, _IMAGE_H, _IMAGE_W)
        img_tensor = (
            torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(self._device)
        )

        # Prepare state tensor: (1, STATE_DIM)
        state_np = np.array(obs.robot_state.joint_positions, dtype=np.float32)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self._device)

        with torch.no_grad():
            chunk = self._network(img_tensor, state_tensor)  # (1, K, ACTION_DIM)

        chunk_np = chunk.squeeze(0).cpu().numpy()  # (K, ACTION_DIM)

        # Add to buffer and compute ensemble action for timestep 0
        self._chunk_buffer.append(chunk_np)
        ensembled = self._temporal_ensemble(chunk_np)

        # Advance chunk index
        self._chunk_idx += 1

        action = Action(
            joint_targets=ensembled.tolist(),
            gripper_left=0.0,
            gripper_right=0.0,
            duration_ms=int(1_000.0 / self.config.inference_freq),
        ).clipped()

        return PolicyOutput(
            action=action,
            confidence=0.85,
            done_signal=False,
            metadata={
                "policy": "act",
                "chunk_idx": self._chunk_idx,
                "num_chunks": len(self._chunk_buffer),
            },
        )

    # ------------------------------------------------------------------
    # Temporal ensemble
    # ------------------------------------------------------------------

    def _temporal_ensemble(self, new_chunk: np.ndarray) -> np.ndarray:
        """Weighted average of the current timestep action across all overlapping chunks.

        Each chunk in the buffer predicts an action for the current timestep
        at a different offset into that chunk. The weights are exponential
        (higher weight → more recent prediction) to reduce boundary jitter.

        Args:
            new_chunk: The freshly predicted chunk, shape (K, action_dim).

        Returns:
            Weighted-average action for this timestep, shape (action_dim,).
        """
        K = self.config.action_chunk_size
        buf = self._chunk_buffer

        # Trim buffer to at most K overlapping chunks
        if len(buf) > K:
            self._chunk_buffer = buf[-K:]
            buf = self._chunk_buffer

        N = len(buf)  # number of overlapping predictions
        # For chunk i (from the end), the current timestep corresponds to
        # position (N - 1 - i) within that chunk.
        weighted_sum = np.zeros(_ACTION_DIM, dtype=np.float64)
        weight_total = 0.0

        for i, chunk in enumerate(reversed(buf)):
            offset = N - 1 - i  # index into this chunk for current timestep
            if offset >= K:
                continue
            w = float(self._ensemble_weights[offset])
            weighted_sum += chunk[offset].astype(np.float64) * w
            weight_total += w

        if weight_total < 1e-8:
            return new_chunk[0].astype(np.float32)

        return (weighted_sum / weight_total).astype(np.float32)

    # ------------------------------------------------------------------
    # Reset / unload
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear chunk buffer and reset index for a new episode."""
        self._chunk_buffer.clear()
        self._chunk_idx = 0
        if self._mock is not None:
            self._mock.reset()
        logger.debug("ACTPolicy: reset.")

    def unload(self) -> None:
        """Release network from GPU memory."""
        if self._mock is not None:
            self._mock = None
            self.is_loaded = False
            return

        try:
            import torch  # type: ignore[import]
            del self._network
            self._network = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ACTPolicy: unloaded and freed GPU memory.")
        except Exception as exc:
            logger.warning("ACTPolicy: error during unload: %s", exc)
        self.is_loaded = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a HxWx3 uint8 image to (h, w, 3)."""
    src_h, src_w = img.shape[:2]
    if src_h == h and src_w == w:
        return img
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
