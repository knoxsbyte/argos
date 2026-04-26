"""
argos.policy.vla — OpenVLA (Vision-Language-Action) inference wrapper.

Wraps openvla/openvla-7b from HuggingFace for language-conditioned robot
manipulation. Supports 4-bit quantization via bitsandbytes for low-VRAM
deployments and LoRA fine-tune adapters via peft.

Falls back to MockPolicy behaviour if torch/transformers are unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from argos.comm import Action, RobotState
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
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not available — OpenVLAPolicy will run in mock mode.")

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available — OpenVLAPolicy will run in mock mode.")

try:
    from peft import PeftModel  # type: ignore[import]
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    from PIL import Image as PILImage  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENVLA_HF_ID = "openvla/openvla-7b"

# OpenVLA uses a 256-bin discretised action space over 7 DOF
# (EEF xyz, rpy, gripper). Bins span [-1, 1] linearly.
_ACTION_DIM = 7
_NUM_BINS = 256
_BIN_CENTERS = np.linspace(-1.0, 1.0, _NUM_BINS, dtype=np.float32)

# Map from 7-DOF compact action (left-arm EEF + gripper) to G1 joint indices.
# Left arm joints: 15-20 (6 joints), gripper signal maps to joint 27.
_G1_LEFT_ARM_INDICES = [15, 16, 17, 18, 19, 20]
_G1_LEFT_GRIPPER_INDEX = 27

# G1 VRAM threshold (GB) below which 4-bit quantization is applied
_QUANTIZE_VRAM_THRESHOLD_GB = 16.0


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


class OpenVLAPolicy(BasePolicy):
    """Language-conditioned manipulation policy using OpenVLA-7B.

    Architecture: PrismaticVLM backbone (SigLIP vision + Llama-2 7B language).
    Action tokens are decoded from the LM head as discretised joint angles.

    When ML libraries are unavailable or loading fails, the policy degrades
    gracefully to MockPolicy behaviour and logs a clear warning.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(config)
        self._mock: MockPolicy | None = None
        self._model: Any = None
        self._processor: Any = None
        self._device: str = config.device

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load OpenVLA model weights.

        Sequence:
        1. Try checkpoint_path (fine-tuned LoRA).
        2. Fall back to openvla/openvla-7b from HuggingFace.
        3. Apply 4-bit quantization if VRAM < 16 GB.
        4. Attach LoRA adapter if checkpoint_path has adapter_config.json.

        On any failure sets mock_mode and continues without raising.
        """
        if not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE):
            logger.warning(
                "OpenVLAPolicy: torch/transformers unavailable. Enabling mock mode."
            )
            self._activate_mock()
            return

        try:
            self._load_model()
            self.is_loaded = True
            logger.info(
                "OpenVLAPolicy: loaded model '%s' on device '%s'.",
                self.config.model_name,
                self._device,
            )
        except Exception as exc:
            logger.warning(
                "OpenVLAPolicy: model loading failed (%s). Enabling mock mode.", exc
            )
            self._activate_mock()

    def _load_model(self) -> None:
        """Internal model loading — may raise; caller handles exceptions."""
        import torch  # re-import inside guarded scope

        model_id = self.config.checkpoint_path or _OPENVLA_HF_ID
        quantization_config = None

        # Determine whether to quantize based on available VRAM
        if self._device != "cpu" and torch.cuda.is_available():
            device_idx = 0
            if ":" in self._device:
                device_idx = int(self._device.split(":")[1])
            free_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
            if free_gb < _QUANTIZE_VRAM_THRESHOLD_GB:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore[import]
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info(
                        "OpenVLAPolicy: applying 4-bit quantization (%.1f GB VRAM).",
                        free_gb,
                    )
                except ImportError:
                    logger.warning(
                        "OpenVLAPolicy: bitsandbytes not installed; "
                        "skipping 4-bit quantization."
                    )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16 if self._device != "cpu" else torch.float32,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["device_map"] = self._device

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)

        # Apply LoRA adapter if the checkpoint directory contains adapter config
        if (
            _PEFT_AVAILABLE
            and self.config.checkpoint_path is not None
            and os.path.isfile(
                os.path.join(self.config.checkpoint_path, "adapter_config.json")
            )
        ):
            logger.info(
                "OpenVLAPolicy: loading LoRA adapter from '%s'.",
                self.config.checkpoint_path,
            )
            self._model = PeftModel.from_pretrained(
                self._model, self.config.checkpoint_path
            )
            self._model = self._model.merge_and_unload()

        if quantization_config is None and self._device != "cpu":
            self._model = self._model.to(self._device)

        self._model.eval()

    def _activate_mock(self) -> None:
        """Switch to MockPolicy behaviour without raising."""
        self._mock = MockPolicy(self.config)
        self._mock.load()
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, obs: PolicyObservation) -> PolicyOutput:
        """Run OpenVLA inference for one timestep.

        Pipeline:
        1. Preprocess image with AutoProcessor.
        2. Format language prompt in OpenVLA's expected template.
        3. Generate 7 action tokens (discretised 256-bin action space).
        4. Decode tokens to continuous joint angles.
        5. Map 7-DOF compact action to G1's 29-DOF joint space.
        """
        if self._mock is not None:
            return self._mock.predict(obs)

        try:
            return self._run_inference(obs)
        except Exception as exc:
            logger.warning("OpenVLAPolicy: inference failed (%s); returning zero action.", exc)
            return _zero_output()

    def _run_inference(self, obs: PolicyObservation) -> PolicyOutput:
        import torch  # re-import inside guarded scope

        # Build prompt
        prompt = (
            f"In: What action should the robot take to "
            f"{obs.language_instruction}?\nOut:"
        )

        # Preprocess image — AutoProcessor expects PIL or numpy HWC uint8
        if _PIL_AVAILABLE:
            from PIL import Image as PILImage  # type: ignore[import]
            pil_image = PILImage.fromarray(obs.image)
            inputs = self._processor(prompt, pil_image, return_tensors="pt")
        else:
            inputs = self._processor(prompt, obs.image, return_tensors="pt")

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=_ACTION_DIM,
                do_sample=False,
            )

        # The model generates one token per DOF; extract the new tokens only
        input_len = inputs["input_ids"].shape[1]
        action_token_ids = generated[0, input_len:].cpu().tolist()

        compact_action = self._decode_action_tokens(action_token_ids)
        joint_targets = self._map_to_g1_joints(compact_action, obs.robot_state)

        action = Action(
            joint_targets=joint_targets,
            gripper_left=float(np.clip(compact_action[6], 0.0, 1.0)),
            gripper_right=0.0,
            duration_ms=int(1_000.0 / self.config.inference_freq),
        ).clipped()

        return PolicyOutput(
            action=action,
            confidence=0.8,
            done_signal=False,
            metadata={"model": "openvla", "tokens": action_token_ids},
        )

    # ------------------------------------------------------------------
    # Token decoding helpers
    # ------------------------------------------------------------------

    def _decode_action_tokens(self, tokens: list[int]) -> np.ndarray:
        """Convert discretised token IDs back to continuous joint angles.

        OpenVLA reserves the top-256 vocabulary entries for action tokens.
        Token value - (vocab_size - 256) gives the bin index in [0, 255].
        """
        vocab_size = getattr(self._model.config, "vocab_size", 32_000)
        action_offset = vocab_size - _NUM_BINS

        actions = np.zeros(_ACTION_DIM, dtype=np.float32)
        for i, token_id in enumerate(tokens[:_ACTION_DIM]):
            bin_idx = int(token_id) - action_offset
            bin_idx = int(np.clip(bin_idx, 0, _NUM_BINS - 1))
            actions[i] = _BIN_CENTERS[bin_idx]

        return actions

    def _map_to_g1_joints(
        self, compact_action: np.ndarray, robot_state: RobotState
    ) -> list[float]:
        """Map 7-DOF arm action to G1's 29-DOF joint space.

        The 7 OpenVLA outputs correspond to left-arm EEF deltas:
          [dx, dy, dz, drx, dry, drz, gripper]

        We apply the first 6 values as position offsets on left-arm joints
        (indices 15-20) and hold all other joints at their current positions.
        Non-arm joints remain at the current robot state values.
        """
        joint_targets = list(robot_state.joint_positions)  # copy current state

        # Apply arm offsets (scaled to reasonable joint-space magnitudes)
        scale = 0.05  # radians per unit action
        for local_idx, g1_idx in enumerate(_G1_LEFT_ARM_INDICES):
            joint_targets[g1_idx] += float(compact_action[local_idx]) * scale

        return joint_targets

    # ------------------------------------------------------------------
    # Reset / unload
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """No episode-level state to reset for VLA (stateless per-call)."""
        if self._mock is not None:
            self._mock.reset()
        logger.debug("OpenVLAPolicy: reset.")

    def unload(self) -> None:
        """Release model from GPU memory."""
        if self._mock is not None:
            self._mock = None
            self.is_loaded = False
            return

        if self._model is not None:
            try:
                import torch  # type: ignore[import]
                del self._model
                self._model = None
                self._processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("OpenVLAPolicy: unloaded model and freed GPU memory.")
            except Exception as exc:
                logger.warning("OpenVLAPolicy: error during unload: %s", exc)
        self.is_loaded = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
