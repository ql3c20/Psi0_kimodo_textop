#!/usr/bin/env python3
"""HTTP bridge for a UNITREE_G1_SONIC GR00T N1.7 checkpoint."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr
import tyro
import uvicorn

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def _numpy_encode(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numpy_encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numpy_encode(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        arr = np.asarray(value)
        data = arr.data if arr.flags["C_CONTIGUOUS"] else arr.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(arr.dtype),
            "shape": arr.shape,
        }
    return value


def _numpy_decode(value: Any) -> Any:
    if isinstance(value, dict) and "__numpy__" in value:
        arr = np.frombuffer(
            b64decode(value["__numpy__"]), descr_to_dtype(value["dtype"])
        )
        return arr.reshape(value["shape"]) if value["shape"] else arr[0]
    if isinstance(value, dict):
        return {key: _numpy_decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_decode(item) for item in value]
    return value


def _btd(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, None]
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected state array with 1-3 dims, got {arr.shape}")


def _video_bthwc(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 3:
        return arr[None, None]
    if arr.ndim == 4:
        return arr[None]
    if arr.ndim == 5:
        return arr
    raise ValueError(f"Expected video array with 3-5 dims, got {arr.shape}")


@dataclass
class Config:
    model_path: Path = Path(
        "/pfs/pfs-ilWc5D/yzh/Isaac-GR00T/outputs/task1-gr00t-n1.7-finetune/checkpoint-80000"
    )
    backbone_path: Path = Path(
        "/pfs/pfs-ilWc5D/yzh/Psi0/huggingface/hub/"
        "models--nvidia--Cosmos-Reason2-2B/snapshots/"
        "9ce19a195e423419c349abfc86fd07178b230561"
    )
    host: str = "0.0.0.0"
    port: int = 22095
    device: str = "cuda"
    strict: bool = True
    # Prefix-RTC predicts Tp frames, executes Ta frames downstream, and uses
    # the remaining Tp-Ta normalized actions as the next denoising prefix.
    prefix_rtc: bool = False
    # None selects the mode stored in the checkpoint; old checkpoints without
    # the field resolve to legacy_zero.
    prefix_rtc_timestep_mode: str | None = None
    action_exec_horizon: int = 34


class Server:
    STATE_PARTS = (
        ("left_leg", 0, 6),
        ("right_leg", 6, 12),
        ("waist", 12, 15),
        ("left_arm", 15, 22),
        ("left_hand", 22, 29),
        ("right_arm", 29, 36),
        ("right_hand", 36, 43),
        ("projected_gravity", 43, 46),
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._relocate_backbone()
        self._allow_local_cosmos_backbone()
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.UNITREE_G1_SONIC,
            model_path=str(cfg.model_path),
            device=cfg.device,
            strict=cfg.strict,
        )
        self.prefix_rtc = bool(cfg.prefix_rtc)
        if self.prefix_rtc:
            import sys

            deploy_dir = str(Path(__file__).resolve().parent)
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from gr00t_n17_prefix_rtc import apply_prefix_rtc

            apply_prefix_rtc(
                self.policy,
                prefix_timestep_mode=cfg.prefix_rtc_timestep_mode,
            )

        self.action_chunk_size = len(
            self.policy.modality_configs["action"].delta_indices
        )
        self.action_exec_horizon = int(cfg.action_exec_horizon)
        self.rtc_overlap_steps = (
            self.action_chunk_size - self.action_exec_horizon
        )
        self.previous_normalized_action: np.ndarray | None = None
        if self.prefix_rtc and not (
            0 < self.rtc_overlap_steps < self.action_chunk_size
        ):
            raise ValueError(
                "Prefix-RTC requires 0 < prediction_horizon - "
                "action_exec_horizon < prediction_horizon; got "
                f"prediction_horizon={self.action_chunk_size}, "
                f"action_exec_horizon={self.action_exec_horizon}"
            )

        print(f"[gr00t-sonic-server] loaded {cfg.model_path}")
        print(
            "[gr00t-sonic-server] prediction horizon="
            f"{self.action_chunk_size}"
        )
        if self.prefix_rtc:
            print(
                "[gr00t-sonic-server] Prefix-RTC enabled: "
                f"execute={self.action_exec_horizon}, "
                f"prefix overlap={self.rtc_overlap_steps}, "
                "timestep_mode="
                f"{self.policy.model.action_head._prefix_rtc_timestep_mode}"
            )
        else:
            print("[gr00t-sonic-server] Prefix-RTC disabled")

    @staticmethod
    def _allow_local_cosmos_backbone() -> None:
        """Teach this process that a local Cosmos snapshot is Qwen3-based."""
        import gr00t.model.gr00t_n1d7.gr00t_n1d7 as n1d7

        original = n1d7.get_backbone_cls

        def get_backbone_cls(config):
            if "models--nvidia--Cosmos-Reason2-2B" in config.model_name:
                from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

                return Qwen3Backbone
            return original(config)

        n1d7.get_backbone_cls = get_backbone_cls

    def _relocate_backbone(self) -> None:
        """Repair absolute backbone paths embedded by training on another host."""
        if not self.cfg.backbone_path.is_dir():
            raise FileNotFoundError(
                f"GR00T backbone directory does not exist: {self.cfg.backbone_path}"
            )

        # A local path prevents Transformers from making model-info API calls
        # while constructing the tokenizer. The narrow process-local patch
        # above supplies the backbone identification that GR00T otherwise does
        # only by matching the canonical Hub model id.
        cache_home = self.cfg.backbone_path.parents[3]
        os.environ.setdefault("HF_HOME", str(cache_home))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        old_names = (
            "/mnt/data1/yzh/Isaac-GR00T/huggingface/hub/"
            "models--nvidia--Cosmos-Reason2-2B/snapshots/"
            "9ce19a195e423419c349abfc86fd07178b230561",
            "nvidia/Cosmos-Reason2-2B",
        )

        changed: list[Path] = []
        for path in self.cfg.model_path.rglob("*.json"):
            text = path.read_text()
            if not any(name in text for name in old_names):
                continue
            json.loads(text)
            repaired = text
            for old_name in old_names:
                repaired = repaired.replace(old_name, str(self.cfg.backbone_path))
            if repaired != text:
                # Parse once more before writing so a malformed replacement can
                # never damage checkpoint metadata.
                json.loads(repaired)
                path.write_text(repaired)
                changed.append(path)
        if changed:
            print(
                "[gr00t-server] Relocated Cosmos backbone in: "
                + ", ".join(str(path.relative_to(self.cfg.model_path)) for path in changed)
            )

    @staticmethod
    def _pick(action: dict[str, Any], key: str) -> np.ndarray:
        value = action.get(key, action.get(f"action.{key}"))
        if value is None:
            raise KeyError(f"GR00T response is missing action {key!r}")
        return _btd(value)

    def act(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            request = _numpy_decode(payload)
            state = _btd(request["state"]["sonic_state"])
            if state.shape[-1] != 46:
                raise ValueError(f"Expected SONIC state46, got {state.shape}")

            image_dict = request["image"]
            image = image_dict.get("ego_view", image_dict.get("rgb_head_stereo_left"))
            if image is None:
                raise KeyError("Missing ego_view/rgb_head_stereo_left image")

            observation: dict[str, Any] = {
                "video": {"ego_view": _video_bthwc(image)},
                "state": {},
                "language": {
                    "annotation.human.task_description": [[request["instruction"]]]
                },
            }
            for name, start, end in self.STATE_PARTS:
                observation["state"][name] = state[..., start:end]

            options: dict[str, Any] | None = None
            if self.prefix_rtc:
                history = request.get("history") or {}
                if isinstance(history, dict) and history.get("reset", False):
                    self.previous_normalized_action = None
                if self.previous_normalized_action is not None:
                    options = {
                        "rtc_prev_action": self.previous_normalized_action,
                        "action_horizon": int(
                            self.previous_normalized_action.shape[1]
                        ),
                        "rtc_overlap_steps": self.rtc_overlap_steps,
                    }
                    print(
                        "[gr00t-sonic-server] Prefix-RTC step: carry "
                        f"tail[{self.action_exec_horizon}:"
                        f"{self.action_chunk_size}]"
                    )
                else:
                    print(
                        "[gr00t-sonic-server] Prefix-RTC reset/first step: "
                        "unconditioned generation"
                    )

            action, info = self.policy.get_action(observation, options)
            if self.prefix_rtc:
                normalized_pred = info.get("normalized_action_pred")
                if normalized_pred is None:
                    raise RuntimeError(
                        "Prefix-RTC policy did not return normalized_action_pred"
                    )
                self.previous_normalized_action = np.asarray(
                    normalized_pred, dtype=np.float32
                )

            motion = self._pick(action, "motion_token")
            left_hand = self._pick(action, "left_hand_joints")
            right_hand = self._pick(action, "right_hand_joints")
            output = np.concatenate([motion, left_hand, right_hand], axis=-1)
            if output.shape[0] == 1:
                output = output[0]
            if output.ndim != 2 or output.shape[1] != 78:
                raise ValueError(
                    f"Expected GR00T SONIC output (T, 78), got {output.shape}"
                )

            return JSONResponse(
                content=_numpy_encode(
                    {
                        "action": output.astype(np.float32),
                        "err": 0.0,
                        "traj_image": np.zeros((1, 1, 3), dtype=np.uint8),
                    }
                )
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"status": repr(exc)})

    def run(self) -> None:
        app = FastAPI()
        app.post("/act")(self.act)
        app.get("/health")(lambda: {"status": "ok"})
        uvicorn.run(app, host=self.cfg.host, port=self.cfg.port)


if __name__ == "__main__":
    Server(tyro.cli(Config)).run()
