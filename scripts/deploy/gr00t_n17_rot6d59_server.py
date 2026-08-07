#!/usr/bin/env python3
"""HTTP bridge for GR00T N1.7 rot6d59 policy actions.

This server is a drop-in replacement for the Psi0 policy server used by the
existing SIMPLE Psi0 -> Kimodo -> TextOp tracker agent.  The client still sends
the same SIMPLE observation package:

    image["rgb_head_stereo_left"], state["states"] = hand14 + body29 + root6

The bridge converts root RPY to the rot6d layout used by the rot6d59 dataset,
queries the finetuned GR00T checkpoint, and returns a (T, 59) action chunk:

    hand14 + root9(xyz, rot6d) + 4 EE poses * 9(xyz, rot6d)
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr
import tyro
import uvicorn


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
    raise ValueError(f"Expected array with 1-3 dims, got {arr.shape}")


def _video_bthwc(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        return arr[None, None]
    if arr.ndim == 4:
        return arr[None]
    if arr.ndim == 5:
        return arr
    raise ValueError(f"Expected video array with 3-5 dims, got {arr.shape}")


def _rpy_to_rot6d_cols_rowmajor(rpy: np.ndarray) -> np.ndarray:
    """Convert xyz Euler angles to [r00, r01, r10, r11, r20, r21]."""
    rpy = np.asarray(rpy, dtype=np.float64)
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.stack(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            sy * cp,
            sy * sp * sr + cy * cr,
            -sp,
            cp * sr,
        ],
        axis=-1,
    ).astype(np.float32)


def _state49_to_state52(states: Any) -> np.ndarray:
    state = _btd(states)
    if state.shape[-1] == 52:
        return state.astype(np.float32)
    if state.shape[-1] != 49:
        raise ValueError(f"Expected SIMPLE policy state49 or rot6d state52, got {state.shape}")

    root_xyz = state[..., 43:46]
    root_rpy = state[..., 46:49]
    root_rot6d = _rpy_to_rot6d_cols_rowmajor(root_rpy)
    return np.concatenate([state[..., :43], root_xyz, root_rot6d], axis=-1).astype(np.float32)


@dataclass
class Config:
    model_path: Path = Path(
        "/pfs/pfs-ilWc5D/yzh/Isaac-GR00T/outputs/"
        "task1-gr00t-n17-rot6d59-kimodo-textop/"
        "task1-gr00t-n17-rot6d59-kimodo-textop/checkpoint-120000"
    )
    backbone_path: Path = Path(
        "/pfs/pfs-ilWc5D/yzh/Psi0/huggingface/hub/"
        "models--nvidia--Cosmos-Reason2-2B/snapshots/"
        "9ce19a195e423419c349abfc86fd07178b230561"
    )
    modality_config_path: Path = Path(
        "/pfs/pfs-ilWc5D/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py"
    )
    host: str = "0.0.0.0"
    port: int = 22096
    device: str = "cuda"
    strict: bool = True
    # --- Real-Time Chunking (RTC) -------------------------------------------
    # When enabled the server keeps the last predicted (normalized) chunk and
    # seeds the next generation so the new chunk continues from the unexecuted
    # tail of the previous one.
    #   overlap = action_chunk_size (Tp) - action_exec_horizon (Ta)
    # e.g. Tp=40, Ta=34 -> overlap=6 (predict 40, execute 34, 6-frame carry-over).
    #
    # Modes:
    #   --enable-rtc            official soft vel_strength freeze (needs frozen/ramp)
    #   --prefix-rtc            Hard-rewrite prefix + per-token timestep AdaLN
    #                           (via gr00t_n17_prefix_rtc.apply_prefix_rtc; also
    #                           implies RTC continuity). The timestep convention
    #                           defaults to the checkpoint config.
    enable_rtc: bool = False
    prefix_rtc: bool = False
    prefix_rtc_timestep_mode: str | None = None
    # Number of chunk frames the downstream SIMPLE executor consumes before it
    # re-requests (must match POLICY_EXECUTION_HORIZON on the client side).
    action_exec_horizon: int = 34
    # Soft-freeze knobs for official --enable-rtc (ignored under --prefix-rtc).
    rtc_frozen_steps: int = 2
    rtc_ramp_rate: float = 2.0
    # --- TensorRT ------------------------------------------------------------
    # First-stage TRT support accelerates only the Qwen3-VL backbone
    # (ViT + LLM) and leaves the RTC action head in PyTorch.
    trt_engine_dir: Path | None = None
    trt_mode: str = "vit_llm_only"
    trt_deploy_dir: Path | None = None


class Server:
    STATE_PARTS = (
        ("rot59_left_hand", 0, 7),
        ("rot59_right_hand", 7, 14),
        ("rot59_left_leg", 14, 20),
        ("rot59_right_leg", 20, 26),
        ("rot59_waist", 26, 29),
        ("rot59_left_arm", 29, 36),
        ("rot59_right_arm", 36, 43),
        ("rot59_root", 43, 52),
    )
    ACTION_KEYS = (
        "rot59_hand",
        "rot59_root",
        "rot59_left_hand_pose",
        "rot59_right_hand_pose",
        "rot59_left_foot_pose",
        "rot59_right_foot_pose",
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._configure_offline_backbone()
        self._allow_local_cosmos_backbone()
        self._import_modality_config()

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(cfg.model_path),
            device=cfg.device,
            strict=cfg.strict,
        )
        self.prefix_rtc = bool(cfg.prefix_rtc)
        if self.prefix_rtc:
            import sys

            _deploy_dir = str(Path(__file__).resolve().parent)
            if _deploy_dir not in sys.path:
                sys.path.insert(0, _deploy_dir)
            from gr00t_n17_prefix_rtc import apply_prefix_rtc

            apply_prefix_rtc(
                self.policy,
                prefix_timestep_mode=cfg.prefix_rtc_timestep_mode,
            )
        self._setup_tensorrt_if_requested()

        horizon = len(self.policy.modality_configs["action"].delta_indices)
        print(f"[gr00t-rot6d59-server] loaded {cfg.model_path}")
        print(f"[gr00t-rot6d59-server] GR00T prediction horizon={horizon}")

        # --- RTC state ------------------------------------------------------
        # --prefix-rtc implies RTC continuity (hard-rewrite path).
        self.action_chunk_size = int(horizon)  # Tp (prediction horizon)
        self.enable_rtc = bool(cfg.enable_rtc) or self.prefix_rtc
        self.action_exec_horizon = int(cfg.action_exec_horizon)  # Ta
        self.rtc_overlap_steps = self.action_chunk_size - self.action_exec_horizon
        self.rtc_frozen_steps = int(cfg.rtc_frozen_steps)
        self.rtc_ramp_rate = float(cfg.rtc_ramp_rate)
        self.previous_normalized_action: np.ndarray | None = None
        if self.enable_rtc:
            if not (0 < self.rtc_overlap_steps < self.action_chunk_size):
                raise ValueError(
                    "RTC requires 0 < (action_chunk_size - action_exec_horizon) < action_chunk_size; "
                    f"got action_chunk_size={self.action_chunk_size}, "
                    f"action_exec_horizon={self.action_exec_horizon}, "
                    f"overlap={self.rtc_overlap_steps}"
                )
            self.rtc_frozen_steps = max(0, min(self.rtc_frozen_steps, self.rtc_overlap_steps))
            if self.prefix_rtc:
                resolved_timestep_mode = (
                    self.policy.model.action_head._prefix_rtc_timestep_mode
                )
                mode = (
                    "prefix-rtc (hard-rewrite + token AdaLN, "
                    f"timestep_mode={resolved_timestep_mode})"
                )
            else:
                mode = (
                    f"official soft_freeze(frozen={self.rtc_frozen_steps}, "
                    f"ramp={self.rtc_ramp_rate})"
                )
            print(
                "[gr00t-rot6d59-server] RTC enabled: "
                f"action_chunk_size(Tp)={self.action_chunk_size}, "
                f"action_exec_horizon(Ta)={self.action_exec_horizon}, "
                f"overlap={self.rtc_overlap_steps}, mode={mode}"
            )
        else:
            print("[gr00t-rot6d59-server] RTC disabled (stateless full-chunk prediction)")

    def _configure_offline_backbone(self) -> None:
        if not self.cfg.model_path.is_dir():
            raise FileNotFoundError(f"GR00T checkpoint does not exist: {self.cfg.model_path}")
        if not self.cfg.backbone_path.is_dir():
            raise FileNotFoundError(f"GR00T backbone directory does not exist: {self.cfg.backbone_path}")

        os.environ.setdefault("HF_HOME", str(self.cfg.backbone_path.parents[3]))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cfg.backbone_path.parents[2]))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        os.environ["GR00T_BACKBONE_PATH"] = str(self.cfg.backbone_path)

    @staticmethod
    def _allow_local_cosmos_backbone() -> None:
        import gr00t.model.gr00t_n1d7.gr00t_n1d7 as n1d7

        original = n1d7.get_backbone_cls

        def get_backbone_cls(config):
            model_name = str(config.model_name)
            if (
                "Cosmos-Reason2-2B" in model_name
                or "models--nvidia--Cosmos-Reason2-2B" in model_name
            ):
                from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

                return Qwen3Backbone
            return original(config)

        n1d7.get_backbone_cls = get_backbone_cls

    def _import_modality_config(self) -> None:
        if not self.cfg.modality_config_path.exists():
            raise FileNotFoundError(
                f"GR00T rot6d59 modality config does not exist: {self.cfg.modality_config_path}"
            )
        spec = importlib.util.spec_from_file_location(
            "unitree_g1_rot6d59_config", self.cfg.modality_config_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {self.cfg.modality_config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    def _resolve_trt_deploy_dir(self) -> Path:
        if self.cfg.trt_deploy_dir is not None:
            deploy_dir = self.cfg.trt_deploy_dir
            if (deploy_dir / "trt_model_forward.py").is_file():
                return deploy_dir
            raise FileNotFoundError(
                "Could not locate trt_model_forward.py under explicit "
                f"--trt-deploy-dir: {deploy_dir}"
            )

        candidates: list[Path] = []
        env_root = os.environ.get("GR00T_ROOT")
        if env_root:
            candidates.append(Path(env_root) / "scripts" / "deployment")
        modality_repo_root = self.cfg.modality_config_path.resolve().parents[1]
        candidates.append(modality_repo_root / "scripts" / "deployment")
        candidates.append(Path.cwd() / "scripts" / "deployment")

        for candidate in candidates:
            if (candidate / "trt_model_forward.py").is_file():
                return candidate
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not locate Isaac-GR00T scripts/deployment/trt_model_forward.py; "
            f"searched: {searched}. Set --trt-deploy-dir explicitly."
        )

    def _setup_tensorrt_if_requested(self) -> None:
        if self.cfg.trt_engine_dir is None:
            return
        engine_dir = self.cfg.trt_engine_dir
        if not engine_dir.is_dir():
            raise FileNotFoundError(f"TRT engine directory does not exist: {engine_dir}")
        trt_mode = str(self.cfg.trt_mode).strip()
        rtc_requested = bool(self.cfg.enable_rtc) or bool(self.cfg.prefix_rtc)
        if rtc_requested and trt_mode != "vit_llm_only":
            raise ValueError(
                "RTC/Prefix-RTC currently supports only --trt-mode vit_llm_only. "
                "Action-head TRT does not yet implement rtc_prev_action hard-prefix semantics."
            )

        import sys

        deploy_dir = self._resolve_trt_deploy_dir()
        if str(deploy_dir) not in sys.path:
            sys.path.insert(0, str(deploy_dir))
        from trt_model_forward import setup_tensorrt_engines

        setup_tensorrt_engines(self.policy, str(engine_dir), mode=trt_mode)
        print(
            "[gr00t-rot6d59-server] TensorRT enabled: "
            f"mode={trt_mode}, engine_dir={engine_dir}"
        )

    @staticmethod
    def _pick(action: dict[str, Any], key: str) -> np.ndarray:
        value = action.get(key, action.get(f"action.{key}"))
        if value is None:
            raise KeyError(f"GR00T response is missing action {key!r}")
        return _btd(value)

    @staticmethod
    def _pick_image(image_dict: dict[str, Any]) -> Any:
        for key in ("ego_view", "rgb_head_stereo_left", "head_stereo_left"):
            if key in image_dict:
                return image_dict[key]
        raise KeyError(
            "Missing ego image; expected one of ego_view/rgb_head_stereo_left/head_stereo_left"
        )

    def act(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            request = _numpy_decode(payload)
            state52 = _state49_to_state52(request["state"]["states"])
            image = self._pick_image(request["image"])
            instruction = request.get("instruction") or "move forward and grasp the green cylinder"
            history = request.get("history") or {}

            observation: dict[str, Any] = {
                "video": {"ego_view": _video_bthwc(image)},
                "state": {},
                "language": {"annotation.human.task_description": [[instruction]]},
            }
            for name, start, end in self.STATE_PARTS:
                observation["state"][name] = state52[..., start:end].astype(np.float32)

            # --- RTC continuity -------------------------------------------
            # On episode reset (or the first request) generate a fresh chunk from
            # pure noise. Otherwise seed the flow sampler with the unexecuted tail
            # of the previous chunk so the new chunk stays continuous.
            options: dict[str, Any] | None = None
            if self.enable_rtc:
                if isinstance(history, dict) and "reset" in history:
                    self.previous_normalized_action = None
                if self.previous_normalized_action is not None:
                    options = {
                        "rtc_prev_action": self.previous_normalized_action,
                        "action_horizon": int(self.previous_normalized_action.shape[1]),
                        "rtc_overlap_steps": self.rtc_overlap_steps,
                    }
                    if not self.prefix_rtc:
                        # Official soft-freeze path still needs these knobs.
                        options["rtc_frozen_steps"] = self.rtc_frozen_steps
                        options["rtc_ramp_rate"] = self.rtc_ramp_rate
                    mode = "prefix-rtc" if self.prefix_rtc else "soft_freeze"
                    print(
                        "[gr00t-rot6d59-server] RTC step: seeding new chunk with previous "
                        f"tail[{self.action_exec_horizon}:{self.action_chunk_size}] "
                        f"(overlap={self.rtc_overlap_steps}, mode={mode})"
                    )
                else:
                    print("[gr00t-rot6d59-server] RTC reset/first step: unconditioned generation")

            action, info = self.policy.get_action(observation, options)

            if self.enable_rtc:
                normalized_pred = info.get("normalized_action_pred")
                if normalized_pred is not None:
                    self.previous_normalized_action = np.asarray(
                        normalized_pred, dtype=np.float32
                    )

            chunks = [self._pick(action, key) for key in self.ACTION_KEYS]
            output = np.concatenate(chunks, axis=-1)
            if output.shape[0] == 1:
                output = output[0]
            if output.ndim != 2 or output.shape[1] != 59:
                raise ValueError(f"Expected GR00T rot6d59 output (T, 59), got {output.shape}")

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
