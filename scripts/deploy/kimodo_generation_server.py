#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R


KIMODO_ROOT = Path(os.environ.get("KIMODO_ROOT", "/pfs/pfs-ilWc5D/yzh/kimodo_my")).resolve()
KIMODO_ANCHOR_MODE = os.environ.get("KIMODO_ANCHOR_MODE", "policy_only")
if KIMODO_ANCHOR_MODE not in {"current", "policy_delta", "policy_only"}:
    raise ValueError(
        f"Unsupported KIMODO_ANCHOR_MODE={KIMODO_ANCHOR_MODE!r}; "
        "use policy_only, policy_delta, or current."
    )
os.chdir(KIMODO_ROOT)
if str(KIMODO_ROOT) not in sys.path:
    sys.path.insert(0, str(KIMODO_ROOT))
SCRIPTS_DIR = KIMODO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_g1_with_first_heading import (  # noqa: E402
    _build_model_from_distill,
    _generate_flowmatch_motion,
    _resolve_cfg_kwargs,
    _resolve_sampler_mode,
    extract_first_heading_angle_from_npz,
    load_model,
    load_constraints_lst,
    resolve_num_frames,
)
from kimodo.exports.mujoco import MujocoQposConverter  # noqa: E402
from kimodo.constraints import FullBodyConstraintSet  # noqa: E402
from kimodo.geometry import matrix_to_axis_angle  # noqa: E402
from kimodo.model.registry import get_model_info  # noqa: E402
from kimodo.tools import seed_everything  # noqa: E402


def _csv_path(output_base: str) -> Path:
    path = Path(output_base)
    if path.suffix == ".csv":
        return path
    return path.with_suffix(".csv")


class GenerateRequest(BaseModel):
    prompt: str
    duration: float
    constraints: str | None = None
    heading_source_npz: str
    output: str
    diffusion_steps: int = 20
    sampler: str = "auto"
    num_samples: int = 1
    num_transition_frames: int = 5
    hard_project_observed_motion: bool = True
    hard_project_prefix_frames: int = 0
    hard_project_release_frames: int = 0
    cfg_type: str = "separated"
    cfg_weight: list[float] = [2.0, 2.0]
    seed: int | None = None
    start_qpos_mujoco: list[float] | None = None
    prefix_qpos_mujoco: list[list[float]] | None = None


class KimodoGenerationServer:
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_trt_value = os.environ.get("KIMODO_USE_TRT", "0").strip()
        if use_trt_value not in {"0", "1"}:
            raise ValueError(
                f"KIMODO_USE_TRT must be 0 or 1, got {use_trt_value!r}"
            )
        text_encoder_mode = os.environ.get("KIMODO_TEXT_ENCODER_MODE", "original").strip()
        if text_encoder_mode not in {"original", "dummy"}:
            raise ValueError(
                "KIMODO_TEXT_ENCODER_MODE must be original or dummy, "
                f"got {text_encoder_mode!r}"
            )
        print(f"[kimodo-server] Using device: {self.device}", flush=True)

        distill_config = os.environ.get(
            "KIMODO_DISTILL_CONFIG",
            str(
                KIMODO_ROOT
                / "outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/resolved_config.yaml"
            ),
        )
        distill_ckpt = os.environ.get(
            "KIMODO_DISTILL_CKPT",
            str(
                KIMODO_ROOT
                / "outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/ema_final.pt"
            ),
        )

        if distill_config and distill_ckpt:
            self.model = _build_model_from_distill(
                distill_config_path=distill_config,
                distill_ckpt_path=distill_ckpt,
                device=self.device,
            )
            print(
                f"[kimodo-server] Loaded distill student config={distill_config} ckpt={distill_ckpt}",
                flush=True,
            )
        else:
            self.model, resolved_model = load_model(
                os.environ.get("KIMODO_MODEL", None),
                device=self.device,
                default_family="Kimodo",
                return_resolved_name=True,
            )
            info = get_model_info(resolved_model)
            display = info.display_name if info else resolved_model
            print(f"[kimodo-server] Loaded model: {display} ({resolved_model})", flush=True)

        self.text_encoder_mode = text_encoder_mode
        if text_encoder_mode == "dummy":
            from kimodo.model.dummy_text_encoder import DummyTextEncoder

            original_text_encoder = self.model.text_encoder
            llm_dim = int(getattr(original_text_encoder, "llm_dim", 4096))
            self.model.text_encoder = DummyTextEncoder(llm_dim=llm_dim).to(self.device).eval()
            del original_text_encoder
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                "[kimodo-server] WARNING: dummy text encoder enabled; "
                f"prompt text is ignored and replaced by zeros (llm_dim={llm_dim})",
                flush=True,
            )
        else:
            print("[kimodo-server] Original text encoder enabled", flush=True)

        self.inference_backend = "pytorch"
        if use_trt_value == "1":
            if not distill_config or not distill_ckpt:
                raise ValueError(
                    "KIMODO_USE_TRT=1 requires KIMODO_DISTILL_CONFIG and "
                    "KIMODO_DISTILL_CKPT"
                )
            engine_value = os.environ.get("KIMODO_TRT_ENGINE_PATH", "").strip()
            if not engine_value:
                raise ValueError(
                    "KIMODO_USE_TRT=1 requires explicit KIMODO_TRT_ENGINE_PATH"
                )
            metadata_value = os.environ.get("KIMODO_TRT_METADATA_PATH", "").strip()
            if not metadata_value:
                raise ValueError(
                    "KIMODO_USE_TRT=1 requires explicit KIMODO_TRT_METADATA_PATH"
                )
            from kimodo_trt_backend import KimodoTensorRTDenoiser

            self.model.denoiser = KimodoTensorRTDenoiser(
                engine_value,
                metadata_path=metadata_value,
                expected_config=distill_config,
                expected_checkpoint=distill_ckpt,
                device=self.device,
            )
            self.inference_backend = "tensorrt"
            print(
                f"[kimodo-server] TensorRT enabled: engine={Path(engine_value).resolve()}",
                flush=True,
            )
        else:
            print("[kimodo-server] PyTorch backend enabled", flush=True)

        self.converter = MujocoQposConverter(self.model.skeleton)
        qpos_heading_alignment_value = os.environ.get(
            "KIMODO_QPOS_ALIGN_ROOT_HEADING", "0"
        ).strip()
        if qpos_heading_alignment_value not in {"0", "1"}:
            raise ValueError(
                "KIMODO_QPOS_ALIGN_ROOT_HEADING must be 0 or 1, "
                f"got {qpos_heading_alignment_value!r}"
            )
        self.qpos_heading_aligner = None
        self.qpos_heading_max_correction_rad = float(
            os.environ.get(
                "KIMODO_QPOS_ALIGN_ROOT_HEADING_MAX_DELTA_RAD",
                str(np.pi / 2.0),
            )
        )
        if qpos_heading_alignment_value == "1":
            from kimodo_qpos_heading_alignment import align_qpos_root_heading

            if (
                not np.isfinite(self.qpos_heading_max_correction_rad)
                or self.qpos_heading_max_correction_rad <= 0.0
            ):
                raise ValueError(
                    "KIMODO_QPOS_ALIGN_ROOT_HEADING_MAX_DELTA_RAD must be "
                    "finite and positive"
                )
            self.qpos_heading_aligner = align_qpos_root_heading
            print(
                "[kimodo-server] G1 qpos root-heading alignment enabled: "
                f"max_delta={self.qpos_heading_max_correction_rad:.6f}rad",
                flush=True,
            )
        else:
            print(
                "[kimodo-server] G1 qpos root-heading alignment disabled",
                flush=True,
            )
        qpos_projection_value = os.environ.get(
            "KIMODO_QPOS_CONSTRAINT_PROJECTION", "0"
        ).strip()
        if qpos_projection_value not in {"0", "1"}:
            raise ValueError(
                "KIMODO_QPOS_CONSTRAINT_PROJECTION must be 0 or 1, "
                f"got {qpos_projection_value!r}"
            )
        self.qpos_projector = None
        if qpos_projection_value == "1":
            from kimodo_qpos_projection import G1QposConstraintProjector

            projection_xml = os.environ.get(
                "KIMODO_QPOS_PROJECTION_XML", self.converter.xml_path
            )
            self.qpos_projector = G1QposConstraintProjector(
                projection_xml,
                target_scale_m=float(
                    os.environ.get("KIMODO_QPOS_PROJECTION_TARGET_SCALE_M", "0.01")
                ),
                joint_regularization_rad=float(
                    os.environ.get("KIMODO_QPOS_PROJECTION_JOINT_REG_RAD", "0.35")
                ),
                max_nfev=int(
                    os.environ.get("KIMODO_QPOS_PROJECTION_MAX_NFEV", "10")
                ),
                optimizer_tolerance=float(
                    os.environ.get("KIMODO_QPOS_PROJECTION_OPTIMIZER_TOL", "0.001")
                ),
            )
            print(
                "[kimodo-server] G1 qpos constraint projection enabled: "
                f"xml={self.qpos_projector.xml_path}",
                flush=True,
            )
        else:
            print("[kimodo-server] G1 qpos constraint projection disabled", flush=True)

    def _fullbody_constraint_from_qpos(self, qpos_mujoco: list[float] | np.ndarray) -> FullBodyConstraintSet:
        qpos = np.asarray(qpos_mujoco, dtype=np.float32).reshape(-1, 36)
        num_frames = int(qpos.shape[0])
        if num_frames <= 0:
            raise ValueError("prefix_qpos_mujoco must contain at least one frame")
        device = self.device
        dtype = torch.float32

        root_pos_m = torch.tensor(qpos[:, :3], dtype=dtype, device=device)
        mujoco_to_kimodo = self.converter.mujoco_to_kimodo_matrix.to(device=device, dtype=dtype)
        root_positions = torch.matmul(mujoco_to_kimodo[None, ...], root_pos_m[..., None]).squeeze(-1)

        root_rot_m = R.from_quat(qpos[:, 3:7], scalar_first=True).as_matrix().astype(np.float32)
        root_rot_m = torch.tensor(root_rot_m, dtype=dtype, device=device)
        root_rot_k = torch.matmul(
            torch.matmul(mujoco_to_kimodo[None, ...], root_rot_m),
            mujoco_to_kimodo.T[None, ...],
        )

        local_rot_mats = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 1, 3, 3).repeat(
            1, num_frames, self.model.skeleton.nbjoints, 1, 1
        )
        local_rot_mats[:, :, int(self.model.skeleton.root_idx)] = root_rot_k[None, ...]

        joint_dofs = torch.tensor(qpos[:, 7:36], dtype=dtype, device=device).reshape(1, num_frames, 29)
        local_rot_mats = self.converter._joint_dofs_to_local_rot_mats(
            joint_dofs,
            local_rot_mats,
            device=torch.device(device),
            dtype=dtype,
            use_relative=False,
        )

        global_joints_rots, global_joints_positions, _ = self.model.skeleton.fk(local_rot_mats[0], root_positions)
        return FullBodyConstraintSet(
            skeleton=self.model.skeleton,
            frame_indices=torch.arange(num_frames, dtype=torch.long),
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=root_positions[:, [0, 2]],
        )

    @torch.inference_mode()
    def generate(self, req: GenerateRequest) -> dict[str, Any]:
        total_t0 = time.perf_counter()
        if req.seed is not None:
            seed_everything(req.seed)

        num_frames, max_constraint_frame = resolve_num_frames(req.duration, self.model.fps, req.constraints)
        constraint_lst = []
        if req.constraints:
            constraint_lst = load_constraints_lst(req.constraints, self.model.skeleton)
        if req.prefix_qpos_mujoco is not None:
            prefix_qpos = np.asarray(req.prefix_qpos_mujoco, dtype=np.float32)
            if prefix_qpos.ndim != 2 or prefix_qpos.shape[1] != 36:
                raise ValueError(
                    "prefix_qpos_mujoco must have shape (T, 36), "
                    f"got {prefix_qpos.shape}"
                )
            if prefix_qpos.shape[0] > num_frames:
                raise ValueError(
                    "prefix_qpos_mujoco is longer than the generated motion: "
                    f"{prefix_qpos.shape[0]} > {num_frames}"
                )
            constraint_lst.append(self._fullbody_constraint_from_qpos(prefix_qpos))
            print(
                f"[kimodo-server] Applying dense qpos prefix constraint: frames={prefix_qpos.shape[0]}",
                flush=True,
            )
        elif req.start_qpos_mujoco is not None:
            constraint_lst.append(self._fullbody_constraint_from_qpos(req.start_qpos_mujoco))

        first_heading_angle = extract_first_heading_angle_from_npz(req.heading_source_npz)
        first_heading_tensor = torch.tensor([first_heading_angle], dtype=torch.float32, device=self.device)

        args = type(
            "Args",
            (),
            {
                "sampler": req.sampler,
                "distill_config": os.environ.get("KIMODO_DISTILL_CONFIG"),
                "cfg_type": req.cfg_type,
                "cfg_weight": req.cfg_weight,
            },
        )()
        cfg_kwargs = _resolve_cfg_kwargs(args)
        sampler_mode = _resolve_sampler_mode(args)

        if sampler_mode == "diffusion":
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
            inference_t0 = time.perf_counter()
            output = self.model(
                req.prompt,
                num_frames,
                constraint_lst=constraint_lst,
                num_denoising_steps=req.diffusion_steps,
                num_samples=req.num_samples,
                multi_prompt=False,
                num_transition_frames=req.num_transition_frames,
                post_processing=False,
                return_numpy=True,
                first_heading_angle=first_heading_tensor,
                hard_project_observed_motion=req.hard_project_observed_motion,
                hard_project_prefix_frames=req.hard_project_prefix_frames,
                hard_project_release_frames=req.hard_project_release_frames,
                **cfg_kwargs,
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
            inference_elapsed = time.perf_counter() - inference_t0
        else:
            if req.num_samples != 1:
                raise ValueError("Flowmatch sampler currently supports num_samples=1 in this server.")
            if req.cfg_type == "nocfg":
                raise ValueError("Flowmatch sampler does not support cfg_type=nocfg here.")

            lengths = torch.tensor([num_frames], device=self.device)
            max_frames = int(num_frames)
            motion_pad_mask = torch.arange(max_frames, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            observed_motion, motion_mask = None, None
            if constraint_lst:
                observed_motion, motion_mask = self.model.motion_rep.create_conditions_from_constraints_batched(
                    constraint_lst,
                    lengths,
                    to_normalize=True,
                    device=self.device,
                )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
            inference_t0 = time.perf_counter()
            motion = _generate_flowmatch_motion(
                model=self.model,
                texts=[req.prompt],
                max_frames=max_frames,
                num_steps=req.diffusion_steps,
                pad_mask=motion_pad_mask,
                first_heading_angle=first_heading_tensor,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=cfg_kwargs["cfg_weight"],
                cfg_type=cfg_kwargs["cfg_type"],
                hard_project_observed_motion=req.hard_project_observed_motion,
                hard_project_prefix_frames=req.hard_project_prefix_frames,
                hard_project_release_frames=req.hard_project_release_frames,
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
            inference_elapsed = time.perf_counter() - inference_t0
            output = self.model.motion_rep.inverse(motion, is_normalized=True, return_numpy=True)

        n_samples = int(output["posed_joints"].shape[0])
        if n_samples != 1:
            raise ValueError(f"Expected one Kimodo sample, got {n_samples}")

        single = {
            k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
            for k, v in output.items()
        }
        output_path = Path(req.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path.with_suffix(".npz"), **single)

        qpos = self.converter.dict_to_qpos(output, self.device)
        if self.qpos_heading_aligner is not None:
            qpos, heading_alignment = self.qpos_heading_aligner(
                qpos,
                output["global_root_heading"],
                max_correction_rad=self.qpos_heading_max_correction_rad,
            )
            print(
                "[kimodo-server] G1 qpos root-heading alignment "
                f"frames={heading_alignment.frames} "
                f"mean_delta={heading_alignment.mean_abs_correction_rad:.4f}rad "
                f"max_delta={heading_alignment.max_abs_correction_rad:.4f}rad",
                flush=True,
            )
        if self.qpos_projector is not None and req.constraints:
            qpos, projection = self.qpos_projector.project(qpos, req.constraints)
            print(
                "[kimodo-server] G1 qpos constraint projection "
                f"frames={projection.constrained_frames} "
                f"mean={projection.before_mean_m:.4f}->{projection.after_mean_m:.4f}m "
                f"max={projection.before_max_m:.4f}->{projection.after_max_m:.4f}m "
                f"joint_max={projection.max_joint_correction_rad:.3f}rad "
                f"elapsed={projection.elapsed_s:.3f}s",
                flush=True,
            )
        csv_path = _csv_path(req.output)
        self.converter.save_csv(qpos, str(csv_path))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        total_elapsed = time.perf_counter() - total_t0
        print(
            f"[kimodo-server] Kimodo inference time={inference_elapsed:.3f}s "
            f"total time={total_elapsed:.3f}s",
            flush=True,
        )
        return {
            "status": "ok",
            "csv_path": str(csv_path),
            "npz_path": str(output_path.with_suffix(".npz")),
            "frames": num_frames,
            "max_constraint_frame": max_constraint_frame,
            "sampler": sampler_mode,
        }


server = KimodoGenerationServer()
app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, int | str]:
    return {
        "keyframe_step": max(1, int(os.environ.get("KIMODO_KEYFRAME_STEP", "10"))),
        "anchor_mode": KIMODO_ANCHOR_MODE,
        "text_encoder_mode": server.text_encoder_mode,
        "inference_backend": server.inference_backend,
        "qpos_root_heading_alignment": (
            "enabled" if server.qpos_heading_aligner is not None else "disabled"
        ),
        "qpos_constraint_projection": "enabled" if server.qpos_projector is not None else "disabled",
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    return server.generate(req)


def main() -> None:
    host = os.environ.get("KIMODO_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("KIMODO_SERVER_PORT", "22185"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
