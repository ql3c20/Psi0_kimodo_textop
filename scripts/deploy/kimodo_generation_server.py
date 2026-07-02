#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R


KIMODO_ROOT = Path(os.environ.get("KIMODO_ROOT", "/pfs/pfs-ilWc5D/yzh/kimodo_my")).resolve()
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


class KimodoGenerationServer:
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

        self.converter = MujocoQposConverter(self.model.skeleton)

    def _fullbody_start_constraint_from_qpos(self, qpos_mujoco: list[float] | np.ndarray) -> FullBodyConstraintSet:
        qpos = np.asarray(qpos_mujoco, dtype=np.float32).reshape(1, 36)
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
            1, 1, self.model.skeleton.nbjoints, 1, 1
        )
        local_rot_mats[:, :, int(self.model.skeleton.root_idx)] = root_rot_k[:, None]

        joint_dofs = torch.tensor(qpos[:, 7:36], dtype=dtype, device=device).reshape(1, 1, 29)
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
            frame_indices=torch.tensor([0], dtype=torch.long),
            global_joints_positions=global_joints_positions,
            global_joints_rots=global_joints_rots,
            smooth_root_2d=root_positions[:, [0, 2]],
        )

    @torch.inference_mode()
    def generate(self, req: GenerateRequest) -> dict[str, Any]:
        if req.seed is not None:
            seed_everything(req.seed)

        num_frames, max_constraint_frame = resolve_num_frames(req.duration, self.model.fps, req.constraints)
        constraint_lst = []
        if req.constraints:
            constraint_lst = load_constraints_lst(req.constraints, self.model.skeleton)
        if req.start_qpos_mujoco is not None:
            constraint_lst.append(self._fullbody_start_constraint_from_qpos(req.start_qpos_mujoco))

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
        csv_path = _csv_path(req.output)
        self.converter.save_csv(qpos, str(csv_path))
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


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    return server.generate(req)


def main() -> None:
    host = os.environ.get("KIMODO_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("KIMODO_SERVER_PORT", "22185"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
