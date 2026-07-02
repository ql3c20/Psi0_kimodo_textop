#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch saved Psi0-Kimodo chunks into one executed qpos trajectory.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/psi0_kimodo_eval"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/psi0_kimodo_eval/stitched_executed_qpos50.csv"))
    parser.add_argument(
        "--pose-npz",
        type=Path,
        default=None,
        help="Optional Kimodo-viewer-compatible NPZ. Defaults to OUTPUT with .kimodo.npz suffix.",
    )
    parser.add_argument("--start-chunk", type=int, default=None)
    parser.add_argument("--end-chunk", type=int, default=None, help="Inclusive chunk index.")
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--chunk-policy-frames", type=int, default=30)
    parser.add_argument("--exec-frames", type=int, default=24)
    parser.add_argument("--blend-frames", type=int, default=4)
    parser.add_argument("--force-first-heading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", type=Path, default=None, help="Optional root XY/height plot PNG.")
    return parser.parse_args()


def _chunk_index(path: Path) -> int:
    return int(path.name.split("_")[-1])


def _select_chunks(work_dir: Path, start: int | None, end: int | None, num_chunks: int | None) -> list[Path]:
    chunks = sorted(work_dir.glob("chunk_*"), key=_chunk_index)
    if start is not None:
        chunks = [chunk for chunk in chunks if _chunk_index(chunk) >= start]
    if end is not None:
        chunks = [chunk for chunk in chunks if _chunk_index(chunk) <= end]
    if num_chunks is not None:
        chunks = chunks[:num_chunks]
    chunks = [chunk for chunk in chunks if (chunk / "g1_generated.csv").exists() and (chunk / "heading_source.npz").exists()]
    if not chunks:
        raise FileNotFoundError(f"No complete chunk_* folders found under {work_dir}")
    return chunks


def _resample_qpos(qpos: np.ndarray, input_fps: float, output_fps: float, target_frames: int) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None]
    t_in = np.arange(qpos.shape[0], dtype=np.float64) / float(input_fps)
    t_out = np.arange(target_frames, dtype=np.float64) / float(output_fps)
    t_out = np.clip(t_out, t_in[0], t_in[-1])

    out = np.zeros((target_frames, qpos.shape[1]), dtype=np.float64)
    for dim in (0, 1, 2):
        out[:, dim] = np.interp(t_out, t_in, qpos[:, dim])

    quat_wxyz = qpos[:, 3:7]
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)
    interp_quat = Slerp(t_in, R.from_quat(quat_xyzw))(t_out).as_quat()
    out[:, 3:7] = np.concatenate([interp_quat[:, 3:4], interp_quat[:, 0:3]], axis=1)

    for dim in range(7, qpos.shape[1]):
        out[:, dim] = np.interp(t_out, t_in, qpos[:, dim])
    return out


def _load_chunk_qpos50(chunk: Path, args: argparse.Namespace) -> np.ndarray:
    qpos = np.loadtxt(chunk / "g1_generated.csv", delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None]
    heading_data = np.load(chunk / "heading_source.npz")
    heading = heading_data["qpos"].reshape(-1, 36)[0].astype(np.float64)
    if "anchor_xy" in heading_data.files:
        anchor_xy = np.asarray(heading_data["anchor_xy"], dtype=np.float64).reshape(2)
    else:
        anchor_xy = heading[:2]

    qpos50 = _resample_qpos(
        qpos,
        input_fps=args.source_fps,
        output_fps=args.output_fps,
        target_frames=args.chunk_policy_frames,
    )
    qpos50[:, 0] += anchor_xy[0]
    qpos50[:, 1] += anchor_xy[1]
    if args.force_first_heading and "anchor_xy" not in heading_data.files:
        qpos50[0] = heading
    return qpos50.astype(np.float32)


def _blend_qpos_start(prev_qpos: np.ndarray, next_qpos: np.ndarray, blend_frames: int) -> np.ndarray:
    blend_frames = min(int(blend_frames), len(prev_qpos), len(next_qpos))
    if blend_frames <= 0:
        return next_qpos
    out = next_qpos.copy()
    prev_tail = prev_qpos[-blend_frames:]
    for idx in range(blend_frames):
        alpha = float(idx + 1) / float(blend_frames + 1)
        out[idx, :3] = (1.0 - alpha) * prev_tail[idx, :3] + alpha * next_qpos[idx, :3]
        out[idx, 7:] = (1.0 - alpha) * prev_tail[idx, 7:] + alpha * next_qpos[idx, 7:]

        prev_quat = R.from_quat(prev_tail[idx, [4, 5, 6, 3]])
        next_quat = R.from_quat(next_qpos[idx, [4, 5, 6, 3]])
        interp = Slerp([0.0, 1.0], R.concatenate([prev_quat, next_quat]))([alpha]).as_quat()[0]
        out[idx, 3:7] = np.asarray([interp[3], interp[0], interp[1], interp[2]], dtype=out.dtype)
    return out


def _save_plot(qpos: np.ndarray, path: Path, fps: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot export.")
        return

    t = np.arange(qpos.shape[0]) / float(fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(qpos[:, 0], qpos[:, 1], linewidth=1.5)
    axes[0].set_title("Root XY")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[1].plot(t, qpos[:, 2], label="root z")
    axes[1].plot(t, qpos[:, 7:].std(axis=1), label="joint std")
    axes[1].set_title("Root Height / Joint Spread")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _joint_to_body_name(joint_name: str) -> str | None:
    if joint_name == "pelvis_skel":
        return "pelvis"
    if joint_name == "waist_pitch_skel":
        return "torso_link"
    if joint_name.endswith("_skel"):
        return joint_name[: -len("_skel")] + "_link"
    return None


def _qpos_to_kimodo_pose(qpos: np.ndarray) -> dict[str, np.ndarray]:
    import mujoco
    import torch

    from kimodo.assets import skeleton_asset_path
    from kimodo.skeleton import G1Skeleton34

    skeleton = G1Skeleton34()
    xml_path = str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    qpos = np.asarray(qpos, dtype=np.float64)
    n = qpos.shape[0]
    j = skeleton.nbjoints
    posed_joints = np.zeros((n, j, 3), dtype=np.float32)
    global_rot_mats = np.zeros((n, j, 3, 3), dtype=np.float32)

    body_id_cache = {}
    for name in skeleton.bone_order_names:
        body_name = _joint_to_body_name(name)
        if body_name is None:
            body_id_cache[name] = -1
        else:
            body_id_cache[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for idx, joint_name in enumerate(skeleton.bone_order_names):
            body_id = body_id_cache[joint_name]
            if body_id >= 0:
                p_m = np.asarray(data.xpos[body_id], dtype=np.float64)
                r_m = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                posed_joints[t, idx] = np.array([p_m[1], p_m[2], p_m[0]], dtype=np.float32)
                global_rot_mats[t, idx] = (MUJOCO_TO_KIMODO @ r_m @ MUJOCO_TO_KIMODO.T).astype(np.float32)
            else:
                parent = int(skeleton.joint_parents[idx].item())
                if parent >= 0:
                    posed_joints[t, idx] = posed_joints[t, parent]
                    global_rot_mats[t, idx] = global_rot_mats[t, parent]
                else:
                    posed_joints[t, idx] = 0.0
                    global_rot_mats[t, idx] = np.eye(3, dtype=np.float32)

        min_y = posed_joints[t, :, 1].min()
        posed_joints[t, :, 1] -= min_y

    root_positions = posed_joints[:, int(skeleton.root_idx), :].copy()
    local_rot_mats = np.tile(np.eye(3, dtype=np.float32), (n, j, 1, 1))
    global_root_heading = np.zeros((n, 2), dtype=np.float32)
    root_quat = qpos[:, 3:7]
    yaw_m = R.from_quat(
        np.concatenate([root_quat[:, 1:4], root_quat[:, 0:1]], axis=1)
    ).as_euler("xyz")[:, 2]
    global_root_heading[:, 0] = np.cos(yaw_m).astype(np.float32)
    global_root_heading[:, 1] = np.sin(yaw_m).astype(np.float32)

    try:
        local_rot_t, global_rot_t = torch.from_numpy(local_rot_mats), torch.from_numpy(global_rot_mats)
        # Keep this try block intentionally lightweight: viewer paths mostly need posed_joints/global_rot_mats.
        local_rot_mats = local_rot_t.numpy()
        global_rot_mats = global_rot_t.numpy()
    except Exception:
        pass

    return {
        "local_rot_mats": local_rot_mats.astype(np.float32),
        "global_rot_mats": global_rot_mats.astype(np.float32),
        "posed_joints": posed_joints.astype(np.float32),
        "root_positions": root_positions.astype(np.float32),
        "smooth_root_pos": root_positions.astype(np.float32),
        "foot_contacts": np.zeros((n, 4), dtype=bool),
        "global_root_heading": global_root_heading.astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    chunks = _select_chunks(args.work_dir, args.start_chunk, args.end_chunk, args.num_chunks)
    stitched = []
    chunk_rows = []
    for chunk in chunks:
        qpos50 = _load_chunk_qpos50(chunk, args)
        take = min(args.exec_frames, qpos50.shape[0])
        segment = qpos50[:take]
        if stitched and args.blend_frames > 0:
            segment = _blend_qpos_start(stitched[-1], segment, args.blend_frames)
        stitched.append(segment)
        chunk_rows.append({"chunk": chunk.name, "frames_taken": int(take)})

    out = np.concatenate(stitched, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, out, delimiter=",", fmt="%.8f")
    qpos_npz = args.output.with_suffix(".qpos.npz")
    pose_npz = args.pose_npz or args.output.with_suffix(".kimodo.npz")
    np.savez(qpos_npz, qpos=out, fps=float(args.output_fps), chunks=np.asarray(chunk_rows, dtype=object))
    np.savez(pose_npz, **_qpos_to_kimodo_pose(out))

    meta = {
        "work_dir": str(args.work_dir),
        "output_csv": str(args.output),
        "qpos_npz": str(qpos_npz),
        "pose_npz": str(pose_npz),
        "num_chunks": len(chunks),
        "num_frames": int(out.shape[0]),
        "fps": float(args.output_fps),
        "duration_sec": float(out.shape[0] / args.output_fps),
        "exec_frames_per_chunk": int(args.exec_frames),
        "blend_frames": int(args.blend_frames),
        "chunks": chunk_rows,
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.plot is not None:
        _save_plot(out, args.plot, args.output_fps)
        meta["plot"] = str(args.plot)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
