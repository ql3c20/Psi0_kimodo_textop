from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import mujoco
import numpy as np


PSI0_ROOT = Path(__file__).resolve().parents[2]
SIMPLE_ROOT = PSI0_ROOT / "third_party/SIMPLE"
TEXTOP_ROOT = PSI0_ROOT.parent / "TextOp/TextOpTracker"

sys.path.insert(0, str(SIMPLE_ROOT / "src"))

from simple.baselines.textop_tracker_adapter import (  # noqa: E402
    DEFAULT_BODY_NAMES,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    TextOpTrackerAdapter,
)


def _load_deploy_module():
    viewer_module = types.ModuleType("mujoco_viewer")
    viewer_module.MujocoViewer = type("MujocoViewer", (), {})
    sys.modules.setdefault("mujoco_viewer", viewer_module)
    path = TEXTOP_ROOT / "scripts/deploy_mujoco.py"
    spec = importlib.util.spec_from_file_location("textop_deploy_mujoco", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body_fk(model: mujoco.MjModel, qpos36: np.ndarray, body_qpos_adrs: np.ndarray):
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in DEFAULT_BODY_NAMES]
    if min(body_ids) < 0:
        missing = [n for n, i in zip(DEFAULT_BODY_NAMES, body_ids) if i < 0]
        raise ValueError(f"Missing bodies: {missing}")
    pos = np.empty((len(qpos36), len(body_ids), 3), dtype=np.float32)
    quat = np.empty((len(qpos36), len(body_ids), 4), dtype=np.float32)
    for i, qpos in enumerate(qpos36):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[:7] = qpos[:7]
        data.qpos[body_qpos_adrs] = qpos[7:36]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pos[i] = data.xpos[body_ids]
        quat[i] = data.xquat[body_ids]
    return pos, quat


def _set_qpos36(model: mujoco.MjModel, body_qpos_adrs: np.ndarray, qpos36: np.ndarray):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[:7] = qpos36[:7]
    data.qpos[body_qpos_adrs] = qpos36[7:36]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def _max_report(name: str, a: np.ndarray, b: np.ndarray):
    diff = np.asarray(a) - np.asarray(b)
    idx = int(np.argmax(np.abs(diff)))
    print(
        f"{name}: max_abs={np.max(np.abs(diff)):.6g}, "
        f"mean_abs={np.mean(np.abs(diff)):.6g}, idx={idx}, diff={diff.reshape(-1)[idx]:.6g}"
    )


def _obs_block_report(obs_a: np.ndarray, obs_b: np.ndarray):
    names = [
        "command", "motion_anchor_pos_b", "motion_anchor_ori_b",
        "robot_anchor_pos_w", "robot_anchor_ori_w", "projected_gravity",
        "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel",
        "actions", "motion_ee_pos_b", "motion_ee_ori_b",
    ]
    dims = [128, 30, 60, 3, 6, 3, 3, 3, 29, 29, 29, 120, 240]
    start = 0
    for name, dim in zip(names, dims):
        a = obs_a[start:start + dim]
        b = obs_b[start:start + dim]
        print(
            f"  {name:22s} max_abs={np.max(np.abs(a-b)):.6g} "
            f"mean_abs={np.mean(np.abs(a-b)):.6g}"
        )
        start += dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(PSI0_ROOT / "outputs/kimodo_eval_ep0_policy_only_v2/stitched_executed_qpos50.csv"),
    )
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    deploy = _load_deploy_module()

    qpos36 = np.loadtxt(args.csv, delimiter=",").astype(np.float32)
    if qpos36.ndim == 1:
        qpos36 = qpos36[None]
    if qpos36.shape[1] != 36:
        raise ValueError(f"Expected 36 columns, got {qpos36.shape}")

    textop_xml = TEXTOP_ROOT / "source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act.xml"
    simple_xml = SIMPLE_ROOT / "third_party/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
    textop_model = mujoco.MjModel.from_xml_path(str(textop_xml))
    simple_model = mujoco.MjModel.from_xml_path(str(simple_xml))
    textop_adapter = TextOpTrackerAdapter(textop_model)
    simple_adapter = TextOpTrackerAdapter(simple_model)

    t = args.frame
    q0 = qpos36[t]

    textop_body_pos, textop_body_quat = _body_fk(textop_model, qpos36, textop_adapter.body_qpos_adrs)
    simple_body_pos, simple_body_quat = _body_fk(simple_model, qpos36, simple_adapter.body_qpos_adrs)

    print("== Model / FK difference for same qpos36 ==")
    print("csv", args.csv, "frames", qpos36.shape[0], "frame", t)
    print("textop nq nu mass", textop_model.nq, textop_model.nu, float(textop_model.body_mass.sum()))
    print("simple nq nu mass", simple_model.nq, simple_model.nu, float(simple_model.body_mass.sum()))
    _max_report("body_pos(simple-textop)", simple_body_pos[t], textop_body_pos[t])
    _max_report("body_quat(simple-textop)", simple_body_quat[t], textop_body_quat[t])
    print("qpos root xyz", q0[:3])
    print("textop pelvis xpos", textop_body_pos[t, 0], "delta", textop_body_pos[t, 0] - q0[:3])
    print("simple pelvis xpos", simple_body_pos[t, 0], "delta", simple_body_pos[t, 0] - q0[:3])

    ref_textop = textop_adapter.prepare_reference(qpos36)
    ref_simple = simple_adapter.prepare_reference(qpos36)
    _max_report("latent(simple-textop)", ref_simple["latents"], ref_textop["latents"])

    # Deploy parity: TextOp original deploy observation vs our adapter on TextOp XML.
    motion_loader = SimpleNamespace(
        joint_pos=qpos36[:, 7:36][:, MUJOCO_TO_ISAACLAB],
        joint_vel=np.zeros((qpos36.shape[0], 29), dtype=np.float32),
        body_pos=textop_body_pos,
        body_ori=textop_body_quat,
        body_names=DEFAULT_BODY_NAMES,
        anchor_body_name="pelvis",
        anchor_body_index=0,
        future_steps=textop_adapter.future_steps,
        motion_ae_latents=ref_textop["latents"],
        T=qpos36.shape[0],
        motion_body_index_by_name={n: i for i, n in enumerate(DEFAULT_BODY_NAMES)},
        motion_body_index=lambda name: DEFAULT_BODY_NAMES.index(name),
    )
    textop_data = mujoco.MjData(textop_model)
    textop_data.qpos[:] = 0.0
    textop_data.qpos[3] = 1.0
    textop_data.qpos[:3] = motion_loader.body_pos[t, 0]
    textop_data.qpos[3:7] = motion_loader.body_ori[t, 0]
    textop_data.qpos[7:] = motion_loader.joint_pos[t][ISAACLAB_TO_MUJOCO]
    textop_data.qvel[:] = 0.0
    mujoco.mj_forward(textop_model, textop_data)

    terms = deploy.get_observation_terms(task=deploy.TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE)
    obs_deploy = deploy.compute_observation(
        textop_data,
        motion_loader,
        t,
        np.zeros(29, dtype=np.float32),
        task=deploy.TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE,
        observation_terms=terms,
    )
    obs_adapter_textop = textop_adapter._obs683(
        textop_data, qpos36, ref_textop["body_pos"], ref_textop["body_quat"], ref_textop["latents"], t
    )
    print("\n== A1: adapter vs deploy_mujoco on TextOp XML ==")
    _max_report("obs(adapter_textop-deploy)", obs_adapter_textop, obs_deploy)
    _obs_block_report(obs_adapter_textop, obs_deploy)

    action_deploy = textop_adapter.policy_session.run(
        None, {textop_adapter.policy_input: obs_deploy.reshape(1, -1).astype(np.float32)}
    )[0].reshape(-1)
    action_adapter = textop_adapter.policy_session.run(
        None, {textop_adapter.policy_input: obs_adapter_textop.reshape(1, -1).astype(np.float32)}
    )[0].reshape(-1)
    _max_report("policy_action(adapter_textop-deploy)", action_adapter, action_deploy)

    # TextOp XML vs SIMPLE XML, same qpos reference and zero velocities.
    simple_data = _set_qpos36(simple_model, simple_adapter.body_qpos_adrs, q0)
    obs_simple = simple_adapter._obs683(
        simple_data, qpos36, ref_simple["body_pos"], ref_simple["body_quat"], ref_simple["latents"], t
    )
    print("\n== B/A2: SIMPLE XML obs vs TextOp XML obs, both initialized to same qpos36 ==")
    _max_report("obs(simple-textop)", obs_simple, obs_adapter_textop)
    _obs_block_report(obs_simple, obs_adapter_textop)
    action_simple = simple_adapter.policy_session.run(
        None, {simple_adapter.policy_input: obs_simple.reshape(1, -1).astype(np.float32)}
    )[0].reshape(-1)
    _max_report("policy_action(simple-textop)", action_simple, action_adapter)
    target_simple = simple_adapter.target_from_reference(ref_simple, simple_data, t)
    target_textop2 = textop_adapter.target_from_reference(ref_textop, textop_data, t)
    _max_report("target_q(simple-textop)", target_simple, target_textop2)


if __name__ == "__main__":
    main()
