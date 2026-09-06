#!/usr/bin/env python3
"""Export and build fixed-length TensorRT engines for a Kimodo student denoiser.

This is an offline tool. It is not imported by the Kimodo server and therefore
does not change the existing PyTorch serving path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_NAMES = (
    "x",
    "x_pad_mask",
    "text_feat",
    "text_pad_mask",
    "timesteps",
    "heading",
    "motion_mask",
    "observed_motion",
)
OUTPUT_NAME = "pred_clean"
ONNX_OPSET = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-frame Kimodo denoiser ONNX/TensorRT engines."
    )
    parser.add_argument("--distill-config", type=Path, required=True)
    parser.add_argument("--distill-ckpt", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        help="One or more fixed frame counts. Required unless --check-only is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <checkpoint-dir>/trt_engines.",
    )
    parser.add_argument(
        "--kimodo-root",
        type=Path,
        default=Path(os.environ.get("KIMODO_ROOT", "/home/ubuntu/yzh/kimodo_my")),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help=(
            "Disable TensorRT TF32 tactics so FP32 engine math matches a "
            "PyTorch runtime with torch.backends.cuda.matmul.allow_tf32=False."
        ),
    )
    parser.add_argument("--cfg-type", choices=("separated",), default="separated")
    parser.add_argument("--cfg-weight-text", type=float, default=2.0)
    parser.add_argument("--cfg-weight-constraint", type=float, default=2.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--onnx-only", action="store_true")
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="Build from an existing verified ONNX without replacing it.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dependencies/config/checkpoint without loading the model or writing files.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def resolved_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing_onnx(
    metadata_path: Path,
    onnx_path: Path,
    config_path: Path,
    ckpt_path: Path,
    frame_count: int,
) -> None:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Existing ONNX not found for --build-only: {onnx_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"ONNX metadata not found for --build-only: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "config": sha256_file(config_path),
        "checkpoint": sha256_file(ckpt_path),
        "onnx": sha256_file(onnx_path),
    }
    if int(metadata.get("format_version", -1)) != 2:
        raise RuntimeError(f"Unsupported ONNX metadata format: {metadata_path}")
    if int(metadata.get("frames", -1)) != frame_count:
        raise RuntimeError(
            f"ONNX frame contract mismatch: expected={frame_count}, "
            f"metadata={metadata.get('frames')}"
        )
    for label, actual in expected.items():
        recorded = metadata.get("sha256", {}).get(label)
        if recorded != actual:
            raise RuntimeError(
                f"ONNX metadata {label} hash mismatch: recorded={recorded}, actual={actual}"
            )
    print(f"verified existing ONNX and metadata: {onnx_path}")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    config_path = resolved_file(args.distill_config, "distill config")
    ckpt_path = resolved_file(args.distill_ckpt, "distill checkpoint")
    kimodo_root = args.kimodo_root.expanduser().resolve()
    if not (kimodo_root / "kimodo").is_dir():
        raise FileNotFoundError(f"Kimodo package directory not found under: {kimodo_root}")
    if not args.check_only and not args.frames:
        raise ValueError("--frames is required unless --check-only is used")
    if args.frames and any(frame <= 0 for frame in args.frames):
        raise ValueError(f"All --frames values must be positive, got {args.frames}")
    if args.workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ckpt_path.parent / "trt_engines"
    )
    return config_path, ckpt_path, kimodo_root, output_dir


def check_dependencies() -> dict[str, str]:
    import onnx
    import tensorrt as trt
    import torch

    versions = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "tensorrt": trt.__version__,
    }
    if tuple(int(part) for part in trt.__version__.split(".")[:2]) < (10, 15):
        raise RuntimeError(f"TensorRT 10.15 or newer is required, got {trt.__version__}")
    print("dependencies:", json.dumps(versions, sort_keys=True))
    return versions


def check_config_and_checkpoint(config_path: Path, ckpt_path: Path) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    student = cfg.model.student_denoiser
    student_target = str(student.get("_target_", ""))
    num_layers = int(student.get("num_layers", -1))
    num_base_steps = int(cfg.model.num_base_steps)
    if not student_target:
        raise ValueError(f"Missing model.student_denoiser._target_ in {config_path}")

    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {ckpt_path}")
    if "student" in checkpoint and isinstance(checkpoint["student"], dict):
        checkpoint_format = "student"
        tensor_count = len(checkpoint["student"])
    elif (
        "ema" in checkpoint
        and isinstance(checkpoint["ema"], dict)
        and isinstance(checkpoint["ema"].get("shadow"), dict)
    ):
        checkpoint_format = "ema.shadow"
        tensor_count = len(checkpoint["ema"]["shadow"])
    elif checkpoint and all(isinstance(key, str) for key in checkpoint):
        checkpoint_format = "state_dict"
        tensor_count = len(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    details = {
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "checkpoint_bytes": ckpt_path.stat().st_size,
        "checkpoint_format": checkpoint_format,
        "checkpoint_tensor_count": tensor_count,
        "student_target": student_target,
        "student_num_layers": num_layers,
        "num_base_steps": num_base_steps,
    }
    print("model inputs:", json.dumps(details, sort_keys=True))
    return details


def load_export_model(
    kimodo_root: Path,
    config_path: Path,
    ckpt_path: Path,
    device: str,
):
    """Load the student and its lightweight CFG wrapper, without LLM weights."""
    for path in (kimodo_root, kimodo_root / "scripts"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    import torch
    from omegaconf import OmegaConf

    from kimodo.model.dummy_text_encoder import DummyTextEncoder
    from kimodo.model.kimodo_model import Kimodo
    from kimodo.model.loading import instantiate_from_dict

    cfg = OmegaConf.load(config_path)
    student_cfg = OmegaConf.to_container(cfg.model.student_denoiser, resolve=True)
    denoiser = instantiate_from_dict(student_cfg).to(device)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "student" in checkpoint and isinstance(checkpoint["student"], dict):
        state = checkpoint["student"]
        source = "student"
    elif (
        "ema" in checkpoint
        and isinstance(checkpoint["ema"], dict)
        and isinstance(checkpoint["ema"].get("shadow"), dict)
    ):
        state = checkpoint["ema"]["shadow"]
        source = "ema.shadow"
    elif checkpoint and all(isinstance(key, str) for key in checkpoint):
        state = checkpoint
        source = "state_dict"
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    missing, unexpected = denoiser.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Denoiser checkpoint mismatch for {source}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    denoiser.eval()
    model = Kimodo(
        denoiser=denoiser,
        text_encoder=DummyTextEncoder(llm_dim=int(student_cfg.get("llm_shape", [1, 4096])[-1])),
        num_base_steps=int(cfg.model.num_base_steps),
        device=device,
        cfg_type="separated",
    )
    print("loaded student denoiser with DummyTextEncoder; LLM weights were not instantiated")
    return model


def make_step_module(denoiser: Any, cfg_weights: tuple[float, float], cfg_type: str):
    import torch

    class StepModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.denoiser = denoiser

        def forward(
            self,
            x,
            x_pad_mask,
            text_feat,
            text_pad_mask,
            timesteps,
            heading,
            motion_mask,
            observed_motion,
        ):
            return self.denoiser(
                cfg_weights,
                x,
                x_pad_mask,
                text_feat,
                text_pad_mask,
                timesteps,
                heading,
                motion_mask,
                observed_motion,
                cfg_type=cfg_type,
            )

    return StepModule()


def make_example(frame_count: int, motion_dim: int, device: str):
    import torch

    return (
        torch.randn(1, frame_count, motion_dim, device=device),
        torch.ones(1, frame_count, dtype=torch.bool, device=device),
        torch.zeros(1, 1, 4096, device=device),
        torch.ones(1, 1, dtype=torch.bool, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.float32, device=device),
        torch.zeros(1, frame_count, motion_dim, dtype=torch.int64, device=device),
        torch.zeros(1, frame_count, motion_dim, device=device),
    )


def export_onnx(step_module: Any, example: tuple[Any, ...], onnx_path: Path):
    import torch

    temporary = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    with torch.inference_mode():
        reference = step_module(*example)
        torch.onnx.export(
            step_module,
            example,
            str(temporary),
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )
    os.replace(temporary, onnx_path)
    print(f"exported ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return reference


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: float,
    *,
    disable_tf32: bool,
):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gib * (1 << 30)),
    )
    if disable_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    print(f"TensorRT TF32 enabled: {config.get_flag(trt.BuilderFlag.TF32)}")
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")
    temporary = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary.write_bytes(bytes(serialized))
    os.replace(temporary, engine_path)
    print(
        f"built TensorRT engine in {time.perf_counter() - started:.2f}s: "
        f"{engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)"
    )
    return logger


def latency_stats(run: Any, *, warmup: int = 10, samples: int = 50) -> dict[str, float | int]:
    import torch

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        torch.cuda.synchronize()
        started = time.perf_counter()
        run()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()

    def percentile(fraction: float) -> float:
        position = fraction * (len(timings) - 1)
        lower = int(position)
        upper = min(lower + 1, len(timings) - 1)
        weight = position - lower
        return timings[lower] * (1.0 - weight) + timings[upper] * weight

    return {
        "samples": samples,
        "median_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "p95_ms": percentile(0.95),
    }


def check_engine(
    engine_path: Path,
    example: tuple[Any, ...],
    reference: Any,
    logger: Any,
    step_module: Any,
):
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"Failed to create execution context: {engine_path}")
    engine_names = {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }
    source = dict(zip(INPUT_NAMES, example, strict=True))
    buffers: dict[str, Any] = {}
    for name in INPUT_NAMES:
        if name not in engine_names:
            continue
        tensor = source[name].contiguous()
        buffers[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())
    output = torch.empty_like(reference).contiguous()
    context.set_tensor_address(OUTPUT_NAME, output.data_ptr())
    def run_trt() -> None:
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {engine_path}")

    run_trt()
    torch.cuda.synchronize()
    if tuple(output.shape) != tuple(reference.shape):
        raise RuntimeError(
            f"TensorRT output shape mismatch: reference={tuple(reference.shape)}, "
            f"actual={tuple(output.shape)}"
        )
    reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
    output_nonfinite = int((~torch.isfinite(output)).sum().item())
    if reference_nonfinite or output_nonfinite:
        raise RuntimeError(
            "Non-finite synthetic output: "
            f"pytorch={reference_nonfinite}, tensorrt={output_nonfinite}"
        )
    error = (output - reference).abs()
    with torch.inference_mode():
        pytorch_latency = latency_stats(lambda: step_module(*example))
    tensorrt_latency = latency_stats(run_trt)
    result = {
        "mean_abs_error": float(error.mean().item()),
        "max_abs_error": float(error.max().item()),
        "pytorch_nonfinite": reference_nonfinite,
        "tensorrt_nonfinite": output_nonfinite,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "pytorch_latency": pytorch_latency,
        "tensorrt_latency": tensorrt_latency,
    }
    print("numeric check:", json.dumps(result, sort_keys=True))
    return result


def write_metadata(
    metadata_path: Path,
    *,
    args: argparse.Namespace,
    config_path: Path,
    ckpt_path: Path,
    frame_count: int,
    motion_dim: int,
    versions: dict[str, str],
    numeric: dict[str, float] | None,
    onnx_path: Path,
    engine_path: Path,
    example: tuple[Any, ...],
    reference: Any,
) -> None:
    import torch

    input_specs = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for name, tensor in zip(INPUT_NAMES, example, strict=True)
    }
    metadata = {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "build_mode": (
            "onnx_only"
            if args.onnx_only
            else "build_from_onnx"
            if args.build_only
            else "onnx_and_engine"
        ),
        "distill_config": str(config_path),
        "distill_checkpoint": str(ckpt_path),
        "distill_checkpoint_bytes": ckpt_path.stat().st_size,
        "artifacts": {
            "onnx": str(onnx_path),
            "engine": str(engine_path) if engine_path.is_file() else None,
        },
        "sha256": {
            "config": sha256_file(config_path),
            "checkpoint": sha256_file(ckpt_path),
            "onnx": sha256_file(onnx_path),
            "engine": sha256_file(engine_path) if engine_path.is_file() else None,
        },
        "frames": frame_count,
        "motion_dim": motion_dim,
        "cfg_type": args.cfg_type,
        "cfg_weight": [args.cfg_weight_text, args.cfg_weight_constraint],
        "input_names": list(INPUT_NAMES),
        "input_specs": input_specs,
        "output_name": OUTPUT_NAME,
        "output_spec": {
            "shape": list(reference.shape),
            "dtype": str(reference.dtype).removeprefix("torch."),
        },
        "versions": {**versions, "cuda": torch.version.cuda},
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "gpu_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "build_parameters": {
            "device": args.device,
            "onnx_opset": ONNX_OPSET,
            "precision": "fp32",
            "tf32_enabled": not args.disable_tf32,
            "workspace_gib": args.workspace_gib,
        },
        "numeric_check": numeric,
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    print(f"wrote metadata: {metadata_path}")


def main() -> None:
    args = parse_args()
    config_path, ckpt_path, kimodo_root, output_dir = validate_args(args)
    versions = check_dependencies()
    check_config_and_checkpoint(config_path, ckpt_path)
    if args.check_only:
        print("check-only passed; no files written")
        return

    import torch

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("TensorRT export/build requires a CUDA device")
    # Kimodo configs resolve checkpoint-owned assets (for example stats/motion)
    # relative to KIMODO_ROOT. Match the generation server's loading semantics.
    os.chdir(kimodo_root)
    print(f"Kimodo working directory: {kimodo_root}")
    model = load_export_model(
        kimodo_root,
        config_path,
        ckpt_path,
        args.device,
    )
    torch.backends.mha.set_fastpath_enabled(False)
    cfg_weights = (args.cfg_weight_text, args.cfg_weight_constraint)
    step_module = make_step_module(model.denoiser, cfg_weights, args.cfg_type)
    step_module.eval().to(args.device)
    motion_dim = int(model.motion_rep.motion_rep_dim)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_count in dict.fromkeys(args.frames):
        onnx_path = output_dir / f"kimodo_T{frame_count}.onnx"
        engine_path = output_dir / f"kimodo_T{frame_count}.trt"
        engine_candidate_path = engine_path.with_suffix(engine_path.suffix + ".candidate")
        metadata_path = output_dir / f"kimodo_T{frame_count}.json"
        if args.build_only:
            validate_existing_onnx(
                metadata_path,
                onnx_path,
                config_path,
                ckpt_path,
                frame_count,
            )
            expected = (engine_path, engine_candidate_path)
        else:
            expected = (onnx_path, metadata_path) if args.onnx_only else (
                onnx_path,
                engine_path,
                metadata_path,
            )
        existing = [path for path in expected if path.exists()]
        if existing and not args.force:
            raise FileExistsError(
                "Refusing to replace existing outputs without --force: "
                + ", ".join(str(path) for path in existing)
            )
        example = make_example(frame_count, motion_dim, args.device)
        if args.build_only:
            with torch.inference_mode():
                reference = step_module(*example)
        else:
            reference = export_onnx(step_module, example, onnx_path)
        numeric = None
        if not args.onnx_only:
            logger = build_engine(
                onnx_path,
                engine_candidate_path,
                args.workspace_gib,
                disable_tf32=args.disable_tf32,
            )
            numeric = check_engine(
                engine_candidate_path,
                example,
                reference,
                logger,
                step_module,
            )
            os.replace(engine_candidate_path, engine_path)
            print(f"promoted validated TensorRT engine: {engine_path}")
        write_metadata(
            metadata_path,
            args=args,
            config_path=config_path,
            ckpt_path=ckpt_path,
            frame_count=frame_count,
            motion_dim=motion_dim,
            versions=versions,
            numeric=numeric,
            onnx_path=onnx_path,
            engine_path=engine_path,
            example=example,
            reference=reference,
        )


if __name__ == "__main__":
    main()
