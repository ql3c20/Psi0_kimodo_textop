#!/usr/bin/env python3
"""Strict TensorRT drop-in for Kimodo's classifier-free-guided denoiser.

The module is inert unless explicitly imported and instantiated. The current
Kimodo server does neither, so adding this file does not alter PyTorch serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import torch


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
TORCH_DTYPES = {
    "DataType.FLOAT": torch.float32,
    "DataType.HALF": torch.float16,
    "DataType.BOOL": torch.bool,
    "DataType.INT32": torch.int32,
    "DataType.INT64": torch.int64,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KimodoTensorRTDenoiser(torch.nn.Module):
    """Execute one fixed-shape Kimodo CFG denoising step with TensorRT."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        expected_config: str | Path | None = None,
        expected_checkpoint: str | Path | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        super().__init__()
        import tensorrt as trt

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        self.metadata_path = (
            Path(metadata_path).expanduser().resolve()
            if metadata_path is not None
            else self.engine_path.with_suffix(".json")
        )
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"TensorRT metadata not found: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text())
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Kimodo TensorRT backend requires an available CUDA device")
        self._validate_metadata(
            expected_config=expected_config,
            expected_checkpoint=expected_checkpoint,
            trt_version=trt.__version__,
        )

        self.cfg_type_default = str(self.metadata["cfg_type"])
        self.cfg_weight = tuple(float(value) for value in self.metadata["cfg_weight"])
        self.frames = int(self.metadata["frames"])
        self.motion_dim = int(self.metadata["motion_dim"])
        self.model = None
        self._lock = threading.Lock()
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {self.engine_path}")

        self.tensor_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            dtype_name = str(self.engine.get_tensor_dtype(name))
            if dtype_name not in TORCH_DTYPES:
                raise TypeError(f"Unsupported TensorRT dtype for {name}: {dtype_name}")
            self.tensor_specs[name] = (shape, TORCH_DTYPES[dtype_name])
        self._validate_engine_contract()

        self.buffers: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in self.tensor_specs.items():
            tensor = torch.empty(shape, dtype=dtype, device=self.device).contiguous()
            self.buffers[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def _validate_metadata(
        self,
        *,
        expected_config: str | Path | None,
        expected_checkpoint: str | Path | None,
        trt_version: str,
    ) -> None:
        required = {
            "format_version",
            "distill_config",
            "distill_checkpoint",
            "distill_checkpoint_bytes",
            "frames",
            "motion_dim",
            "cfg_type",
            "cfg_weight",
            "versions",
            "gpu_capability",
            "artifacts",
            "sha256",
            "input_specs",
            "output_spec",
            "build_parameters",
        }
        missing = sorted(required - self.metadata.keys())
        if missing:
            raise ValueError(f"TensorRT metadata is missing fields: {missing}")
        if int(self.metadata["format_version"]) != 2:
            raise ValueError(
                f"Unsupported TensorRT metadata format: {self.metadata['format_version']}"
            )
        built_trt = str(self.metadata["versions"]["tensorrt"])
        if built_trt != trt_version:
            raise RuntimeError(
                f"TensorRT version mismatch: engine metadata={built_trt}, runtime={trt_version}"
            )
        built_capability = tuple(int(value) for value in self.metadata["gpu_capability"])
        runtime_capability = tuple(torch.cuda.get_device_capability(self.device))
        if built_capability != runtime_capability:
            raise RuntimeError(
                "GPU capability mismatch: "
                f"engine metadata={built_capability}, runtime={runtime_capability}"
            )
        self._validate_expected_path(
            "distill config",
            expected_config,
            self.metadata["distill_config"],
        )
        self._validate_expected_path(
            "distill checkpoint",
            expected_checkpoint,
            self.metadata["distill_checkpoint"],
        )
        checkpoint = Path(self.metadata["distill_checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Engine checkpoint no longer exists: {checkpoint}")
        if checkpoint.stat().st_size != int(self.metadata["distill_checkpoint_bytes"]):
            raise RuntimeError(f"Engine checkpoint size changed since build: {checkpoint}")
        config = Path(self.metadata["distill_config"]).expanduser().resolve()
        onnx = Path(self.metadata["artifacts"]["onnx"]).expanduser().resolve()
        recorded_engine = Path(self.metadata["artifacts"]["engine"]).expanduser().resolve()
        if recorded_engine != self.engine_path:
            raise RuntimeError(
                f"TensorRT engine path mismatch: metadata={recorded_engine}, "
                f"runtime={self.engine_path}"
            )
        for label, path in {
            "config": config,
            "checkpoint": checkpoint,
            "onnx": onnx,
            "engine": self.engine_path,
        }.items():
            if not path.is_file():
                raise FileNotFoundError(f"TensorRT {label} artifact not found: {path}")
            actual_hash = sha256_file(path)
            recorded_hash = self.metadata["sha256"].get(label)
            if actual_hash != recorded_hash:
                raise RuntimeError(
                    f"TensorRT {label} hash mismatch: "
                    f"metadata={recorded_hash}, runtime={actual_hash}"
                )

    @staticmethod
    def _validate_expected_path(label: str, expected: str | Path | None, built: str) -> None:
        if expected is None:
            return
        expected_path = Path(expected).expanduser().resolve()
        built_path = Path(built).expanduser().resolve()
        if expected_path != built_path:
            raise RuntimeError(
                f"TensorRT {label} mismatch: expected={expected_path}, engine={built_path}"
            )

    def _validate_engine_contract(self) -> None:
        required_engine_inputs = {
            "x",
            "x_pad_mask",
            "text_feat",
            "timesteps",
            "heading",
            "motion_mask",
            "observed_motion",
        }
        missing = sorted(required_engine_inputs - self.tensor_specs.keys())
        if missing:
            raise ValueError(f"TensorRT engine is missing inputs: {missing}")
        if OUTPUT_NAME not in self.tensor_specs:
            raise ValueError(f"TensorRT engine is missing output: {OUTPUT_NAME}")
        expected_motion = (1, int(self.metadata["frames"]), int(self.metadata["motion_dim"]))
        for name in ("x", "motion_mask", "observed_motion", OUTPUT_NAME):
            actual = self.tensor_specs[name][0]
            if actual != expected_motion:
                raise ValueError(
                    f"TensorRT {name} shape mismatch: expected={expected_motion}, actual={actual}"
                )

    @staticmethod
    def _weights_tuple(cfg_weight: float | Iterable[float]) -> tuple[float, ...]:
        if isinstance(cfg_weight, (float, int)):
            return (float(cfg_weight),)
        return tuple(float(value) for value in cfg_weight)

    def _validate_call(
        self,
        *,
        cfg_weight: float | Iterable[float],
        cfg_type: str | None,
        tensors: dict[str, torch.Tensor | None],
    ) -> None:
        requested_type = cfg_type or self.cfg_type_default
        if requested_type != self.cfg_type_default:
            raise ValueError(
                f"TensorRT cfg_type mismatch: engine={self.cfg_type_default}, request={requested_type}"
            )
        requested_weight = self._weights_tuple(cfg_weight)
        if len(requested_weight) != len(self.cfg_weight) or any(
            abs(actual - expected) > 1e-6
            for actual, expected in zip(requested_weight, self.cfg_weight, strict=True)
        ):
            raise ValueError(
                f"TensorRT cfg_weight mismatch: engine={self.cfg_weight}, request={requested_weight}"
            )
        for name, tensor in tensors.items():
            if tensor is None:
                raise ValueError(f"TensorRT input {name} must not be None")
        text_pad_mask = tensors["text_pad_mask"]
        assert text_pad_mask is not None
        if tuple(text_pad_mask.shape) != (1, 1):
            raise ValueError(
                f"TensorRT text_pad_mask shape mismatch: expected=(1, 1), "
                f"actual={tuple(text_pad_mask.shape)}"
            )
        for name in self.tensor_specs:
            if name == OUTPUT_NAME:
                continue
            tensor = tensors[name]
            assert tensor is not None
            expected_shape, _ = self.tensor_specs[name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"TensorRT {name} shape mismatch: expected={expected_shape}, "
                    f"actual={tuple(tensor.shape)}"
                )

    def forward(
        self,
        cfg_weight,
        x,
        x_pad_mask,
        text_feat,
        text_pad_mask,
        timesteps,
        first_heading_angle=None,
        motion_mask=None,
        observed_motion=None,
        cfg_type=None,
        **_unused,
    ):
        tensors = {
            "x": x,
            "x_pad_mask": x_pad_mask,
            "text_feat": text_feat,
            "text_pad_mask": text_pad_mask,
            "timesteps": timesteps,
            "heading": first_heading_angle,
            "motion_mask": motion_mask,
            "observed_motion": observed_motion,
        }
        self._validate_call(
            cfg_weight=cfg_weight,
            cfg_type=cfg_type,
            tensors=tensors,
        )
        with self._lock:
            stream = torch.cuda.current_stream(self.device)
            for name, destination in self.buffers.items():
                if name == OUTPUT_NAME:
                    continue
                source = tensors[name]
                assert source is not None
                destination.copy_(
                    source.to(device=self.device, dtype=destination.dtype).contiguous(),
                    non_blocking=True,
                )
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT Kimodo denoiser execution failed")
            output = self.buffers[OUTPUT_NAME].clone()
            stream.synchronize()
            return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline validation for Kimodo TRT backend.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--distill-config", type=Path, required=True)
    parser.add_argument("--distill-ckpt", type=Path, required=True)
    parser.add_argument(
        "--kimodo-root",
        type=Path,
        default=Path(os.environ.get("KIMODO_ROOT", "/home/ubuntu/yzh/kimodo_my")),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-mean-error", type=float, default=1e-3)
    parser.add_argument("--max-abs-error", type=float, default=1e-2)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--real-constraints", type=Path)
    parser.add_argument("--real-heading-source", type=Path)
    parser.add_argument("--prompt", default="perform the manipulation task in simulation")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    return parser.parse_args()


def import_model_loader(kimodo_root: Path):
    os.environ.setdefault("TEXT_ENCODER", "dummy")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")
    for path in (kimodo_root, kimodo_root / "scripts"):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from generate_g1_with_first_heading import _build_model_from_distill

    return _build_model_from_distill


def make_case(
    case_index: int,
    *,
    frames: int,
    motion_dim: int,
    device: str,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + case_index)
    scale = 0.25 + 0.25 * case_index
    x = torch.randn(1, frames, motion_dim, generator=generator, device=device) * scale
    observed = torch.randn(1, frames, motion_dim, generator=generator, device=device) * 0.1
    if case_index == 0:
        motion_mask = torch.zeros(1, frames, motion_dim, dtype=torch.int64, device=device)
    elif case_index == 1:
        motion_mask = torch.ones(1, frames, motion_dim, dtype=torch.int64, device=device)
    else:
        motion_mask = (
            torch.rand(1, frames, motion_dim, generator=generator, device=device) > 0.8
        ).to(torch.int64)
    return (
        x,
        torch.ones(1, frames, dtype=torch.bool, device=device),
        torch.zeros(1, 1, 4096, device=device),
        torch.ones(1, 1, dtype=torch.bool, device=device),
        torch.tensor([(case_index * 251) % 1000], dtype=torch.int64, device=device),
        torch.tensor([0.1 * case_index], dtype=torch.float32, device=device),
        motion_mask,
        observed,
    )


def validate_rejections(backend: KimodoTensorRTDenoiser, example: tuple[Any, ...]) -> None:
    tests = {
        "wrong_frames": lambda: backend(
            [2.0, 2.0],
            example[0][:, :-1],
            example[1][:, :-1],
            *example[2:],
            cfg_type="separated",
        ),
        "wrong_cfg_type": lambda: backend(
            [2.0, 2.0], *example, cfg_type="regular"
        ),
        "wrong_cfg_weight": lambda: backend(
            [1.0, 2.0], *example, cfg_type="separated"
        ),
    }
    for name, call in tests.items():
        try:
            call()
        except ValueError as error:
            print(f"rejection {name}: {error}")
        else:
            raise AssertionError(f"TensorRT backend failed to reject {name}")


def benchmark(label: str, call, repeats: int) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
    print(f"benchmark {label}: {elapsed_ms:.3f} ms/step")
    return elapsed_ms


def capture_real_calls(model: Any, args: argparse.Namespace, frames: int):
    if bool(args.real_constraints) != bool(args.real_heading_source):
        raise ValueError("--real-constraints and --real-heading-source must be used together")
    if args.real_constraints is None:
        return []
    constraints_path = args.real_constraints.expanduser().resolve()
    heading_path = args.real_heading_source.expanduser().resolve()
    if not constraints_path.is_file() or not heading_path.is_file():
        raise FileNotFoundError(
            f"Real validation inputs not found: {constraints_path}, {heading_path}"
        )

    kimodo_scripts = args.kimodo_root.expanduser().resolve() / "scripts"
    if str(kimodo_scripts) not in sys.path:
        sys.path.insert(0, str(kimodo_scripts))
    from generate_g1_with_first_heading import (
        extract_first_heading_angle_from_npz,
        load_constraints_lst,
    )

    captured = []

    def pre_hook(_module, positional, keywords):
        cfg_weight = tuple(float(value) for value in positional[0])
        tensors = tuple(value.detach().clone() for value in positional[1:9])
        captured.append((cfg_weight, tensors, keywords.get("cfg_type")))

    hook = model.denoiser.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        constraints = load_constraints_lst(str(constraints_path), model.skeleton)
        heading = extract_first_heading_angle_from_npz(str(heading_path))
        first_heading = torch.tensor(
            [heading], dtype=torch.float32, device=args.device
        )
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            model(
                args.prompt,
                frames,
                constraint_lst=constraints,
                num_denoising_steps=args.diffusion_steps,
                num_samples=1,
                multi_prompt=False,
                num_transition_frames=5,
                post_processing=False,
                return_numpy=False,
                first_heading_angle=first_heading,
                hard_project_observed_motion=True,
                cfg_type="separated",
                cfg_weight=[2.0, 2.0],
            )
    finally:
        hook.remove()
    print(f"captured real denoiser calls: {len(captured)}")
    if not captured:
        raise RuntimeError("Real validation captured no denoiser calls")
    return captured


def main() -> None:
    args = parse_args()
    if args.cases < 0 or args.benchmark_repeats <= 0:
        raise ValueError("--cases must be non-negative and --benchmark-repeats must be positive")
    if args.cases == 0 and args.real_constraints is None:
        raise ValueError("Use --cases > 0 or provide real validation inputs")
    kimodo_root = args.kimodo_root.expanduser().resolve()
    build_model = import_model_loader(kimodo_root)
    os.chdir(kimodo_root)
    print(f"Kimodo working directory: {kimodo_root}")
    model = build_model(
        distill_config_path=str(args.distill_config.resolve()),
        distill_ckpt_path=str(args.distill_ckpt.resolve()),
        device=args.device,
    )
    model.eval()
    backend = KimodoTensorRTDenoiser(
        args.engine,
        metadata_path=args.metadata,
        expected_config=args.distill_config,
        expected_checkpoint=args.distill_ckpt,
        device=args.device,
    )
    results = []
    example = None
    for case_index in range(args.cases):
        example = make_case(
            case_index,
            frames=backend.frames,
            motion_dim=backend.motion_dim,
            device=args.device,
            seed=args.seed,
        )
        with torch.inference_mode():
            reference = model.denoiser(
                [2.0, 2.0], *example, cfg_type="separated"
            )
            actual = backend([2.0, 2.0], *example, cfg_type="separated")
        error = (actual - reference).abs()
        reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
        actual_nonfinite = int((~torch.isfinite(actual)).sum().item())
        result = {
            "case": case_index,
            "mean_abs_error": float(error.mean().item()),
            "max_abs_error": float(error.max().item()),
            "pytorch_nonfinite": reference_nonfinite,
            "tensorrt_nonfinite": actual_nonfinite,
        }
        results.append(result)
        print("case:", json.dumps(result, sort_keys=True))
    real_calls = capture_real_calls(model, args, backend.frames)
    for call_index, (cfg_weight, real_example, cfg_type) in enumerate(real_calls):
        with torch.inference_mode():
            reference = model.denoiser(cfg_weight, *real_example, cfg_type=cfg_type)
            actual = backend(cfg_weight, *real_example, cfg_type=cfg_type)
        error = (actual - reference).abs()
        reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
        actual_nonfinite = int((~torch.isfinite(actual)).sum().item())
        result = {
            "case": f"real_{call_index:03d}",
            "mean_abs_error": float(error.mean().item()),
            "max_abs_error": float(error.max().item()),
            "pytorch_nonfinite": reference_nonfinite,
            "tensorrt_nonfinite": actual_nonfinite,
        }
        results.append(result)
        print("case:", json.dumps(result, sort_keys=True))
        example = real_example
    assert example is not None
    validate_rejections(backend, example)

    with torch.inference_mode():
        torch_ms = benchmark(
            "torch",
            lambda: model.denoiser([2.0, 2.0], *example, cfg_type="separated"),
            args.benchmark_repeats,
        )
        trt_ms = benchmark(
            "tensorrt",
            lambda: backend([2.0, 2.0], *example, cfg_type="separated"),
            args.benchmark_repeats,
        )
    worst_mean = max(result["mean_abs_error"] for result in results)
    worst_abs = max(result["max_abs_error"] for result in results)
    nonfinite = sum(
        result["pytorch_nonfinite"] + result["tensorrt_nonfinite"]
        for result in results
    )
    summary = {
        "cases": args.cases,
        "worst_mean_abs_error": worst_mean,
        "worst_max_abs_error": worst_abs,
        "torch_ms": torch_ms,
        "tensorrt_ms": trt_ms,
        "speedup": torch_ms / trt_ms,
        "nonfinite": nonfinite,
    }
    print("summary:", json.dumps(summary, sort_keys=True))
    if nonfinite or worst_mean > args.max_mean_error or worst_abs > args.max_abs_error:
        raise RuntimeError(
            "TensorRT numeric tolerance exceeded: "
            f"mean={worst_mean} (limit={args.max_mean_error}), "
            f"max={worst_abs} (limit={args.max_abs_error})"
        )


if __name__ == "__main__":
    main()
