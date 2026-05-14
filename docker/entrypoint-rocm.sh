#!/usr/bin/env bash
set -euo pipefail

mkdir -p input output web_runs log

python -m py_compile web_app.py full_pipeline.py

python - <<'PY'
import os
import sys

require_gpu = os.getenv("REQUIRE_GPU", "1").strip().lower() in {"1", "true", "yes", "y", "on"}

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Failed to import torch: {exc}") from exc

cuda_available = torch.cuda.is_available()
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
runtime_label = f"ROCm/HIP {torch.version.hip}" if is_rocm else "CUDA"
print(f"torch={torch.__version__} runtime={runtime_label} gpu_available={cuda_available}")

if cuda_available:
    count = torch.cuda.device_count()
    print(f"gpu_device_count={count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        if is_rocm:
            props = torch.cuda.get_device_properties(i)
            arch = getattr(props, "gcnArchName", "unknown")
            print(f"gpu_device_{i}={name} [gcnArch={arch}]")
        else:
            major, minor = torch.cuda.get_device_capability(i)
            print(f"gpu_device_{i}={name} [SM {major}.{minor}]")
elif require_gpu:
    raise SystemExit(
        f"GPU is required but torch.cuda.is_available() is false ({runtime_label}). "
        "Ensure /dev/kfd and /dev/dri are mounted (docker-compose.rocm.yml devices:) "
        "and the container user is in the video and render groups (group_add:)."
    )

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"onnxruntime_providers={providers}")

    gpu_eps = {"CUDAExecutionProvider", "MIGraphXExecutionProvider", "ROCMExecutionProvider"}
    active_gpu_eps = [p for p in providers if p in gpu_eps]

    if require_gpu and not active_gpu_eps:
        raise SystemExit(
            "GPU is required but no GPU ONNX Runtime provider is available. "
            f"Available providers: {providers}. "
            "Expected MIGraphXExecutionProvider for ROCm."
        )
    if active_gpu_eps:
        print(f"onnxruntime_gpu_provider={active_gpu_eps[0]}")
        if active_gpu_eps[0] == "MIGraphXExecutionProvider":
            print("NOTE: MIGraphX compiles ONNX models on first inference. "
                  "The first OCR request may take 30-90 seconds longer than usual.")
except Exception as exc:
    if require_gpu:
        raise
    print(f"onnxruntime check skipped after error: {exc}", file=sys.stderr)
PY

exec "$@"
