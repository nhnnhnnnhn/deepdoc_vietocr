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
print(f"torch={torch.__version__} cuda_available={cuda_available}")
if cuda_available:
    count = torch.cuda.device_count()
    print(f"cuda_device_count={count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        arch_list = [a for a in getattr(torch.cuda, "get_arch_list", list)() if a.startswith("sm_")]
        if arch_list:
            min_sm = min(int(a[3:]) for a in arch_list)
            device_sm = major * 10 + minor
            compat = "OK" if device_sm >= min_sm else f"INCOMPATIBLE (SM {major}.{minor} < min SM {min_sm//10}.{min_sm%10})"
        else:
            compat = f"SM {major}.{minor}"
        print(f"cuda_device_{i}={name} [{compat}]")
elif require_gpu:
    raise SystemExit(
        "GPU is required but torch.cuda.is_available() is false. "
        "Start the service with Docker GPU support, for example: docker compose up with gpus: all."
    )

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"onnxruntime_providers={providers}")
    if require_gpu and "CUDAExecutionProvider" not in providers:
        raise SystemExit(
            "GPU is required but ONNX Runtime CUDAExecutionProvider is unavailable. "
            "Check the NVIDIA container runtime and CUDA/cuDNN compatibility."
        )
except Exception as exc:
    if require_gpu:
        raise
    print(f"onnxruntime check skipped after error: {exc}", file=sys.stderr)
PY

exec "$@"
