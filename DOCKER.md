# Docker GPU Web UI

This project ships GPU-enabled Docker setups for the FastAPI web UI — one for NVIDIA CUDA and one for AMD ROCm.

## Requirements

- Docker Engine or Docker Desktop with Compose v2.
- NVIDIA driver installed on the host.
- NVIDIA Container Toolkit available to Docker.

Quick host GPU check:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## Configure

`docker-compose.yml` reads `.env` at runtime. The image build does not copy `.env` or bake API keys.

The image itself contains the application code, web UI, `conf/`, `onnx/`, and VietOCR weights. Runtime output is stored in Docker named volumes, not host bind mounts.

For Ollama running on the host, use this shape in `.env`:

```env
AI_BASE_URL=http://host.docker.internal:11434/v1
AI_MODEL=qwen3-vl:8b
```

For a cloud OpenAI-compatible provider, set that provider's base URL, key, and model.

## Build And Run

```powershell
docker compose build
docker compose up
```

Open:

```text
http://127.0.0.1:8000
```

The compose file uses Docker named volumes for runtime data:

- `deepdoc_input`
- `deepdoc_output`
- `deepdoc_web_runs`
- `deepdoc_log`

It does not mount host `conf/` or `onnx/`; changing prompts, code, or model files requires rebuilding the image.

At startup the container runs `py_compile` and checks both PyTorch CUDA and ONNX Runtime CUDA provider. Set `REQUIRE_GPU=0` only if you intentionally want to start without GPU validation.

---

## AMD ROCm 7 (docker-compose.rocm.yml)

### Requirements

- Docker Engine with Compose v2.
- ROCm 7.x driver installed on the host (`amdgpu-dkms`).
- `/dev/kfd` and `/dev/dri` present on the host.
- Host user in the `video` and `render` groups:

```bash
sudo usermod -aG video,render $USER
```

Quick host GPU check:

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  rocm/onnxruntime:rocm7.2.3_ub24.04_ort1.23_torch2.10.0 \
  rocminfo | grep -E "Name|Marketing"
```

### Configure

Copy `.env.rocm.example` to `.env` and adjust values. Key variable:

```env
HIP_VISIBLE_DEVICES=0   # 0 = first GPU, 0,1 = first two, unset = all
DEVICE=auto             # auto, cpu, cuda, cuda:<id>, rocm, or rocm:<id>
REQUIRE_GPU=1
```

### Build And Run

```bash
docker compose -f docker-compose.rocm.yml build
docker compose -f docker-compose.rocm.yml up
```

Open `http://127.0.0.1:8000`.

### GPU Provider

The ROCm image uses **MIGraphXExecutionProvider** (AMD's official ONNX Runtime backend for ROCm 7+). The `module/ocr.py` code auto-detects whichever GPU provider is available — `CUDAExecutionProvider` on NVIDIA, `MIGraphXExecutionProvider` on AMD — so the same Python code runs on both platforms.

### First-Run Compilation

MIGraphX compiles the ONNX models (`det.onnx`, `layout.onnx`, `tsr.onnx`) to its internal representation on the first inference. **Expect the first OCR request to take 30-90 seconds longer than normal.** Subsequent requests use the cached compilation and run at full speed.

The healthcheck `start_period` is set to 120 seconds to accommodate this.

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied: /dev/kfd` | Add host user to `video` and `render` groups, re-login |
| `No GPU provider found` | Verify ROCm driver: `rocm-smi` should show your GPU |
| Container exits immediately | Check `docker logs deepdoc-vietocr-rocm`; look for `/dev/kfd` or `MIGraphX` errors |
| `seccomp` errors | `security_opt: [seccomp=unconfined]` is already set in `docker-compose.rocm.yml` |
| Slow first request (>60s) | Normal — MIGraphX JIT compilation; wait for it to finish |
