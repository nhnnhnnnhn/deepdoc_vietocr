ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv

COPY requirements-gpu.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements-gpu.txt

COPY . .
RUN chmod +x docker/entrypoint.sh \
    && mkdir -p input output web_runs log \
    && test -f conf/ai_postcheck_system_prompt.txt \
    && test -f onnx/det.onnx \
    && test -f onnx/layout.onnx \
    && test -f onnx/tsr.onnx \
    && test -f vietocr/weight/vgg_seq2seq.pth \
    && python -m py_compile web_app.py full_pipeline.py

EXPOSE 8008

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8008"]
