---
layout: default
title: Deployment Guide
description: Complete guide to deploying VIVTransformer in development, staging, and production.
permalink: /pages/deployment-guide/
---

# Deployment Guide

This page explains how to deploy VIVTransformer reliably, with a focus on simplicity and production readiness.

## Overview
- Goals: high availability, high throughput, secure access, and easy maintenance.
- Supported backends: PyTorch (eager/TorchScript), ONNX Runtime, TensorRT (optional).
- Environments: local dev, Docker containers, and cloud platforms.

## Environment
- Python >= 3.8, CPU with 4+ cores, 8GB+ RAM.
- Recommended packages: torch, numpy, fastapi, uvicorn, onnx, onnxruntime.

Quick setup:

```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-deployment.txt
```

## Model Export
Export to ONNX or TorchScript to improve portability and performance.

```python
# pseudo-code
model = load_model(checkpoint)
model.eval()
export_onnx(model, onnx_path, dynamic_axes={"input": {0: "batch", 1: "seq"}})
export_torchscript(model, ts_path)
```

## Containerized Deployment
Minimal Dockerfile:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements-deployment.txt .
RUN pip install --no-cache-dir -r requirements-deployment.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run container:

```bash
docker build -t vivtransformer:latest .
docker run -d --name vivt-api -p 8000:8000 vivtransformer:latest
```

## Health Check and Docs
- Health: http://localhost:8000/health
- API Docs (OpenAPI): http://localhost:8000/docs

## Useful Commands
- Logs: docker logs -f vivt-api
- Stop: docker stop vivt-api; Remove: docker rm vivt-api
- Restart: docker restart vivt-api

## Best Practices
- Enable monitoring (Prometheus metrics), structured logs, and request timeouts.
- Use mixed precision and dynamic batching for GPU inference when appropriate.
- Keep models and configs mounted read-only; never bake secrets into images.
