# Dockerfile
FROM ubuntu:22.04

# FROM python:3.9-slim

# ── Install system deps ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
      curl tar ca-certificates python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ollama CLI + server ─────────────────────────────────
# RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Install Python deps ─────────────────────────────────────────
WORKDIR /app

RUN pip install --upgrade pip


COPY requirements.txt /app/requirements.txt
#RUN pip3 install --no-cache-dir -r /app/requirements.txt
# CPU-only install to save time/space.
RUN pip3 install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r /app/requirements.txt


# ── Copy application code & entrypoint ──────────────────────────
COPY . /app
RUN chmod +x entrypoint.sh

# ── Environment variables ───────────────────────────────────────
ENV OLLAMA_MODELS=/models
ENV PORT=5000

# ── Expose ports for Ollama (11434) and Flask (5000) ────────────
EXPOSE 11434 5000

# ── Launch both services via our entrypoint script ──────────────
ENTRYPOINT ["./entrypoint.sh"]
