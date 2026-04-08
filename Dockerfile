FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY . .
RUN uv pip install --system -e .

# organizer-mandated variables
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# project-internal variables
# Fix #8: SERVER_URL added to Dockerfile
ENV SERVER_URL=http://localhost:7860
ENV ENV_SEED=42
ENV PORT=7860

# HF_TOKEN is NOT set here — must be added as HuggingFace Space Secret

EXPOSE 7860

# HEALTHCHECK should be read-only and must not mutate episode state.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["python", "server.py"]

