# Hugging Face Space (Docker SDK) image for the SupportHR CV industry classifier.
# Build context = the staged Space folder produced by .github/workflows/deploy-hf-space.yml:
#   Dockerfile, README.md, requirements.txt, server.py, models/text_classifier_model.pkl
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    MODEL_PATH=/app/models/text_classifier_model.pkl \
    CLASSIFIER_ALLOW_SEED_FALLBACK=0

# HF Spaces run containers as UID 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user server.py .
COPY --chown=user models ./models

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).read()"

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers"]
