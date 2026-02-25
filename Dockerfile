FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/     ./app/
COPY scripts/ ./scripts/

# Directories that will be bind-mounted at runtime;
# create them here so the image works without mounts too.
RUN mkdir -p data models

EXPOSE 8000

# Entrypoint: if trained models are absent, generate the dataset and train
# before starting the server. This lets `docker-compose up` work out of the box.
CMD ["sh", "-c", "\
  if [ ! -f models/task_classifier.joblib ]; then \
    echo '==> ML models not found — generating dataset (needs ANTHROPIC_API_KEY)...' && \
    python scripts/generate_dataset.py && \
    echo '==> Training classifier...' && \
    python scripts/train_classifier.py; \
  fi && \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 \
"]
