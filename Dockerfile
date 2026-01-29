# Dockerfile for Hugging Face Spaces / container deployment
FROM python:3.11-slim

# Install system dependencies for audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project (root requirements + src)
COPY requirements.txt .
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run from src so that app:app and engine imports resolve
WORKDIR /app/src

EXPOSE 7860

# Start the FastAPI app (PORT set by HF Spaces or default 7860)
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
