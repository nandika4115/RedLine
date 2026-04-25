FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — split for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# HuggingFace Spaces uses port 7860 by default
EXPOSE 7860

# Start Gradio dashboard (or swap for uvicorn app for API-only mode)
CMD ["python", "dashboard.py"]
