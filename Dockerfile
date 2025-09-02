# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

WORKDIR /app

# System deps for faiss/pymupdf and friends
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    git libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /app

# Create data directory and set as volume for persistence
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 7860

CMD ["python", "app.py"]


