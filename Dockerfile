# Use Python 3.10 as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements and install Python dependencies
COPY --chown=app:app requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Install the package
RUN pip install --user -e .

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Create output directory
RUN mkdir -p voice_clone

# Set the entry point
ENTRYPOINT ["python", "main.py"]
