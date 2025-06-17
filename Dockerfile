# Use Python 3.10 as base image
FROM python:3.10-slim-bullseye

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
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Set up model cache directories as volumes
ENV HF_HOME=/home/app/.cache/huggingface \
    TORCH_HOME=/home/app/.cache/torch \
    XDG_CACHE_HOME=/home/app/.cache

# Create cache directories
RUN mkdir -p /home/app/.cache/huggingface \
             /home/app/.cache/torch \
             /home/app/.cache/pip

# Copy only requirements first for better caching
COPY --chown=app:app requirements.txt .

# Install Python dependencies (this layer will be cached as long as requirements.txt doesn't change)
RUN pip install --user --cache-dir=/home/app/.cache/pip -r requirements.txt

# Clone Voice_Extractor dependency (cached as long as this RUN instruction doesn't change)
RUN git clone --depth 1 https://github.com/ReisCook/Voice_Extractor.git extern/Voice_Extractor

# Install Voice_Extractor dependencies (avoiding conflicts by using --no-deps for conflicting packages)
RUN pip install --user --cache-dir=/home/app/.cache/pip -r extern/Voice_Extractor/requirements.txt --no-deps || true && \
    pip install --user --cache-dir=/home/app/.cache/pip wespeaker@git+https://github.com/wenet-e2e/wespeaker.git

# Copy application source code (this will only invalidate if your source code changes)
COPY --chown=app:app setup.py pyproject.toml ./
COPY --chown=app:app src/ src/
COPY --chown=app:app main.py example.py ./

# Install the package
RUN pip install --user -e .

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Create output directory
RUN mkdir -p voice_clone

# Set the entry point
ENTRYPOINT ["python", "main.py"]
