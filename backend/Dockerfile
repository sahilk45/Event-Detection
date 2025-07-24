FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory for app
WORKDIR /home/app

# Copy backend requirements from correct path
COPY --chown=app:app backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend application
COPY --chown=app:app backend/ .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py"]
