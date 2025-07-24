# Use Python 3.12.2 specifically to match your local environment
FROM python:3.12.2-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r app && useradd -r -g app app

# Install system dependencies required for audio processing and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
# Since build context is backend/, requirements.txt should be in backend/
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
# This copies everything from backend/ directory
COPY --chown=app:app . .

# Create models directory if it doesn't exist
RUN mkdir -p models && chown -R app:app models

# Switch to non-root user
USER app

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Command to run the application
# Adjust this based on your FastAPI app structure in backend/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
