#!/bin/bash

# Render build script for Docker deployment
set -o errexit

echo "🐳 Building Docker image for Acoustic Event Detection API..."

# Build the Docker image
docker build -t acoustic-detection-api .

echo "✅ Docker build completed successfully!"
