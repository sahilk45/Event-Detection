from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api.routes import router
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Acoustic Event Detection API",
    description="AI-powered audio analysis with smart alerts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS configuration for production
origins = [
    "http://localhost:3000",  # Local development
    "https://*.onrender.com",  # Render domains
    "https://your-frontend-name.onrender.com",  # Your deployed frontend URL
    "https://*.onrender.com",  # Allow all Render subdomains
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "Acoustic Event Detection API",
        "status": "healthy",
        "version": "1.0.0",
        "features": ["file_upload", "live_recording", "smart_alerts"],
        "environment": "docker"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "acoustic-event-detection",
        "environment": "docker"
    }

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("🚀 Acoustic Event Detection API starting up...")
    logger.info("🐳 Running in Docker environment")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("🛑 Acoustic Event Detection API shutting down...")
