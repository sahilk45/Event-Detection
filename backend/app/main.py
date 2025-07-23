from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os

app = FastAPI(
    title="Acoustic Event Detection API",
    description="AI-powered audio analysis with smart alerts",
    version="1.0.0"
)

# Production CORS configuration
origins = [
    "http://localhost:3000",  # Local development
    "https://*.onrender.com",  # All Render domains
    "https://acoustic-event-detection.onrender.com",  # Your frontend URL (update after deployment)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Acoustic Event Detection API",
        "status": "healthy",
        "version": "1.0.0",
        "features": ["file_upload", "live_recording", "smart_alerts"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "acoustic-event-detection"}
