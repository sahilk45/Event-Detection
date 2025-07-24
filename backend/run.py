import os
import uvicorn
from app.main import app

def main():
    """Run the FastAPI application with production settings."""
    # Get configuration from environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    # Production configuration
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for model consistency
        log_level="info",
        access_log=True,
        loop="uvloop",  # Better performance
        http="httptools"  # Better performance
    )

if __name__ == "__main__":
    main()
