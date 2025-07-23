import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Production configuration
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for model consistency
        log_level="info"
    )
