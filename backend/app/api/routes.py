from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import Dict, Any
from datetime import datetime

from app.models.audio_classifier import AudioClassifier
from app.utils.audio_processing import AudioProcessor

router = APIRouter()

# Updated paths to use the .pkl file
MODEL_PATH = "models/model1.pkl"  # Changed from audio_processing.h5
METADATA_PATH = "models/metadata.npy"

try:
    classifier = AudioClassifier(MODEL_PATH, METADATA_PATH)
    processor = AudioProcessor()
    print("Model and processor initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize model: {e}")
    classifier = None
    processor = None

# Rest of your routes.py code remains the same...
@router.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Predict audio class with live chunk support."""
    if not classifier or not processor:
        raise HTTPException(
            status_code=500, 
            detail="Model not initialized. Please check model files."
        )
    
    # Detect live recording chunks
    is_live_chunk = file.filename and file.filename.startswith('chunk_')
    print(f"Processing {'live chunk' if is_live_chunk else 'uploaded file'}: {file.filename}")
    
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400,
            detail="Please upload an audio file"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Special handling for live chunks
            if is_live_chunk:
                # Validate chunk size
                if len(content) < 500:  # Very small chunk
                    print(f"⚠️ Skipping tiny chunk: {len(content)} bytes")
                    return JSONResponse(content={
                        'predicted_category': 'silence',
                        'confidence': 0.0,
                        'class_probabilities': {'silence': 1.0},
                        'filename': file.filename,
                        'file_size': len(content),
                        'processed': True,
                        'is_live_chunk': True,
                        'alert_info': {
                            'should_alert': False,
                            'message': 'Chunk too small for analysis'
                        }
                    })
            
            # Process audio with enhanced error handling
            try:
                audio, sr = processor.load_audio(tmp_path)
                features = processor.process_for_model(audio)
                
                # Make prediction
                result = classifier.predict(features)
                
            except Exception as processing_error:
                print(f"❌ Processing error for {file.filename}: {str(processing_error)}")
                
                # For live chunks, return graceful fallback instead of 500 error
                if is_live_chunk:
                    return JSONResponse(content={
                        'predicted_category': 'error',
                        'confidence': 0.0,
                        'class_probabilities': {'error': 1.0},
                        'filename': file.filename,
                        'file_size': len(content),
                        'processed': False,
                        'is_live_chunk': True,
                        'error': 'Processing failed',
                        'alert_info': {
                            'should_alert': False,
                            'message': 'Audio processing error'
                        }
                    })
                else:
                    # For uploaded files, still raise proper error
                    raise HTTPException(
                        status_code=500,
                        detail=f"Audio processing failed: {str(processing_error)}"
                    )
            
            # Add file information to successful results
            result.update({
                'filename': file.filename,
                'file_size': len(content),
                'processed': True,
                'is_live_chunk': is_live_chunk,
                'processing_timestamp': datetime.now().isoformat()
            })
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        
        # For live chunks, return graceful error response
        if is_live_chunk:
            return JSONResponse(
                status_code=200,  # Don't return 500 for live chunks
                content={
                    'predicted_category': 'error',
                    'confidence': 0.0,
                    'class_probabilities': {'error': 1.0},
                    'filename': file.filename if file else 'unknown',
                    'processed': False,
                    'is_live_chunk': True,
                    'error': 'Unexpected processing error',
                    'alert_info': {
                        'should_alert': False,
                        'message': 'System error during processing'
                    }
                }
            )
        else:
            # For uploaded files, maintain proper error handling
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio: {str(e)}"
            )


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if not classifier:
        raise HTTPException(
            status_code=500,
            detail="Model not initialized"
        )
    
    try:
        info = classifier.get_model_info()
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats."""
    return {
        "supported_formats": [
            ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"
        ],
        "recommended_format": ".wav",
        "max_file_size_mb": 10,
        "duration_seconds": 5
    }
