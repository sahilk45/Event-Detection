import librosa
import numpy as np
from typing import Dict, Any, Tuple

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050, duration: int = 5):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = 40
        self.n_fft = 2048
        self.hop_length = 512
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f"Audio loaded: {len(audio)} samples at {sr}Hz")
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio to fixed length (exactly like teammate's approach)."""
        target_length = self.sample_rate * self.duration
        
        if len(audio) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant', constant_values=0)
        elif len(audio) > target_length:
            # Trim if too long
            audio = audio[:target_length]
            
        print(f"Audio preprocessed to length: {len(audio)}")
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features exactly like your teammate's code."""
        try:
            # Extract MFCCs with exact same parameters
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Normalize MFCCs (matching teammate's normalization)
            mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
            
            print(f"MFCC shape: {mfccs.shape}")
            return mfccs
            
        except Exception as e:
            raise ValueError(f"Error extracting MFCC features: {str(e)}")
    
    def process_for_model(self, audio: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline matching your teammate's approach."""
        # Preprocess audio length
        audio = self.preprocess_audio(audio)
        
        # Extract MFCC features
        mfccs = self.extract_mfcc(audio)
        
        # Add batch dimension for consistency
        mfccs = np.expand_dims(mfccs, axis=0)  # Shape: (1, n_mfcc, time_steps)
        
        print(f"Final feature shape: {mfccs.shape}")
        return mfccs
