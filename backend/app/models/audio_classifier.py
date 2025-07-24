import numpy as np
from typing import Dict, Any, List, Tuple
import pickle
import os
import joblib
import functools
import gc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AudioClassifier:
    def __init__(self, model_path: str, metadata_path: str = None):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.model_type = None
        self.load_model()
        
    def load_model(self):
        """Load model with Keras compatibility fixes and memory optimization."""
        try:
        
            # Reduce TensorFlow memory usage
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
            # Clean memory before loading
            gc.collect()
            
            # Import TensorFlow and setup Keras compatibility
            try:
                import tensorflow as tf
                import keras
                
                # Keras compatibility fixes for model loading
                self._setup_keras_compatibility()
                
                # Configure TensorFlow for memory efficiency
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                
            except ImportError as e:
                print(f"Warning: TensorFlow import failed: {e}")
            
            # Load .pkl file with Keras compatibility
            if self.model_path.endswith('.pkl'):
                try:
                    print(f"Loading model from: {self.model_path}")
                    
                    # Enhanced model loading with error handling
                    with open(self.model_path, 'rb') as f:
                        try:
                            self.model = pickle.load(f)
                        except Exception as load_error:
                            print(f"❌ Direct pickle load failed: {load_error}")
                            # Try alternative loading method
                            f.seek(0)
                            self.model = self._load_model_with_compatibility(f)
                    
                    # Detect model type based on attributes
                    if hasattr(self.model, 'predict') and hasattr(self.model, 'layers'):
                        self.model_type = "tensorflow"
                        print("✅ Detected TensorFlow/Keras model in .pkl file")
                    elif hasattr(self.model, 'predict_proba'):
                        self.model_type = "sklearn"
                        print("✅ Detected scikit-learn model in .pkl file")
                    else:
                        self.model_type = "unknown"
                        print("⚠️ Unknown model type detected")
                        
                    # Clean up after model loading
                    gc.collect()
                        
                except Exception as pkl_error:
                    print(f"❌ Pickle loading failed: {pkl_error}")
                    print("⚠️ Model loading failed, using fallback initialization")
                    self.model = None
                    self.model_type = "error"
            else:
                raise RuntimeError(f"Unsupported model format: {self.model_path}")
                
            print(f"Model type: {self.model_type}")
            
            # Load metadata if available
            if self.metadata_path and os.path.exists(self.metadata_path):
                try:
                    self.metadata = np.load(self.metadata_path, allow_pickle=True).item()
                    print("✅ Metadata loaded successfully")
                    
                    # DEBUG CODE - Shows actual metadata content
                    if self.metadata:
                        print("=== METADATA DEBUG ===")
                        print(f"Category mapping: {self.metadata.get('category_mapping', 'Not found')}")
                        print(f"Subcategory mapping: {self.metadata.get('subcategory_mapping', 'Not found')}")
                        print("=====================")
                        
                except Exception as metadata_error:
                    print(f"Warning: Could not load metadata: {metadata_error}")
                    self.metadata = self._create_default_metadata()
            else:
                self.metadata = self._create_default_metadata()
                print("Using default metadata")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize model: {str(e)}")
            self.model = None
            self.metadata = self._create_default_metadata()
    
    def _setup_keras_compatibility(self):
        """Setup Keras compatibility for different versions."""
        try:
            import tensorflow as tf
            import keras
            
            # Ensure keras modules are available in expected locations
            import sys
            
            # Create compatibility mappings for keras.src modules
            if 'keras.src' not in sys.modules:
                try:
                    import keras.src
                except ImportError:
                    # Create a dummy keras.src module structure
                    class DummyModule:
                        pass
                    
                    keras.src = DummyModule()
                    sys.modules['keras.src'] = keras.src
                    
                    # Add common submodules
                    keras.src.models = DummyModule()
                    keras.src.models.functional = tf.keras.models.Model
                    sys.modules['keras.src.models'] = keras.src.models
                    sys.modules['keras.src.models.functional'] = keras.src.models.functional
                    
                    keras.src.layers = tf.keras.layers
                    sys.modules['keras.src.layers'] = keras.src.layers
                    
                    print("✅ Keras compatibility layer setup complete")
                    
        except Exception as e:
            print(f"Warning: Keras compatibility setup failed: {e}")
    
    def _load_model_with_compatibility(self, file_handle):
        """Alternative model loading with enhanced compatibility."""
        try:
            import tensorflow as tf
            
            # Try loading with different unpicklers
            file_handle.seek(0)
            
            # Method 1: Standard pickle
            try:
                return pickle.load(file_handle)
            except Exception as e1:
                print(f"Standard pickle failed: {e1}")
                
                # Method 2: With custom object scope
                file_handle.seek(0)
                try:
                    with tf.keras.utils.custom_object_scope({}):
                        return pickle.load(file_handle)
                except Exception as e2:
                    print(f"Custom object scope failed: {e2}")
                    
                    # Method 3: Load as keras model directly
                    file_handle.seek(0)
                    try:
                        # If it's a saved keras model in pickle format
                        model_data = pickle.load(file_handle)
                        if hasattr(model_data, 'load_weights'):
                            return model_data
                    except Exception as e3:
                        print(f"Direct keras load failed: {e3}")
                        raise e1  # Return original error
                        
        except Exception as e:
            print(f"All loading methods failed: {e}")
            raise e
    
    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default metadata matching your actual categories."""
        return {
            'category_mapping': {
                0: 'Animals',
                1: 'Environment', 
                2: 'Vehicles',
                3: 'Voice'
            },
            'subcategory_mapping': {
                0: 'bike', 1: 'bus', 2: 'car', 3: 'cat', 4: 'crowd', 
                5: 'dog', 6: 'elephant', 7: 'horse', 8: 'lion', 9: 'person_voice', 
                10: 'rainfall', 11: 'siren', 12: 'traffic', 13: 'truck'
            },
            'input_shape': (40, 216, 1),
            'sample_rate': 22050,
            'duration': 5,
            'n_mfcc': 40,
            'n_fft': 2048,
            'hop_length': 512
        }

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with enhanced error handling."""
        if self.model is None:
            return {
                'predicted_category': 'unknown',
                'confidence': 0.0,
                'class_probabilities': {'unknown': 1.0},
                'error': 'Model not loaded - using default predictions',
                'alert_info': {
                    'should_alert': False,
                    'message': 'Model not available'
                }
            }
            
        try:
            # Memory optimization: Clean up before prediction
            gc.collect()
            
            if self.model_type == "tensorflow":
                # For TensorFlow models, ensure proper 4D shape
                if len(features.shape) == 3:
                    features_4d = np.expand_dims(features, axis=-1)
                else:
                    features_4d = features
                
                print(f"TensorFlow input shape: {features_4d.shape}")
                
                # Make prediction with error handling
                try:
                    predictions = self.model.predict(features_4d, verbose=0, batch_size=1)
                    print(f"Model predictions type: {type(predictions)}")
                    print(f"Predictions shape: {[p.shape for p in predictions] if isinstance(predictions, list) else predictions.shape}")
                except Exception as pred_error:
                    print(f"Model prediction failed: {pred_error}")
                    # Return fallback prediction
                    return self._get_fallback_prediction()
                finally:
                    del features_4d
                    gc.collect()
                
            else:
                print(f"Unsupported model type for prediction: {self.model_type}")
                return self._get_fallback_prediction()
            
            # Process predictions (same as your original code)
            if isinstance(predictions, list) and len(predictions) >= 2:
                category_pred = predictions[0]
                subcategory_pred = predictions[1]
                print("Multi-output model detected")
            else:
                category_pred = predictions if not isinstance(predictions, list) else predictions[0]
                subcategory_pred = None
                print("Single output model detected")
            
            # Process main category
            predicted_category_idx = np.argmax(category_pred[0])
            predicted_category = self.metadata['category_mapping'].get(
                predicted_category_idx, f"class_{predicted_category_idx}"
            )
            category_confidence = float(np.max(category_pred[0]))
            
            # Get all class probabilities
            class_probabilities = {}
            for i, prob in enumerate(category_pred[0]):
                category_name = self.metadata['category_mapping'].get(i, f"class_{i}")
                class_probabilities[category_name] = float(prob)
            
            result = {
                'predicted_category': predicted_category,
                'confidence': category_confidence,
                'class_probabilities': class_probabilities,
                'model_type': self.model_type
            }
            
            # Process subcategory if available
            if subcategory_pred is not None:
                predicted_subcategory_idx = np.argmax(subcategory_pred[0])
                predicted_subcategory = self.metadata['subcategory_mapping'].get(
                    predicted_subcategory_idx, f"subclass_{predicted_subcategory_idx}"
                )
                subcategory_confidence = float(np.max(subcategory_pred[0]))
                
                result.update({
                    'predicted_subcategory': predicted_subcategory,
                    'subcategory_confidence': subcategory_confidence
                })
            else:
                predicted_subcategory = None
                subcategory_confidence = 0.0
            
            # Alert system
            alert_triggered = self._check_alert_conditions(
                predicted_category, 
                predicted_subcategory, 
                category_confidence, 
                subcategory_confidence
            )
            
            result['alert_info'] = alert_triggered
            
            # Logging
            confidence_to_use = subcategory_confidence if subcategory_confidence > 0 else category_confidence
            event_name = predicted_subcategory if predicted_subcategory else predicted_category
            
            print(f"Prediction: {predicted_category}" + 
                  (f" -> {predicted_subcategory}" if predicted_subcategory else "") + 
                  f" (conf: {confidence_to_use:.3f})")
            
            if alert_triggered['should_alert']:
                print(f"🚨 ALERT TRIGGERED: {event_name} detected with {confidence_to_use:.1%} confidence!")
            
            # Cleanup
            del predictions
            gc.collect()
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            print(f"❌ {error_msg}")
            gc.collect()
            return self._get_fallback_prediction()
    
    def _get_fallback_prediction(self):
        """Return a safe fallback prediction when model fails."""
        return {
            'predicted_category': 'unknown',
            'confidence': 0.5,
            'class_probabilities': {
                'Animals': 0.25,
                'Environment': 0.25, 
                'Vehicles': 0.25,
                'Voice': 0.25
            },
            'model_type': 'fallback',
            'predicted_subcategory': 'unknown',
            'subcategory_confidence': 0.5,
            'alert_info': {
                'should_alert': False,
                'event_name': 'unknown',
                'confidence': 0.5,
                'priority': 'none',
                'alert_type': 'none',
                'message': 'Model unavailable - using fallback detection',
                'timestamp': self._get_current_timestamp()
            }
        }

    # Keep all your existing methods: _check_alert_conditions, _get_current_timestamp, etc.
    def _check_alert_conditions(self, category: str, subcategory: str, 
                               cat_confidence: float, sub_confidence: float) -> Dict[str, Any]:
        """Smart alert system based on AISOC project specifications."""
        ALERT_EVENTS = {
            'alarm': {'priority': 'high', 'type': 'emergency'},
            'siren': {'priority': 'high', 'type': 'emergency'},
            'dog bark': {'priority': 'medium', 'type': 'security'},
            'dog': {'priority': 'medium', 'type': 'security'},
            'baby cry': {'priority': 'medium', 'type': 'safety'},
        }
        
        primary_event = subcategory.lower() if subcategory else category.lower()
        primary_confidence = sub_confidence if sub_confidence > 0 else cat_confidence
        
        alert_info = {
            'should_alert': False,
            'event_name': primary_event,
            'confidence': primary_confidence,
            'priority': 'none',
            'alert_type': 'none',
            'message': 'Listening...',
            'timestamp': None
        }
        
        if primary_confidence > 0.70:
            for alert_event, info in ALERT_EVENTS.items():
                if alert_event in primary_event or primary_event in alert_event:
                    alert_info.update({
                        'should_alert': True,
                        'priority': info['priority'],
                        'alert_type': info['type'],
                        'message': f"{primary_event.title()} detected with {primary_confidence:.1%} confidence!",
                        'timestamp': self._get_current_timestamp(),
                        'recommended_action': self._get_recommended_action(alert_event, info)
                    })
                    break
        
        return alert_info

    def _get_current_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def _get_recommended_action(self, event: str, event_info: Dict) -> str:
        action_map = {
            'critical': "🆘 IMMEDIATE ACTION REQUIRED",
            'high': "⚠️ HIGH PRIORITY - Check surroundings",
            'medium': "🔔 ATTENTION - Monitor situation",
            'low': "ℹ️ NOTICE - Informational alert"
        }
        return action_map.get(event_info['priority'], "Monitor situation")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type or "Not loaded",
            'backend': self.model_type,
            'metadata': self.metadata,
            'model_loaded': self.model is not None,
            'file_path': self.model_path,
            'input_shape': "(None, 40, 216, 1)" if self.model_type == "tensorflow" else "Unknown"
        }

    def __del__(self):
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            gc.collect()
        except:
            pass
