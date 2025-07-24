import numpy as np
from typing import Dict, Any, List, Tuple
import pickle
import os
import joblib
import functools


class AudioClassifier:
    def __init__(self, model_path: str, metadata_path: str = None):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.model_type = None
        self.load_model()
        
    def load_model(self):
        """Load model with automatic type detection."""
        try:
            # Load .keras or .pkl file
            if self.model_path.endswith('.pkl') or self.model_path.endswith('.keras'):
                if self.model_path.endswith('.keras'):
                    # Load TensorFlow native format
                    try:
                        import tensorflow as tf
                        print(f"Loading model from: {self.model_path}")
                        self.model = tf.keras.models.load_model(self.model_path)
                        self.model_type = "tensorflow"
                        print("✅ Model loaded successfully using TensorFlow native format")
                    except Exception as keras_error:
                        print(f"❌ Keras loading failed: {keras_error}")
                        raise RuntimeError(f"Failed to load .keras model: {keras_error}")
                        
                elif self.model_path.endswith('.pkl'):
                    # Original pickle loading code...
                    try:
                        print(f"Loading model from: {self.model_path}")
                        with open(self.model_path, 'rb') as f:
                            self.model = pickle.load(f)
                        
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
                            
                    except Exception as pkl_error:
                        print(f"❌ Pickle loading failed: {pkl_error}")
                        raise RuntimeError(f"Failed to load .pkl model: {pkl_error}")
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
                # Animals subcategories (0-3)
                0: 'Dog', 1: 'Cat', 2: 'Bird', 3: 'Other Animal',
                # Environment subcategories (4-7)
                4: 'Rain', 5: 'Wind', 6: 'Thunder', 7: 'Water',
                # Vehicle subcategories (8-11)
                8: 'Car', 9: 'Truck', 10: 'Motorcycle', 11: 'Aircraft',
                # Voice subcategories (12-15)
                12: 'Male Voice', 13: 'Female Voice', 14: 'Child Voice', 15: 'Crowd'
            },
            'input_shape': (40, 216, 1),  # Expected TensorFlow shape
            'sample_rate': 22050,
            'duration': 5,
            'n_mfcc': 40,
            'n_fft': 2048,
            'hop_length': 512
        }

    @functools.lru_cache(maxsize=1)
    def load_model_cached(model_path):
        # Cache model loading to prevent reload on every request
        return pickle.load(open(model_path, 'rb'))
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with smart alert system for high-confidence dangerous events."""
        if self.model is None:
            return {
                'predicted_category': 'unknown',
                'confidence': 0.0,
                'class_probabilities': {'unknown': 1.0},
                'error': 'Model not loaded'
            }
            
        try:
            if self.model_type == "tensorflow":
                # For TensorFlow models, ensure proper 4D shape: (batch, height, width, channels)
                if len(features.shape) == 3:  # (batch, n_mfcc, time_steps)
                    # Add channel dimension: (batch, n_mfcc, time_steps, 1)
                    features_4d = np.expand_dims(features, axis=-1)
                else:
                    features_4d = features
                
                print(f"TensorFlow input shape: {features_4d.shape}")
                
                # Make prediction
                predictions = self.model.predict(features_4d, verbose=0)
                print(f"Model predictions type: {type(predictions)}")
                print(f"Predictions shape: {[p.shape for p in predictions] if isinstance(predictions, list) else predictions.shape}")
                
            elif self.model_type == "sklearn":
                # For scikit-learn models, flatten the features
                if len(features.shape) > 2:
                    features_flat = features.reshape(features.shape[0], -1)
                else:
                    features_flat = features
                
                print(f"Sklearn input shape: {features_flat.shape}")
                
                # Make prediction
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(features_flat)
                else:
                    # Simple classifier without probabilities
                    predicted_class = self.model.predict(features_flat)
                    n_classes = len(self.metadata['category_mapping'])
                    predictions = np.zeros((1, n_classes))
                    predictions[0, predicted_class[0]] = 1.0
            
            # Handle multi-output predictions properly
            if isinstance(predictions, list) and len(predictions) >= 2:
                # Multi-output model with category and subcategory
                category_pred = predictions[0]
                subcategory_pred = predictions[1]
                print("Multi-output model detected - processing both category and subcategory")
            else:
                # Single output model
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
            class_probabilities = {
                self.metadata['category_mapping'].get(i, f"class_{i}"): float(prob)
                for i, prob in enumerate(category_pred[0])
            }
            
            result = {
                'predicted_category': predicted_category,
                'confidence': category_confidence,
                'class_probabilities': class_probabilities,
                'model_type': self.model_type
            }
            
            # Process subcategory if available
            predicted_subcategory = None
            subcategory_confidence = 0.0
            
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
            
            # SMART ALERT SYSTEM - Based on your AISOC project requirements
            alert_triggered = self._check_alert_conditions(
                predicted_category, 
                predicted_subcategory, 
                category_confidence, 
                subcategory_confidence
            )
            
            result['alert_info'] = alert_triggered
            
            # Log the prediction result
            confidence_to_use = subcategory_confidence if subcategory_confidence > 0 else category_confidence
            event_name = predicted_subcategory if predicted_subcategory else predicted_category
            
            print(f"Prediction: {predicted_category}" + 
                  (f" -> {predicted_subcategory}" if predicted_subcategory else "") + 
                  f" (conf: {confidence_to_use:.3f})")
            
            if alert_triggered['should_alert']:
                print(f"🚨 ALERT TRIGGERED: {event_name} detected with {confidence_to_use:.1%} confidence!")
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'predicted_category': 'error',
                'confidence': 0.0,
                'class_probabilities': {'error': 1.0},
                'error': error_msg,
                'model_type': self.model_type
            }

    def _check_alert_conditions(self, category: str, subcategory: str, 
                               cat_confidence: float, sub_confidence: float) -> Dict[str, Any]:
        """
        Smart alert system based on AISOC project specifications.
        Triggers alerts for dangerous/important events with >70% confidence.
        """
        # Define alert-worthy events from your AISOC project
        ALERT_EVENTS = {
            # Emergency sounds
            'alarm': {'priority': 'high', 'type': 'emergency'},
            'siren': {'priority': 'high', 'type': 'emergency'},
            'fire alarm': {'priority': 'critical', 'type': 'emergency'},
            'smoke alarm': {'priority': 'critical', 'type': 'emergency'},
            
            # Safety events
            'glass breaking': {'priority': 'high', 'type': 'safety'},
            'baby cry': {'priority': 'medium', 'type': 'safety'},
            'baby crying': {'priority': 'medium', 'type': 'safety'},
            'scream': {'priority': 'high', 'type': 'safety'},
            'help': {'priority': 'critical', 'type': 'safety'},
            
            # Security events  
            'dog bark': {'priority': 'medium', 'type': 'security'},
            'dog barking': {'priority': 'medium', 'type': 'security'},
            'dog': {'priority': 'medium', 'type': 'security'},
            'door break': {'priority': 'high', 'type': 'security'},
            'footsteps': {'priority': 'low', 'type': 'security'},
            
            # Vehicle emergencies
            'car alarm': {'priority': 'medium', 'type': 'vehicle'},
            'horn': {'priority': 'low', 'type': 'vehicle'},
            'crash': {'priority': 'high', 'type': 'vehicle'},
            'truck': {'priority': 'low', 'type': 'vehicle'},
        }
        
        # Determine which event to check (prefer subcategory if available)
        primary_event = subcategory.lower() if subcategory else category.lower()
        primary_confidence = sub_confidence if sub_confidence > 0 else cat_confidence
        
        # Check if this event should trigger an alert
        should_alert = False
        alert_info = {
            'should_alert': False,
            'event_name': primary_event,
            'confidence': primary_confidence,
            'priority': 'none',
            'alert_type': 'none',
            'message': 'Listening...',
            'timestamp': None
        }
        
        # Alert logic: >70% confidence for alert-worthy events
        if primary_confidence > 0.70:  # 70% threshold from your requirements
            event_key = None
            
            # Find matching alert event (exact match or partial match)
            for alert_event, info in ALERT_EVENTS.items():
                if alert_event in primary_event or primary_event in alert_event:
                    event_key = alert_event
                    break
            
            if event_key:
                should_alert = True
                alert_info.update({
                    'should_alert': True,
                    'priority': ALERT_EVENTS[event_key]['priority'],
                    'alert_type': ALERT_EVENTS[event_key]['type'],
                    'message': f"{primary_event.title()} detected with {primary_confidence:.1%} confidence!",
                    'timestamp': self._get_current_timestamp(),
                    'recommended_action': self._get_recommended_action(event_key, ALERT_EVENTS[event_key])
                })
        
        return alert_info

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _get_recommended_action(self, event: str, event_info: Dict) -> str:
        """Get recommended action based on detected event."""
        action_map = {
            'critical': "🆘 IMMEDIATE ACTION REQUIRED - Contact emergency services if needed",
            'high': "⚠️ HIGH PRIORITY - Check surroundings and ensure safety",
            'medium': "🔔 ATTENTION - Monitor situation and take appropriate action",
            'low': "ℹ️ NOTICE - Informational alert, no immediate action needed"
        }
        return action_map.get(event_info['priority'], "Monitor situation")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        model_info = {
            'model_type': self.model_type or "Not loaded",
            'backend': self.model_type,
            'metadata': self.metadata,
            'model_loaded': self.model is not None,
            'file_path': self.model_path
        }
        
        try:
            if self.model and hasattr(self.model, 'input_shape'):
                model_info['input_shape'] = str(self.model.input_shape)
            elif self.model_type == "tensorflow":
                model_info['input_shape'] = "(None, 40, 216, 1)"
            elif self.model_type == "sklearn":
                model_info['input_shape'] = "Flattened MFCC features"
        except Exception:
            pass
            
        return model_info
