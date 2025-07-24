import pickle
import tensorflow as tf
import os

# Make sure models directory exists
os.makedirs('models', exist_ok=True)

print("Loading pickle model...")
try:
    with open('models/model1.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("✅ Pickle model loaded successfully!")
    print(f"Model type: {type(model)}")
    
    # Save in TensorFlow native format
    print("Converting to TensorFlow native format...")
    model.save('models/model1.keras')
    
    print("✅ Model converted successfully!")
    print("File saved as: models/model1.keras")
    
    # Test loading the new format
    print("Testing new format...")
    test_model = tf.keras.models.load_model('models/model1.keras')
    print("✅ New format loads correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
