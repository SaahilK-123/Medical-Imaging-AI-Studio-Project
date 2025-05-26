from tensorflow.keras.models import load_model
import os

MODEL_PATH = "MedAI_Results.keras"

def get_model(custom_objects=None):
    if os.path.exists(MODEL_PATH):
        print("🔄 Loading existing model...")
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
    else:
        print("❌ Model file not found. Please train the model first or check the path.")
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    
    return model
