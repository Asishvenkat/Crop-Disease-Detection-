"""
Quick test script to verify tomato model predictions
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load model
MODEL_PATH = Path("models/tomato_model.h5")
model = keras.models.load_model(str(MODEL_PATH))

# Load class mapping
with open("models/class_mapping_tomato.json") as f:
    class_mapping = json.load(f)

print("=" * 70)
print("🍅 TOMATO MODEL TEST")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Classes: {class_mapping}")
print(f"Model shape: {model.input_shape}")
print()

# Get a test image from early blight class
data_path = Path("data/plantvillage dataset/color/Tomato___Early_blight")
test_images = list(data_path.glob("*.JPG"))[:3]

print(f"Testing with {len(test_images)} Early Blight images:")
print("-" * 70)

for img_path in test_images:
    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    img = img.resize((160, 160), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    
    # Predict
    batch = np.expand_dims(img_array, axis=0)
    logits = model.predict(batch, verbose=0)
    probs = logits[0]
    
    pred_idx = np.argmax(probs)
    pred_class = class_mapping[str(pred_idx)]
    confidence = probs[pred_idx]
    
    # Show top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    
    print(f"\n📷 Image: {img_path.name}")
    print(f"   Predicted: {pred_class} ({confidence*100:.1f}%)")
    print("   Top 3 predictions:")
    for idx in top3_idx:
        print(f"      {class_mapping[str(idx)]}: {probs[idx]*100:.1f}%")

print("\n" + "=" * 70)
