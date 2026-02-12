"""
Test script to evaluate corn disease detection model accuracy on validation set
"""

import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    # Set up paths
    model_path = r"E:\Crop Disease Prediction\models\pdd_corn.keras"
    data_path = r"E:\Crop Disease Prediction\data\plantvillage dataset\color"
    class_mapping_path = r"E:\Crop Disease Prediction\models\class_mapping_corn.json"
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Convert string keys to integers
    class_mapping_int = {int(k): v for k, v in class_mapping.items()}
    reverse_mapping = {int(k): v for k, v in class_mapping.items()}
    num_classes = len(class_mapping)
    
    # Load model
    print("🔄 Loading model...")
    model = keras.models.load_model(model_path)
    
    # Prepare validation data
    print("📁 Loading validation data...")
    
    # Find corn disease folders
    corn_classes_dirs = [d for d in os.listdir(data_path) if d.startswith("Corn_")]
    
    # Map directory names to class indices
    class_dir_map = {}
    for class_name, class_idx in class_mapping_int.items():
        class_dir_map[class_name] = class_idx
    
    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    total_samples = 0
    correct_predictions = 0
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    # Process each class
    for class_idx in range(num_classes):
        class_name = reverse_mapping[class_idx]
        
        # Find matching directory
        class_dir = None
        for d in os.listdir(data_path):
            if d.startswith("Corn_") and class_name in d:
                class_dir = os.path.join(data_path, d)
                break
        
        if class_dir is None or not os.path.isdir(class_dir):
            print(f"⚠️  Could not find directory for class {class_idx}: {class_name}")
            continue
        
        print(f"\n📊 Processing class {class_idx}: {class_name}")
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   Found {len(image_files)} images")
        
        class_predictions = {i: 0 for i in range(num_classes)}
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and preprocess image
                img = keras.preprocessing.image.load_img(
                    img_path, target_size=(192, 192)
                )
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                
                # Make prediction
                prediction = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(prediction[0])
                
                # Update confusion matrix
                confusion_matrix[class_idx, predicted_class] += 1
                
                # Track accuracy
                if predicted_class == class_idx:
                    correct_predictions += 1
                    class_correct[class_idx] += 1
                
                class_total[class_idx] += 1
                total_samples += 1
                class_predictions[predicted_class] += 1
                
            except Exception as e:
                print(f"   Error processing {img_file}: {str(e)}")
                continue
        
        if class_total[class_idx] > 0:
            class_acc = class_correct[class_idx] / class_total[class_idx] * 100
            print(f"   ✓ Class accuracy: {class_acc:.2f}%")
            print(f"   Predictions: {class_predictions}")
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_samples * 100 if total_samples > 0 else 0
    
    print("\n" + "="*60)
    print("📊 OVERALL VALIDATION RESULTS")
    print("="*60)
    print(f"Total samples tested: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Correct predictions: {correct_predictions}/{total_samples}")
    
    print("\n📈 PER-CLASS ACCURACY:")
    print("-"*60)
    for class_idx in range(num_classes):
        class_name = reverse_mapping[class_idx]
        if class_total[class_idx] > 0:
            acc = class_correct[class_idx] / class_total[class_idx] * 100
            print(f"{class_name:45} {acc:6.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
    
    print("\n🔢 CONFUSION MATRIX:")
    print("-"*60)
    print("Predicted → (columns)")
    print("Actual ↓ (rows)")
    
    # Print header
    print("\n" + " " * 45, end="")
    for i in range(num_classes):
        print(f"  {i} ", end="")
    print()
    
    # Print rows
    for i in range(num_classes):
        class_name = reverse_mapping[i]
        print(f"{class_name:45}", end="")
        for j in range(num_classes):
            print(f" {int(confusion_matrix[i, j]):3.0f}", end="")
        print()
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
