"""
Strawberry Disease Detection - Specialized Training
Trains on strawberry disease classes with optimized settings
Target: >90% accuracy in <30 minutes
"""

import os
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# Set encoding for terminal output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Enable mixed precision for faster training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("="*60)
print("STRAWBERRY DISEASE DETECTION - SPECIALIZED TRAINING")
print("="*60)

# Configuration
DATA_PATH = r"E:\Crop Disease Prediction\data\plantvillage dataset\color"
MODEL_OUTPUT_PATH = r"E:\Crop Disease Prediction\models\pdd_strawberry.keras"
CLASS_MAPPING_PATH = r"E:\Crop Disease Prediction\models\class_mapping_strawberry.json"
TREATMENTS_PATH = r"E:\Crop Disease Prediction\models\treatments_strawberry.json"

IMAGE_SIZE = 192
BATCH_SIZE = 32
EPOCHS_WARMUP = 3
EPOCHS_FINETUNE = 9
TOTAL_EPOCHS = EPOCHS_WARMUP + EPOCHS_FINETUNE
LEARNING_RATE_WARMUP = 1e-4
LEARNING_RATE_FINETUNE = 1e-5

# Define strawberry disease classes
STRAWBERRY_CLASSES = {
    'Strawberry___Leaf_scorch': {
        'index': 0,
        'disease_name': 'Leaf Scorch',
        'scientific_name': 'Diplocarpon earliana',
        'treatment': 'Fungicide application (Captan or Sulfur-based). Remove infected leaves. Improve air circulation. Ensure proper watering (avoid wetting foliage).',
        'severity': 'Moderate',
        'prevention': 'Use disease-resistant varieties. Apply preventative fungicides in spring. Maintain proper plant spacing.'
    },
    'Strawberry___healthy': {
        'index': 1,
        'disease_name': 'Healthy',
        'scientific_name': 'Healthy Plant',
        'treatment': 'No treatment needed. Continue regular maintenance.',
        'severity': 'None',
        'prevention': 'Maintain good cultural practices.'
    }
}

# Custom TimeLimit callback to enforce strict training time budget
class TimeLimit(keras.callbacks.Callback):
    def __init__(self, max_seconds=1680):  # 28 minutes
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = self.max_seconds - elapsed
        print(f"\nTime elapsed: {elapsed:.0f}s / {self.max_seconds}s ({remaining:.0f}s remaining)")
        if elapsed > self.max_seconds:
            print(f"TIME LIMIT EXCEEDED! Stopping training.")
            self.model.stop_training = True

# Build MobileNetV2 model
print("\nPackage: Building MobileNetV2 model for 2 classes...")
base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Create custom head
inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(2, activation='softmax', dtype='float32')(x)

model = keras.Model(inputs, outputs)
total_params = model.count_params()
print(f"[OK] Model created with {total_params:,} total parameters")

# Data augmentation
print("\nSetting up data augmentation...")
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.75, 1.25],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
print("[OK] Data augmentation configured")

# Load training data
print(f"\nLoading dataset from {DATA_PATH}...")

# Map class names to indices
class_map = {}
for class_name, class_info in STRAWBERRY_CLASSES.items():
    class_map[class_name] = class_info['index']

train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=class_map
)

validation_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=class_map
)

print(f"[OK] Training samples: {train_generator.samples}")
print(f"[OK] Validation samples: {validation_generator.samples}")
print(f"[OK] Classes: {train_generator.class_indices}")

# Training strategy: Warmup + Fine-tune
print(f"\nTraining for {TOTAL_EPOCHS} epochs (warmup {EPOCHS_WARMUP} + fine-tune {EPOCHS_FINETUNE})...")

# Warmup phase: train with frozen base
print("\nWARMUP PHASE (frozen base layers)...")
base_model.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_WARMUP),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_warmup = model.fit(
    train_generator,
    epochs=EPOCHS_WARMUP,
    validation_data=validation_generator,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        TimeLimit(max_seconds=1680)
    ]
)

# Fine-tune phase: unfreeze some base layers
print("\nFINE-TUNE PHASE (unfreezing selective layers)...")
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINETUNE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_WARMUP,
    validation_data=validation_generator,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        TimeLimit(max_seconds=1680)
    ]
)

# Evaluate on validation set
print("\nEvaluating model on validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"[OK] Validation Loss: {val_loss:.4f}")
print(f"[OK] Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Save class mapping
print(f"\nSaving class mapping to {CLASS_MAPPING_PATH}...")
class_mapping_json = {str(idx): name for name, idx in class_map.items()}
with open(CLASS_MAPPING_PATH, 'w') as f:
    json.dump(class_mapping_json, f, indent=2)
print("[OK] Class mapping saved")

# Save treatment information
print(f"\nSaving treatment information to {TREATMENTS_PATH}...")
treatments = {}
for class_name, class_info in STRAWBERRY_CLASSES.items():
    treatments[class_info['index']] = {
        'disease_name': class_info['disease_name'],
        'scientific_name': class_info['scientific_name'],
        'treatment': class_info['treatment'],
        'severity': class_info['severity'],
        'prevention': class_info['prevention']
    }
with open(TREATMENTS_PATH, 'w') as f:
    json.dump(treatments, f, indent=2)
print("[OK] Treatment information saved")

print("\n" + "="*60)
print("SUCCESS: TRAINING COMPLETE!")
print("="*60)
print(f"Model saved to: {MODEL_OUTPUT_PATH}")
print(f"Class mapping: {CLASS_MAPPING_PATH}")
print(f"Final validation accuracy: {val_accuracy*100:.2f}%")
print("="*60)
