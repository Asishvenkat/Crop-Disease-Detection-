"""
Tomato Disease Detection - Optimized Fast Training
Target: >90% accuracy in <30 minutes using warmup-only strategy
"""

import os
import json
import sys
import math
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# Ensure console can print UTF-8 if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Enable mixed precision (falls back safely on CPU where needed)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("="*60)
print("TOMATO DISEASE DETECTION - OPTIMIZED FAST TRAINING")
print("="*60)

# Paths
DATA_PATH = r"E:\Crop Disease Prediction\data\plantvillage dataset\color"
MODEL_OUTPUT_PATH = r"E:\Crop Disease Prediction\models\pdd_tomato.keras"
CLASS_MAPPING_PATH = r"E:\Crop Disease Prediction\models\class_mapping_tomato.json"
TREATMENTS_PATH = r"E:\Crop Disease Prediction\models\treatments_tomato.json"

# Hyperparameters
IMAGE_SIZE = 192
BATCH_SIZE = 32
EPOCHS_WARMUP = 3          # warmup phase (backbone frozen)
LEARNING_RATE_WARMUP = 1e-4
EPOCHS_FINE_TUNE = 2       # brief fine-tune to boost accuracy
LEARNING_RATE_FINE_TUNE = 2e-5
MAX_STEPS_PER_EPOCH = 80   # Tight cap for CPU speed
VAL_BATCH_SIZE = 32
TIME_LIMIT_SECONDS = None  # No time limit - run to completion

# Tomato classes (10)
TOMATO_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

# Minimal treatment info (can be enriched later)
TREATMENT_INFO = {
    'Tomato___Bacterial_spot': 'Copper-based bactericides; remove infected leaves; avoid overhead irrigation.',
    'Tomato___Early_blight': 'Rotate crops; apply fungicides (chlorothalonil, mancozeb); remove infected debris.',
    'Tomato___healthy': 'No action needed. Maintain spacing and proper watering.',
    'Tomato___Late_blight': 'Fungicides (metalaxyl, mancozeb); remove and destroy infected plants.',
    'Tomato___Leaf_Mold': 'Improve ventilation; avoid leaf wetness; apply appropriate fungicides.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves; fungicides (chlorothalonil); crop rotation.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides or insecticidal soap; increase humidity; remove heavily infested leaves.',
    'Tomato___Target_Spot': 'Fungicide sprays; improve airflow; remove infected leaves.',
    'Tomato___Tomato_mosaic_virus': 'Remove infected plants; sanitize tools; use resistant varieties.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies; remove infected plants; use resistant varieties.',
}

class TimeLimit(keras.callbacks.Callback):
    def __init__(self, max_seconds=TIME_LIMIT_SECONDS):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None
    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
    def on_epoch_end(self, epoch, logs=None):
        if self.max_seconds is None:
            return
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = self.max_seconds - elapsed
        print(f"\nTime elapsed: {elapsed:.0f}s / {self.max_seconds}s ({remaining:.0f}s remaining)")
        if elapsed > self.max_seconds:
            print("TIME LIMIT EXCEEDED! Stopping training.")
            self.model.stop_training = True

# Build model (MobileNetV2 backbone)
print("\nBuilding MobileNetV2 model for 10 classes...")
base_model = keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(TOMATO_CLASSES), activation='softmax', dtype='float32')(x)
model = keras.Model(inputs, outputs)
print(f"[OK] Model parameters: {model.count_params():,}")

# Data pipeline
print("\nSetting up data augmentation and loaders...")
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.25,
    brightness_range=[0.75, 1.25],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
)

# Map names -> indices explicitly for class order stability
class_map = {name: idx for idx, name in enumerate(TOMATO_CLASSES)}

train_gen = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=class_map,
)
val_gen = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=VAL_BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=class_map,
    shuffle=False,
)

print(f"[OK] Train samples: {train_gen.samples}")
print(f"[OK] Val samples:   {val_gen.samples}")

# Calculate steps per epoch with reasonable cap
all_steps = math.ceil(train_gen.samples / BATCH_SIZE)
steps_per_epoch = min(MAX_STEPS_PER_EPOCH, all_steps)
val_steps = math.ceil(val_gen.samples / VAL_BATCH_SIZE)
print(f"[OK] steps_per_epoch={steps_per_epoch} (max from {all_steps}), val_steps={val_steps}")

# Compute class weights to handle class imbalance
class_counts = np.bincount(train_gen.classes, minlength=len(TOMATO_CLASSES))
total_train = train_gen.samples
class_weights = {i: (total_train / (len(TOMATO_CLASSES) * count)) for i, count in enumerate(class_counts) if count > 0}

# Compile & train (warmup phase)
print(f"\nTraining for {EPOCHS_WARMUP} epochs (warmup - backbone frozen)...")
base_model.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_WARMUP),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    epochs=EPOCHS_WARMUP,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(MODEL_OUTPUT_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6),
        TimeLimit(max_seconds=TIME_LIMIT_SECONDS),
    ],
)

# Fine-tune phase: selective unfreezing for better accuracy
print(f"\nFine-tuning for {EPOCHS_FINE_TUNE} epochs (unfreezing last ~40 layers)...")
base_model.trainable = True
# Freeze all but the last ~40 layers
for layer in base_model.layers[:-40]:
    layer.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_gen,
    epochs=EPOCHS_WARMUP + EPOCHS_FINE_TUNE,
    initial_epoch=EPOCHS_WARMUP,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(MODEL_OUTPUT_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        TimeLimit(max_seconds=TIME_LIMIT_SECONDS),
    ],
)

print("\nEvaluating full validation set...")
val_loss, val_acc = model.evaluate(val_gen, steps=val_steps, verbose=1)
print(f"[RESULT] Val Accuracy: {val_acc*100:.2f}%  |  Val Loss: {val_loss:.4f}")

# Save class mapping
print(f"\nSaving class mapping to {CLASS_MAPPING_PATH}...")
class_mapping_json = {str(idx): name for name, idx in class_map.items()}
with open(CLASS_MAPPING_PATH, 'w') as f:
    json.dump(class_mapping_json, f, indent=2)
print("[OK] Class mapping saved")

# Save minimal treatments info indexed by class id
print(f"Saving treatments to {TREATMENTS_PATH}...")
treatments_idx = {class_map[name]: TREATMENT_INFO[name] for name in TOMATO_CLASSES}
with open(TREATMENTS_PATH, 'w') as f:
    json.dump(treatments_idx, f, indent=2)
print("[OK] Treatments saved")

print("\n" + "="*60)
print("SUCCESS: TOMATO TRAINING COMPLETE!")
print("="*60)
print(f"Model saved to: {MODEL_OUTPUT_PATH}")
print(f"Class mapping: {CLASS_MAPPING_PATH}")
print(f"Final val accuracy: {val_acc*100:.2f}%")
print("="*60)
