"""
Comprehensive Crop Disease Detection Model Training
Trains on ALL crops: Apple, Potato, Tomato, Grape, Corn
Transfer learning with ImageDataGenerator augmentation
"""
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "pdd_all_crops.h5"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_all_crops.json"

# Hyperparameters - optimized for balanced speed and accuracy
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 6
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001

# ALL 18 classes - Apple, Potato, Tomato, Grape (NO CORN)
CLASSES = [
    # Apple (4 classes)
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    # Grape (4 classes)
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    # Potato (3 classes)
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    # Tomato (7 classes)
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
    "Tomato___healthy"
]


def create_model(num_classes):
    """
    Create MobileNetV2-based transfer learning model
    """
    print(f"\n📦 Building MobileNetV2 model for {num_classes} classes...")
    
    # Base model
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model
    
    # Custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Model created with {model.count_params():,} total parameters")
    return model


def create_data_generators():
    """Create ImageDataGenerator with augmentation"""
    print("\n🔄 Setting up data augmentation...")
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=VALIDATION_SPLIT
    )
    
    # Validation (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    print("✓ Data augmentation configured")
    return train_datagen, val_datagen


def load_datasets():
    """Load and prepare training/validation datasets"""
    print(f"\n📁 Loading dataset from {DATASET_PATH}...")
    
    train_datagen, val_datagen = create_data_generators()
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_gen = val_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='validation',
        shuffle=False
    )
    
    print(f"✓ Training samples: {train_gen.samples}")
    print(f"✓ Validation samples: {val_gen.samples}")
    print(f"✓ Classes loaded: {len(CLASSES)}")
    
    return train_gen, val_gen


def train_model(model, train_gen, val_gen):
    """Train model with callbacks"""
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=200,  # Balanced for comprehensive training
        validation_steps=50
    )
    
    return history


def evaluate_model(model, val_gen):
    """Evaluate model on validation set"""
    print("\n📊 Evaluating model on validation set...")
    loss, accuracy = model.evaluate(val_gen, steps=50)
    print(f"✓ Validation Loss: {loss:.4f}")
    print(f"✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return {"loss": loss, "accuracy": accuracy}


def save_class_mapping(class_indices):
    """Save class index mapping to JSON"""
    print(f"\n💾 Saving class mapping to {CLASS_MAPPING_PATH}...")
    
    # Invert mapping (class_name -> index) to (index -> class_name)
    mapping = {str(v): k for k, v in class_indices.items()}
    
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("✓ Class mapping saved")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🌱 COMPREHENSIVE CROP DISEASE DETECTION - ALL CROPS TRAINING")
    print("=" * 70)
    print(f"📋 Training on {len(CLASSES)} classes:")
    print(f"   - Apple: 4 classes")
    print(f"   - Corn: 4 classes")
    print(f"   - Grape: 4 classes")
    print(f"   - Potato: 3 classes")
    print(f"   - Tomato: 7 classes")
    print("=" * 70)
    
    # Create model
    model = create_model(num_classes=len(CLASSES))
    
    # Load data
    train_gen, val_gen = load_datasets()
    
    # Train
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate
    results = evaluate_model(model, val_gen)
    
    # Save mappings
    save_class_mapping(train_gen.class_indices)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📁 Class mapping: {CLASS_MAPPING_PATH}")
    print(f"🎯 Final validation accuracy: {results['accuracy']*100:.2f}%")
    print(f"📊 Total classes: {len(CLASSES)}")
    print("=" * 70)
    
    return model, history, results


if __name__ == "__main__":
    main()
