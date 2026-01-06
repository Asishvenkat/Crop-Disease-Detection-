"""
Training script for Crop Disease Detection using MobileNetV2
Transfer learning with ImageDataGenerator augmentation
"""
import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json

# Import configuration
from src.config import (
    DATASET_PATH, MODEL_PATH, IMAGE_SIZE, BATCH_SIZE, 
    EPOCHS, VALIDATION_SPLIT, CLASSES, MODELS_DIR
)


def create_model(num_classes):
    """
    Create MobileNetV2-based transfer learning model
    
    Architecture:
    - MobileNetV2 (pre-trained on ImageNet) for feature extraction
    - GlobalAveragePooling2D (better than Flatten for spatial data)
    - Dense(128) with ReLU and Dropout(0.5)
    - Dense(num_classes) with Softmax for classification
    
    Args:
        num_classes: Number of disease classes
    
    Returns:
        model: Compiled Keras model
    """
    print(f"\n📦 Building MobileNetV2 model for {num_classes} classes...")
    
    # Load pre-trained MobileNetV2 (ImageNet weights)
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights (transfer learning)
    base_model.trainable = False
    
    # Build custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Instead of Flatten
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Model created with {model.count_params():,} total parameters")
    print(f"  - Base model: {base_model.count_params():,} parameters (frozen)")
    print(f"  - Custom layers: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,} parameters (trainable)")
    
    return model


def create_data_generators():
    """
    Create ImageDataGenerator with augmentation for training and validation
    
    Augmentation strategy:
    - Rotation: ±20 degrees
    - Shift: ±20% horizontal/vertical
    - Zoom: ±20%
    - Horizontal flip: for disease symmetry
    - Brightness: ±20%
    
    Returns:
        train_gen, val_gen: Generator objects
    """
    print("\n🔄 Setting up data augmentation...")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )
    
    # Test/validation data generator (minimal preprocessing)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("✓ Data augmentation configured")
    return train_datagen, val_datagen


def load_datasets(train_datagen, val_datagen):
    """
    Load datasets from directory structure using flow_from_directory
    
    Expected structure:
    dataset/
      ├── class1/
      ├── class2/
      └── ...
    
    Args:
        train_datagen, val_datagen: ImageDataGenerator instances
    
    Returns:
        train_generator, val_generator, test_generator
    """
    print(f"\n📁 Loading dataset from {DATASET_PATH}...")
    
    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("   Please extract the Kaggle PlantVillage dataset to data/dataset/")
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_PATH}")
    
    # List available classes
    available_classes = [d.name for d in DATASET_PATH.iterdir() if d.is_dir()]
    print(f"   Found classes: {available_classes}")
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=CLASSES,
        seed=42
    )
    
    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=CLASSES,
        seed=42
    )
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator


def train_model(model, train_generator, val_generator):
    """
    Train the model with callbacks for early stopping and checkpointing
    
    Args:
        model: Keras model
        train_generator, val_generator: Data generators
    
    Returns:
        history: Training history
    """
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    
    return history


def evaluate_model(model, val_generator):
    """
    Evaluate model on validation set
    
    Args:
        model: Keras model
        val_generator: Validation data generator
    
    Returns:
        eval_results: Loss and accuracy
    """
    print("\n📊 Evaluating model on validation set...")
    loss, accuracy = model.evaluate(val_generator)
    print(f"✓ Validation Loss: {loss:.4f}")
    print(f"✓ Validation Accuracy: {accuracy:.4f}")
    
    return {"loss": float(loss), "accuracy": float(accuracy)}


def save_class_mapping():
    """Save class index mapping to JSON for inference"""
    class_mapping = {idx: class_name for idx, class_name in enumerate(CLASSES)}
    mapping_path = MODELS_DIR / "class_mapping.json"
    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"✓ Class mapping saved to {mapping_path}")
    return class_mapping


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🌱 CROP DISEASE DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create model
    model = create_model(len(CLASSES))
    
    # Step 2: Create data generators
    train_datagen, val_datagen = create_data_generators()
    
    # Step 3: Load datasets
    train_gen, val_gen = load_datasets(train_datagen, val_datagen)
    
    # Step 4: Train
    history = train_model(model, train_gen, val_gen)
    
    # Step 5: Evaluate
    eval_results = evaluate_model(model, val_gen)
    
    # Step 6: Save class mapping
    class_mapping = save_class_mapping()
    
    # Step 7: Final model info
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"📦 Model saved to: {MODEL_PATH}")
    print(f"🗂️  Class mapping saved to: {MODELS_DIR / 'class_mapping.json'}")
    print(f"📊 Final validation accuracy: {eval_results['accuracy']:.2%}")
    print("=" * 70)
    
    return model, history, eval_results


if __name__ == "__main__":
    try:
        model, history, results = main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
