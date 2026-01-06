"""
Training script for Grape + Corn Disease Detection using MobileNetV2
Transfer learning with ImageDataGenerator augmentation
Parallel training with separate config
"""
import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json

# Configuration for Grape + Corn
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "pdd_grape_corn.h5"  # Separate model file
DATASET_PATH = DATA_DIR / "plantvillage dataset" / "color"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 3  # Fast mode for hackathon
VALIDATION_SPLIT = 0.2

# Grape + Corn classes only (7 classes total)
CLASSES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
]

# Disease treatments (subset for Grape + Corn)
DISEASE_TREATMENTS = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Fungal infection (Cercospora zeae-maydis)",
        "symptoms": "Rectangular gray lesions on leaves",
        "organic_treatment": ["Copper fungicide", "Remove infected leaves", "Improve drainage"],
        "chemical_treatment": ["Azoxystrobin", "Propiconazole"],
        "prevention": ["Crop rotation", "Resistant hybrids", "Remove plant debris"],
        "confidence_boost": True
    },
    "Corn_(maize)___Common_rust_": {
        "cause": "Fungal infection (Puccinia sorghi)",
        "symptoms": "Small reddish-brown pustules on leaves",
        "organic_treatment": ["Sulfur", "Remove infected leaves", "Improve air circulation"],
        "chemical_treatment": ["Propiconazole", "Trifloxystrobin"],
        "prevention": ["Resistant varieties", "Crop rotation", "Destroy plant debris"],
        "confidence_boost": True
    },
    "Corn_(maize)___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cause": "Fungal infection (Exserohilum turcicum)",
        "symptoms": "Long, elliptical lesions on leaves",
        "organic_treatment": ["Copper spray", "Remove infected leaves", "Improve air flow"],
        "chemical_treatment": ["Azoxystrobin", "Propiconazole"],
        "prevention": ["Resistant varieties", "Crop rotation", "Avoid overhead watering"],
        "confidence_boost": True
    },
    "Grape___Black_rot": {
        "cause": "Fungal infection (Guignardia bidwellii)",
        "symptoms": "Black, mummified berries, brown lesions on leaves",
        "organic_treatment": ["Sulfur", "Remove infected clusters", "Prune for air circulation"],
        "chemical_treatment": ["Myclobutanil", "Mancozeb"],
        "prevention": ["Remove infected material", "Resistant varieties", "Pruning"],
        "confidence_boost": True
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "Fungal infection (Phaeomoniella chlamydospora)",
        "symptoms": "Black spots on fruit, yellowing leaves with brown margins",
        "organic_treatment": ["Prune infected canes", "Improve drainage", "Remove plant material"],
        "chemical_treatment": ["Bordeaux mixture", "Sulfur"],
        "prevention": ["Remove infected wood", "Avoid wounding", "Proper sanitation"],
        "confidence_boost": True
    },
    "Grape___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Fungal infection (Phomopsis viticola)",
        "symptoms": "Brown spots with concentric rings on leaves",
        "organic_treatment": ["Sulfur", "Remove infected leaves", "Improve air circulation"],
        "chemical_treatment": ["Captan", "Mancozeb"],
        "prevention": ["Prune for air flow", "Destroy infected material", "Fungicide spray"],
        "confidence_boost": True
    },
}


def create_model(num_classes):
    """Create MobileNetV2-based transfer learning model"""
    print(f"\n📦 Building MobileNetV2 model for {num_classes} classes...")
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Build custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
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
    return model


def create_data_generators():
    """Create ImageDataGenerator with augmentation"""
    print("\n🔄 Setting up data augmentation...")
    
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
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("✓ Data augmentation configured")
    return train_datagen, val_datagen


def load_datasets(train_datagen, val_datagen):
    """Load datasets from directory structure"""
    print(f"\n📁 Loading dataset from {DATASET_PATH}...")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_PATH}")
    
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
    """Train the model with callbacks"""
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    
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
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=100,
        validation_steps=25
    )
    
    return history


def evaluate_model(model, val_generator):
    """Evaluate model on validation set"""
    print("\n📊 Evaluating model on validation set...")
    loss, accuracy = model.evaluate(val_generator)
    print(f"✓ Validation Loss: {loss:.4f}")
    print(f"✓ Validation Accuracy: {accuracy:.4f}")
    
    return {"loss": float(loss), "accuracy": float(accuracy)}


def save_class_mapping():
    """Save class index mapping to JSON"""
    class_mapping = {idx: class_name for idx, class_name in enumerate(CLASSES)}
    mapping_path = MODELS_DIR / "class_mapping_grape_corn.json"
    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"✓ Class mapping saved to {mapping_path}")
    return class_mapping


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🌱 CROP DISEASE DETECTION - GRAPE + CORN TRAINING")
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
    print(f"🗂️  Class mapping saved to: {MODELS_DIR / 'class_mapping_grape_corn.json'}")
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
