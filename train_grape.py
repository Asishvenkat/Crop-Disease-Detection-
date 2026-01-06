"""
Grape Disease Detection Model Training
Transfer learning with ImageDataGenerator augmentation
Trains only on 4 grape disease classes for specialized detection

Goal tweaks:
- Use all grape images (no capped steps per epoch)
- Warmup + fine-tune to push accuracy >90% while staying under ~30 minutes
- Mixed precision on GPU if available
"""
import json
import os
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback,
)

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "pdd_grape.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_grape.json"

# Hyperparameters - optimized for fast + accurate training
IMAGE_SIZE = (192, 192)
BATCH_SIZE = 32
WARMUP_EPOCHS = 3
FINE_TUNE_EPOCHS = 9
EPOCHS = WARMUP_EPOCHS + FINE_TUNE_EPOCHS
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
FINE_TUNE_AT = 100  # freeze lower layers when unfreezing
MAX_TRAINING_MINUTES = 28  # safety wall to stay under 30 minutes

# Only 4 grape classes
CLASSES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
]

# Disease treatments for grape diseases
DISEASE_TREATMENTS = {
    "Grape___Black_rot": {
        "cause": "Fungal infection (Guignardia bidwellii)",
        "symptoms": "Black circular lesions on leaves and fruit, mummified berries",
        "organic_treatment": [
            "Remove infected plant debris",
            "Apply neem oil spray",
            "Improve air circulation",
            "Prune to reduce humidity"
        ],
        "chemical_treatment": [
            "Mancozeb fungicide",
            "Copper-based fungicide",
            "Apply at bud break and continue every 10-14 days"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Proper spacing for airflow",
            "Remove mummified berries",
            "Avoid overhead irrigation"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "Complex fungal infection (multiple pathogens)",
        "symptoms": "Tiger stripe leaf pattern, sudden vine death, internal wood decay",
        "organic_treatment": [
            "Prune infected wood during dormancy",
            "Improve soil health with compost",
            "Reduce plant stress with proper irrigation",
            "Remove severely infected vines"
        ],
        "chemical_treatment": [
            "No effective chemical treatment available",
            "Focus on cultural practices",
            "Proper wound protection during pruning"
        ],
        "prevention": [
            "Minimize pruning wounds",
            "Use clean pruning tools",
            "Avoid water stress",
            "Maintain vine vigor"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Fungal infection (Isariopsis spp.)",
        "symptoms": "Angular brown spots on leaves, premature defoliation",
        "organic_treatment": [
            "Remove infected leaves",
            "Apply sulfur spray",
            "Improve air circulation",
            "Copper-based organic fungicides"
        ],
        "chemical_treatment": [
            "Mancozeb or chlorothalonil",
            "Apply preventively in humid conditions",
            "Rotate fungicide classes"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Maintain proper plant spacing",
            "Remove fallen leaves",
            "Apply preventive sprays in wet seasons"
        ]
    },
    "Grape___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy green foliage, no lesions or spots",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": [
            "Maintain proper irrigation",
            "Ensure good air circulation",
            "Regular monitoring for early disease detection",
            "Balanced fertilization"
        ]
    }
}


class TimeLimit(Callback):
    """Stops training when wall clock exceeds the budget."""

    def __init__(self, max_minutes: float = MAX_TRAINING_MINUTES):
        super().__init__()
        self.max_seconds = max_minutes * 60
        self._start = None

    def on_train_begin(self, logs=None):
        if self._start is None:
            self._start = time.perf_counter()

    def on_batch_end(self, batch, logs=None):
        if time.perf_counter() - self._start > self.max_seconds:
            self.model.stop_training = True
            print("\n⏱️  Time limit reached, stopping early to respect budget.")


def configure_performance():
    """Enable mixed precision on GPU to speed up training."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("✓ Mixed precision enabled (GPU detected)")
        except Exception as exc:  # pragma: no cover - defensive only
            print(f"⚠️  Mixed precision could not be set: {exc}")


def create_model(num_classes):
    """Create MobileNetV2 transfer-learning model."""
    print(f"\n📦 Building MobileNetV2 model for {num_classes} classes...")

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # warmup stage

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"✓ Model created with {model.count_params():,} total parameters")
    return model


def create_data_generators():
    """Create ImageDataGenerator with balanced augmentation."""
    print("\n🔄 Setting up data augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        zoom_range=0.25,
        brightness_range=[0.75, 1.25],
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
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
    print(f"✓ Classes: {train_gen.class_indices}")
    
    return train_gen, val_gen


def train_model(model, train_gen, val_gen):
    """Warmup then fine-tune on full dataset."""
    print(f"\n🚀 Training for {EPOCHS} epochs (warmup {WARMUP_EPOCHS} + fine-tune {FINE_TUNE_EPOCHS})...")

    callbacks = [
        ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        TimeLimit(max_minutes=MAX_TRAINING_MINUTES),
    ]

    # Warmup with frozen base
    warmup_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=WARMUP_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tune upper layers
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE * 0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=warmup_history.epoch[-1] + 1,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Return the final history for downstream reporting
    return fine_tune_history


def evaluate_model(model, val_gen):
    """Evaluate model on validation set"""
    print("\n📊 Evaluating model on validation set...")
    loss, accuracy = model.evaluate(val_gen)
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


def save_treatment_info():
    """Save disease treatment information"""
    treatment_path = PROJECT_ROOT / "models" / "treatments_grape.json"
    print(f"\n💾 Saving treatment information to {treatment_path}...")
    
    with open(treatment_path, 'w') as f:
        json.dump(DISEASE_TREATMENTS, f, indent=2)
    
    print("✓ Treatment information saved")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🍇 GRAPE DISEASE DETECTION - SPECIALIZED TRAINING")
    print("=" * 70)

    # Performance tweaks (safe no-op on CPU)
    configure_performance()
    
    # Create model
    model = create_model(num_classes=len(CLASSES))
    
    # Load data
    train_gen, val_gen = load_datasets()
    
    # Train
    history = train_model(model, train_gen, val_gen)
    
    # Evaluate
    results = evaluate_model(model, val_gen)
    
    # Save mappings and metadata
    save_class_mapping(train_gen.class_indices)
    save_treatment_info()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📁 Class mapping: {CLASS_MAPPING_PATH}")
    print(f"🎯 Final validation accuracy: {results['accuracy']*100:.2f}%")
    print("=" * 70)
    
    return model, history, results


if __name__ == "__main__":
    main()
