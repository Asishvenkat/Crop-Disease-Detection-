"""
Potato Disease Detection Model Training - Enhanced Version
Optimized for >90% accuracy with improved architecture and training
"""
import json
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "potato_model.h5"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_potato.json"

# Enhanced configuration for better accuracy
IMAGE_SIZE = (224, 224)  # Increased from 160x160 for better feature extraction
BATCH_SIZE = 16  # Increased from 8 for more stable training
EPOCHS = 15  # Increased from 6 for better convergence
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001

# 3 Potato disease classes
CLASSES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]


def create_model(num_classes):
    print(f"\n📦 Building Enhanced MobileNetV2 model for {num_classes} potato classes...")
    
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Enhanced architecture with more capacity
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Enhanced model created with {model.count_params():,} parameters")
    return model


def load_datasets():
    print(f"\n📁 Loading potato dataset...")
    
    # Enhanced augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased from 20
        width_shift_range=0.25,  # Increased from 0.2
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip
        zoom_range=0.25,  # Increased from 0.2
        brightness_range=[0.8, 1.2],
        shear_range=0.15,  # Added shear transformation
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    train_gen = train_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='training',
        shuffle=True
    )
    
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
    print(f"✓ Classes found: {list(train_gen.class_indices.keys())}")
    
    return train_gen, val_gen


def train_model(model, train_gen, val_gen):
    print(f"\n🚀 Training enhanced potato model for {EPOCHS} epochs...")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
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
            patience=6,  # Increased patience
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
    
    # Calculate steps for full dataset coverage
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    return history


def evaluate_model(model, val_gen):
    print("\n📊 Evaluating enhanced potato model...")
    
    validation_steps = val_gen.samples // BATCH_SIZE
    loss, accuracy = model.evaluate(val_gen, steps=validation_steps)
    
    print(f"✓ Validation Loss: {loss:.4f}")
    print(f"✓ Validation Accuracy: {accuracy*100:.2f}%")
    
    return {"loss": loss, "accuracy": accuracy}


def save_class_mapping(class_indices):
    mapping = {str(v): k for k, v in class_indices.items()}
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Class mapping saved to {CLASS_MAPPING_PATH}")


def display_training_summary(history, results):
    """Display training summary with key metrics"""
    print("\n" + "=" * 70)
    print("📈 TRAINING SUMMARY")
    print("=" * 70)
    
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc = results['accuracy'] * 100
    best_val_acc = max(history.history['val_accuracy']) * 100
    
    print(f"Final Training Accuracy:   {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Best Validation Accuracy:  {best_val_acc:.2f}%")
    print(f"Total Epochs Completed:    {len(history.history['accuracy'])}")


def main():
    print("=" * 70)
    print("🥔 ENHANCED POTATO DISEASE DETECTION MODEL TRAINING")
    print("=" * 70)
    print(f"Target: >90% Accuracy")
    print(f"Classes: {len(CLASSES)} potato disease types")
    print("=" * 70)
    
    # Create and train model
    model = create_model(num_classes=len(CLASSES))
    train_gen, val_gen = load_datasets()
    history = train_model(model, train_gen, val_gen)
    results = evaluate_model(model, val_gen)
    save_class_mapping(train_gen.class_indices)
    
    # Display summary
    display_training_summary(history, results)
    
    print("\n" + "=" * 70)
    print("✅ ENHANCED POTATO MODEL TRAINING COMPLETE!")
    print(f"📁 Model saved: {MODEL_SAVE_PATH}")
    print(f"🎯 Final Accuracy: {results['accuracy']*100:.2f}%")
    
    if results['accuracy'] >= 0.90:
        print("🎉 TARGET ACHIEVED: >90% Accuracy!")
    else:
        print(f"📊 Target Progress: {results['accuracy']*100:.2f}% / 90%")
    
    print("=" * 70)
    print("\n✓ Tomato model remains unchanged and safe!")
    print(f"✓ Potato model saved separately to: {MODEL_SAVE_PATH}")
    
    return model, history, results


if __name__ == "__main__":
    main()
