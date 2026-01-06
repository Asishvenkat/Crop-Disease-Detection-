"""
WINNING TOMATO MODEL - ResNet50V2 for >95% Accuracy
Optimized specifically for hackathon victory
"""
import json
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration - HACKATHON WINNING SETUP
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "tomato_model.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_tomato.json"

IMAGE_SIZE = (224, 224)  # ResNet standard size
BATCH_SIZE = 32  # Larger batch = faster epochs
EPOCHS = 12  # Reduced epochs
VALIDATION_SPLIT = 0.2
INITIAL_LR = 0.001
FINE_TUNE_LR = 0.0001

# 7 tomato disease classes
CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot"
]


def create_model(num_classes):
    print(f"\n🏆 Building WINNING ResNet50V2 model for {num_classes} classes...")
    
    base_model = ResNet50V2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Initially freeze base model
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ ResNet50V2 model created with {model.count_params():,} parameters")
    return model


def load_datasets():
    print(f"\n📁 Loading tomato dataset with advanced augmentation...")
    
    # WINNING augmentation strategy
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        shear_range=0.2,
        channel_shift_range=0.2,
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
    
    return train_gen, val_gen


def train_phase_1(model, train_gen, val_gen):
    """Phase 1: Train with frozen base"""
    print(f"\n🚀 PHASE 1: Training top layers (base frozen) - 15 epochs...")
    
    callbacks = [
        ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
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
    
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=8,  # Reduced for speed
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    return history1


def train_phase_2(model, train_gen, val_gen):
    """Phase 2: Fine-tune with unfrozen layers"""
    print(f"\n🔥 PHASE 2: Fine-tuning (unfreezing base) - 8 more epochs...")
    
    # Unfreeze base model for fine-tuning
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze first 100 layers, unfreeze rest
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    unfrozen = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"✓ Unfroze {unfrozen} layers for fine-tuning")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
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
    
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=16,  # Total 16 epochs (8+8)
        initial_epoch=8,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    return history2


def evaluate_model(model, val_gen):
    print("\n📊 FINAL EVALUATION...")
    
    validation_steps = val_gen.samples // BATCH_SIZE
    loss, accuracy = model.evaluate(val_gen, steps=validation_steps, verbose=1)
    
    print(f"\n✓ Validation Loss: {loss:.4f}")
    print(f"✓ Validation Accuracy: {accuracy*100:.2f}%")
    
    return {"loss": loss, "accuracy": accuracy}


def save_class_mapping(class_indices):
    mapping = {str(v): k for k, v in class_indices.items()}
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Class mapping saved")


def main():
    print("=" * 70)
    print("🏆 HACKATHON WINNING TOMATO MODEL - ResNet50V2")
    print("=" * 70)
    print(f"Target: >95% Accuracy for VICTORY!")
    print(f"Architecture: ResNet50V2 (powerful deep residual network)")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Classes: {len(CLASSES)} tomato diseases")
    print("=" * 70)
    
    # Create model
    model = create_model(num_classes=len(CLASSES))
    
    # Load datasets
    train_gen, val_gen = load_datasets()
    
    # Two-phase training
    print("\n🎯 TWO-PHASE TRAINING STRATEGY:")
    print("   Phase 1: Train top layers (base frozen)")
    print("   Phase 2: Fine-tune with unfrozen base")
    
    history1 = train_phase_1(model, train_gen, val_gen)
    history2 = train_phase_2(model, train_gen, val_gen)
    
    # Final evaluation
    results = evaluate_model(model, val_gen)
    save_class_mapping(train_gen.class_indices)
    
    print("\n" + "=" * 70)
    print("🏆 HACKATHON MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved: {MODEL_SAVE_PATH}")
    print(f"🎯 FINAL ACCURACY: {results['accuracy']*100:.2f}%")
    
    if results['accuracy'] >= 0.95:
        print("\n🎉🎉🎉 VICTORY! >95% ACCURACY ACHIEVED! 🎉🎉🎉")
        print("🏆 READY TO WIN THE HACKATHON! 🏆")
    elif results['accuracy'] >= 0.90:
        print("\n✅ EXCELLENT! >90% Accuracy - Strong contender!")
    else:
        print(f"\n📊 Accuracy: {results['accuracy']*100:.2f}%")
    
    print("=" * 70)
    
    return model, results


if __name__ == "__main__":
    main()
