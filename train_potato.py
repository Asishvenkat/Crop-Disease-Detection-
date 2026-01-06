"""
Potato Disease Detection Model Training
Specialized model for 3 potato disease classes
"""
import json
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "potato_model.h5"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_potato.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 6
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001

# 3 Potato disease classes
CLASSES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]


def create_model(num_classes):
    print(f"\n📦 Building MobileNetV2 model for {num_classes} potato classes...")
    
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Model created with {model.count_params():,} parameters")
    return model


def load_datasets():
    print(f"\n📁 Loading potato dataset...")
    
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


def train_model(model, train_gen, val_gen):
    print(f"\n🚀 Training potato model for {EPOCHS} epochs...")
    
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
        steps_per_epoch=60,
        validation_steps=15
    )
    
    return history


def evaluate_model(model, val_gen):
    print("\n📊 Evaluating potato model...")
    loss, accuracy = model.evaluate(val_gen, steps=15)
    print(f"✓ Validation Accuracy: {accuracy*100:.2f}%")
    return {"loss": loss, "accuracy": accuracy}


def save_class_mapping(class_indices):
    mapping = {str(v): k for k, v in class_indices.items()}
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Class mapping saved to {CLASS_MAPPING_PATH}")


def main():
    print("=" * 70)
    print("🥔 POTATO DISEASE DETECTION MODEL TRAINING")
    print("=" * 70)
    
    model = create_model(num_classes=len(CLASSES))
    train_gen, val_gen = load_datasets()
    history = train_model(model, train_gen, val_gen)
    results = evaluate_model(model, val_gen)
    save_class_mapping(train_gen.class_indices)
    
    print("\n" + "=" * 70)
    print("✅ POTATO MODEL TRAINING COMPLETE!")
    print(f"📁 Model: {MODEL_SAVE_PATH}")
    print(f"🎯 Accuracy: {results['accuracy']*100:.2f}%")
    print("=" * 70)
    
    return model, history, results


if __name__ == "__main__":
    main()
