"""
Pepper Bell Disease Detection Training
Aimed for >90% validation accuracy using MobileNetV2 transfer learning
"""
from pathlib import Path
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Paths
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "pdd_pepper.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_pepper.json"

# Training hyperparameters tuned for strong accuracy
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4

# Pepper classes (2-class problem)
CLASSES = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy"
]


def create_model(num_classes: int) -> Sequential:
    print(f"\n📦 Building MobileNetV2 model for {num_classes} pepper classes...")

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # keep backbone frozen for stable first stage

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"✓ Model created with {model.count_params():,} parameters")
    return model


def build_data_generators():
    print("\n🔄 Preparing data generators with augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        subset="validation",
        shuffle=False,
    )

    print(f"✓ Training samples: {train_gen.samples}")
    print(f"✓ Validation samples: {val_gen.samples}")
    print(f"✓ Classes: {list(train_gen.class_indices.keys())}")
    return train_gen, val_gen


def train_model(model: Sequential, train_gen, val_gen):
    print(f"\n🚀 Training pepper model for up to {EPOCHS} epochs...")
    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    return history


def evaluate_model(model: Sequential, val_gen):
    print("\n📊 Evaluating pepper model on validation set...")
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)
    loss, accuracy = model.evaluate(val_gen, steps=validation_steps)
    print(f"✓ Validation Loss: {loss:.4f}")
    print(f"✓ Validation Accuracy: {accuracy * 100:.2f}%")
    return {"loss": float(loss), "accuracy": float(accuracy)}


def save_class_mapping(class_indices):
    mapping = {str(v): k for k, v in class_indices.items()}
    CLASS_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Class mapping saved to {CLASS_MAPPING_PATH}")


def display_summary(history, results):
    print("\n" + "=" * 70)
    print("📈 TRAINING SUMMARY")
    print("=" * 70)
    final_train_acc = history.history["accuracy"][-1] * 100
    final_val_acc = results["accuracy"] * 100
    best_val_acc = max(history.history["val_accuracy"]) * 100
    print(f"Final Training Accuracy:   {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Best Validation Accuracy:  {best_val_acc:.2f}%")
    print(f"Total Epochs Completed:    {len(history.history['accuracy'])}")


def main():
    print("=" * 70)
    print("🌶️ PEPPER DISEASE DETECTION MODEL TRAINING")
    print("=" * 70)
    print("Target: >90% validation accuracy")
    print(f"Classes: {len(CLASSES)}")
    print("=" * 70)

    model = create_model(num_classes=len(CLASSES))
    train_gen, val_gen = build_data_generators()
    history = train_model(model, train_gen, val_gen)
    results = evaluate_model(model, val_gen)
    save_class_mapping(train_gen.class_indices)
    display_summary(history, results)

    print("\n" + "=" * 70)
    print("✅ PEPPER MODEL TRAINING COMPLETE")
    print(f"📁 Model saved: {MODEL_SAVE_PATH}")
    print(f"🎯 Final Accuracy: {results['accuracy'] * 100:.2f}%")
    if results["accuracy"] >= 0.90:
        print("🎉 TARGET ACHIEVED: >90% Accuracy!")
    else:
        print("📊 Target not yet reached; consider fine-tuning the backbone for a few epochs.")
    print("=" * 70)

    return model, history, results


if __name__ == "__main__":
    main()
