"""
Test Apple Model Accuracy
Evaluate the trained apple disease detection model on validation set
"""
import json
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "plantvillage dataset" / "color"
MODEL_PATH = PROJECT_ROOT / "models" / "pdd_apple.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping_apple.json"

IMAGE_SIZE = (192, 192)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Apple classes
CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]


def load_validation_data():
    """Load validation dataset"""
    print(f"\n📁 Loading validation dataset from {DATASET_PATH}...")
    
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT
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
    
    print(f"✓ Validation samples: {val_gen.samples}")
    print(f"✓ Classes: {list(val_gen.class_indices.keys())}")
    
    return val_gen


def evaluate_model(model, val_gen):
    """Evaluate model and return detailed metrics"""
    print("\n📊 Evaluating apple model on validation set...")
    
    # Overall accuracy
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    
    print(f"\n{'='*70}")
    print(f"🎯 OVERALL VALIDATION ACCURACY: {accuracy*100:.2f}%")
    print(f"📉 Validation Loss: {loss:.4f}")
    print(f"{'='*70}")
    
    # Get predictions for detailed metrics
    print("\n🔍 Generating predictions for detailed analysis...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Per-class accuracy
    print(f"\n{'='*70}")
    print("📋 PER-CLASS PERFORMANCE:")
    print(f"{'='*70}")
    
    class_names = list(val_gen.class_indices.keys())
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_correct = np.sum((y_pred == i) & class_mask)
        class_total = np.sum(class_mask)
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        
        # Clean display name
        display_name = class_name.replace('Apple___', '').replace('_', ' ')
        print(f"  {display_name:30s}: {class_acc:6.2f}% ({class_correct}/{class_total})")
    
    # Calculate confusion matrix manually
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    print(f"\n{'='*70}")
    print("🔢 CONFUSION MATRIX:")
    print(f"{'='*70}")
    print("         ", end="")
    for name in class_names:
        print(f"{name.replace('Apple___', '')[:8]:>8}", end=" ")
    print()
    for i, name in enumerate(class_names):
        print(f"{name.replace('Apple___', '')[:8]:>8} ", end="")
        for j in range(num_classes):
            print(f"{cm[i,j]:>8}", end=" ")
        print()
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'predictions': predictions,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def main():
    """Main testing pipeline"""
    print("=" * 70)
    print("🍎 APPLE MODEL ACCURACY TEST")
    print("=" * 70)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python train_apple.py")
        return
    
    # Load model
    print(f"\n📦 Loading model from {MODEL_PATH}...")
    model = load_model(str(MODEL_PATH))
    print(f"✓ Model loaded successfully")
    print(f"✓ Model parameters: {model.count_params():,}")
    
    # Load validation data
    val_gen = load_validation_data()
    
    # Evaluate
    results = evaluate_model(model, val_gen)
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    print(f"🎯 Final Accuracy: {results['accuracy']*100:.2f}%")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
