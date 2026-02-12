"""
FastAPI backend for Crop Disease Detection
Implements prediction endpoint with confidence-aware logic, severity scoring,
symptom fusion, and treatment recommendations
"""
import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
import keras
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Import configuration and modules
from src.config import (
    MODEL_PATH, IMAGE_SIZE, CLASSES_DICT, CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_MESSAGE, SEVERITY_THRESHOLDS, DISEASE_TREATMENTS,
    SYMPTOM_DISEASE_MAPPING, SYMPTOM_CONTRADICTION_PENALTY,
    API_HOST, API_PORT, API_RELOAD, BLUR_THRESHOLD, BRIGHTNESS_BOUNDS,
    TEMPERATURE, DEFAULT_CROP, ENABLE_GRADCAM, GRADCAM_RESOLUTION
)
from src.grad_cam import GradCAM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🌱 Crop Disease Detection API",
    description="ML-powered crop disease detection with explainable AI",
    version="1.0.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
grad_cam = None
class_mapping = None
current_crop = DEFAULT_CROP  # Track current crop
CLASSES = CLASSES_DICT[DEFAULT_CROP]  # Current crop's classes
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"

# Model cache to avoid reloading
MODEL_CACHE = {}  # Maps crop_name -> loaded model
GRADCAM_CACHE = {}  # Maps crop_name -> GradCAM object


# Pydantic models for request/response validation
class SymptomInput(BaseModel):
    """Optional symptom checklist"""
    yellowing_leaves: Optional[bool] = False
    brown_spots: Optional[bool] = False
    wilting: Optional[bool] = False
    white_fungal_growth: Optional[bool] = False


class PredictionResponse(BaseModel):
    """Structured prediction response"""
    disease: str
    confidence: float
    confidence_percent: str
    severity_score: float
    severity_level: str
    message: str
    treatment_advice: Dict[str, Any]
    heatmap_base64: Optional[str] = None
    explanation: str
    symptom_analysis: Optional[Dict[str, Any]] = None
    model_info: Dict[str, str]
    image_quality: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """User feedback payload"""
    prediction: str
    user_label: Optional[str] = None
    helpful: Optional[bool] = None
    comments: Optional[str] = None
    confidence: Optional[float] = None


def load_model_for_crop(crop_name):
    """
    Load model for a specific crop (with caching, no dummy forward pass, lazy Grad-CAM)
    ~3-5 seconds instead of 18+
    """
    global model, grad_cam, class_mapping, current_crop, CLASSES
    
    from src.config import MODEL_PATHS, CLASSES_DICT
    
    if crop_name not in MODEL_PATHS:
        raise ValueError(f"Unknown crop: {crop_name}")
    
    model_path = MODEL_PATHS[crop_name]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for {crop_name} at {model_path}")
    
    logger.info(f"Loading {crop_name} model...")
    
    # Check cache first
    if crop_name in MODEL_CACHE:
        logger.info(f"✓ Using cached {crop_name} model")
        model = MODEL_CACHE[crop_name]
    else:
        # Load model (skip dummy forward pass - not necessary)
        model = keras.models.load_model(str(model_path))
        MODEL_CACHE[crop_name] = model
        logger.info(f"✓ {crop_name.upper()} model loaded & cached")
    
    # Grad-CAM is initialized lazily on first predict, not on model switch
    grad_cam = GRADCAM_CACHE.get(crop_name)
    if grad_cam is None:
        logger.info(f"Grad-CAM will be initialized on first prediction for {crop_name}")
    
    # Load class mapping
    mapping_path = model_path.parent / f"class_mapping_{crop_name}.json"
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Update global CLASSES variable
    current_crop = crop_name
    CLASSES = CLASSES_DICT[crop_name]
    
    logger.info(f"✓ {crop_name.upper()} ready ({len(class_mapping)} classes)")


def init_grad_cam_lazy(crop_name):
    """
    Initialize Grad-CAM only when needed (on first prediction)
    """
    global grad_cam
    
    if crop_name in GRADCAM_CACHE:
        grad_cam = GRADCAM_CACHE[crop_name]
        return
    
    from src.config import GRAD_CAM_LAYERS
    layer_name = GRAD_CAM_LAYERS.get(crop_name, None)
    if layer_name and model is not None:
        try:
            grad_cam = GradCAM(model, layer_name=layer_name)
            GRADCAM_CACHE[crop_name] = grad_cam
            logger.info(f"✓ Grad-CAM initialized for {crop_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Grad-CAM for {crop_name}: {e}")
            grad_cam = None
    else:
        grad_cam = None
        logger.warning(f"Grad-CAM layer not configured for {crop_name}")



def load_model_and_grad_cam():
    """
    Pre-load all models on startup for instant model switching.
    Trades slower startup (~60-90s) for instant model switches (~1-2ms).
    """
    from src.config import MODEL_PATHS
    
    logger.info("Pre-loading all crop models for instant switching...")
    for crop_name in sorted(MODEL_PATHS.keys()):
        try:
            load_model_for_crop(crop_name)
            logger.info(f"  ✓ {crop_name} loaded")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {crop_name}: {e}")
    
    logger.info("[OK] All models ready - model switching will be instant")


def preload_all_models():
    """Alias for backwards compatibility"""
    load_model_and_grad_cam()


def preprocess_image(image_file):
    """
    Preprocess uploaded image for model inference (OPTIMIZED)
    
    Args:
        image_file: PIL Image or file path
    
    Returns:
        img_array: Preprocessed array with values in [0, 1]
        original_img: Original image for Grad-CAM visualization
    """
    # Convert to PIL Image if needed
    if isinstance(image_file, bytes):
        img = Image.open(BytesIO(image_file))
    else:
        img = image_file
    
    # Get current crop's image size
    from src.config import IMAGE_SIZES
    crop_image_size = IMAGE_SIZES.get(current_crop, (224, 224))
    
    # Convert to RGB first (before resizing for speed)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size using faster BILINEAR instead of LANCZOS
    img = img.resize(crop_image_size, Image.Resampling.BILINEAR)
    
    # Create array for model (normalized to [0, 1])
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Store for visualization
    original_img = np.array(img)
    
    return img_array, original_img


def assess_image_quality(original_img: np.ndarray) -> Dict[str, Any]:
    """Estimate blur and lighting to warn on poor captures."""
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    is_blurry = blur_score < BLUR_THRESHOLD
    is_too_dark = brightness < BRIGHTNESS_BOUNDS[0]
    is_too_bright = brightness > BRIGHTNESS_BOUNDS[1]

    issues = []
    if is_blurry:
        issues.append("Image appears blurry; try refocusing or steadying the camera.")
    if is_too_dark:
        issues.append("Image is too dark; move to better light.")
    if is_too_bright:
        issues.append("Image is over-exposed; avoid harsh light.")

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "is_blurry": is_blurry,
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
        "issues": issues
    }


def apply_temperature_scaling(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to smooth confidences when the model outputs probabilities."""
    temperature = max(1e-3, float(temperature))
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    scaled = exp_logits / np.sum(exp_logits)
    return scaled


def predict_disease(img_array, original_img):
    """
    Run model prediction
    
    Args:
        img_array: Preprocessed image array
        original_img: Original image for visualization
    
    Returns:
        predictions_dict: Contains disease, confidence, class index
    """
    # Ensure batch dimension
    if len(img_array.shape) == 3:
        img_batch = np.expand_dims(img_array, axis=0)
    else:
        img_batch = img_array
    
    # Predict
    logits = model.predict(img_batch, verbose=0)
    probabilities = logits[0]

    # Temperature scaling for calibrated confidences
    probabilities = apply_temperature_scaling(probabilities, TEMPERATURE)

    pred_index = np.argmax(probabilities)
    confidence = probabilities[pred_index]
    
    # Get disease name
    disease = class_mapping[str(pred_index)]
    
    return {
        "disease": disease,
        "confidence": float(confidence),
        "pred_index": int(pred_index),
        "probabilities": {class_mapping[str(i)]: float(prob) for i, prob in enumerate(probabilities)},
        "img_array": img_array,
        "original_img": original_img
    }


def calculate_severity(disease, confidence):
    """
    Calculate severity score and level based on disease type and confidence
    
    Severity represents how advanced/damaging the disease is, not prediction confidence.
    Different diseases have different inherent severity levels.
    
    Args:
        disease: Disease name (e.g., "Tomato___Early_blight")
        confidence: Model confidence (0-1)
    
    Returns:
        severity_score, severity_level
    """
    # Disease severity mapping (inherent danger level)
    DISEASE_SEVERITY_BASE = {
        # High severity diseases (70-95)
        "Late_blight": 90,
        "Bacterial_spot": 75,
        "Early_blight": 70,
        
        # Medium severity diseases (40-65)
        "Target_Spot": 60,
        "Leaf_Mold": 55,
        "Septoria_leaf_spot": 50,
        
        # Low severity (0-30)
        "healthy": 0,
    }
    
    # Extract disease name from full class name
    disease_short = disease.split("___")[-1] if "___" in disease else disease
    
    # Get base severity for this disease type
    base_severity = DISEASE_SEVERITY_BASE.get(disease_short, 50)  # Default 50 if unknown
    
    # Adjust slightly based on confidence (±10 points)
    # High confidence = more certain about severity
    confidence_factor = (confidence - 0.5) * 20  # -10 to +10
    severity_score = max(0, min(100, base_severity + confidence_factor))
    
    # Determine severity level
    for level, (min_val, max_val) in SEVERITY_THRESHOLDS.items():
        if min_val <= severity_score < max_val:
            return severity_score, level
    
    return severity_score, "Severe"


def apply_symptom_fusion(disease, confidence, symptoms_dict):
    """
    Apply rule-based symptom fusion to adjust confidence
    
    Logic:
    - If symptoms support predicted disease: no change
    - If symptoms contradict predicted disease: reduce confidence by ~12%
    - If no symptoms provided: use model confidence as-is
    
    Args:
        disease: Predicted disease name
        confidence: Model confidence (0-1)
        symptoms_dict: Dictionary of symptom booleans
    
    Returns:
        adjusted_confidence, symptom_analysis
    """
    active_symptoms = [s for s, v in symptoms_dict.items() if v]
    
    if not active_symptoms:
        return confidence, {"active_symptoms": [], "adjustment": 0.0, "reason": "No symptoms provided"}
    
    # Check if symptoms support the predicted disease
    supporting_symptoms = []
    contradicting_symptoms = []
    
    for symptom, disease_list in SYMPTOM_DISEASE_MAPPING.items():
        if symptoms_dict.get(symptom, False):
            if disease in disease_list:
                supporting_symptoms.append(symptom)
            else:
                contradicting_symptoms.append(symptom)
    
    # Apply adjustment
    adjustment = 0.0
    reason = ""
    
    if contradicting_symptoms and not supporting_symptoms:
        # Symptoms contradict prediction
        adjustment = -SYMPTOM_CONTRADICTION_PENALTY
        reason = f"Symptoms {contradicting_symptoms} don't match {disease}. Confidence reduced by {SYMPTOM_CONTRADICTION_PENALTY*100:.0f}%"
    elif supporting_symptoms:
        # Symptoms support prediction
        reason = f"Symptoms {supporting_symptoms} support {disease} prediction"
    
    adjusted_confidence = max(0.0, min(1.0, confidence + adjustment))
    
    return adjusted_confidence, {
        "active_symptoms": active_symptoms,
        "supporting_symptoms": supporting_symptoms,
        "contradicting_symptoms": contradicting_symptoms,
        "confidence_adjustment": adjustment,
        "reason": reason
    }


def generate_heatmap(img_array, original_img, pred_index):
    """
    Generate saliency heatmap for prediction explanation
    
    Args:
        img_array: Preprocessed image
        original_img: Original image
        pred_index: Class index
    
    Returns:
        heatmap_base64: Encoded heatmap image or None
    """
    try:
        if grad_cam is None:
            logger.warning("Grad-CAM not initialized, skipping heatmap")
            return None
            
        logger.info(f"Generating saliency heatmap for class index {pred_index}...")
        
        explanation = grad_cam.explain_prediction(
            img_array, 
            original_img, 
            pred_index=pred_index,
            alpha=0.4
        )
        
        # Convert overlay to base64 for JSON response
        overlay = explanation["overlay"]
        
        # Ensure it's uint8
        if overlay.dtype != np.uint8:
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        # Convert BGR back to RGB for proper display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Encode to PNG
        success, buffer = cv2.imencode('.png', overlay)
        if success:
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info(f"Heatmap generated successfully ({len(heatmap_base64)} bytes)")
            return heatmap_base64
        else:
            logger.warning("Failed to encode heatmap to PNG")
            return None
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}", exc_info=True)
        return None


def get_treatment_advice(disease, confidence):
    """
    Get treatment recommendations for detected disease
    
    Args:
        disease: Disease name
        confidence: Model confidence (used to suggest expert consultation)
    
    Returns:
        treatment_dict: Treatment advice
    """
    if disease not in DISEASE_TREATMENTS:
        return {
            "disease": disease,
            "cause": "Unknown disease",
            "organic_treatment": ["Consult agricultural expert"],
            "chemical_treatment": ["Consult agricultural expert"],
            "prevention": ["Maintain good crop health"]
        }
    
    treatment_data = DISEASE_TREATMENTS[disease].copy()
    
    # Add confidence-based recommendation
    if confidence < CONFIDENCE_THRESHOLD:
        treatment_data["expert_consultation"] = "⚠️ Low confidence detection - strongly recommended to consult expert"
    
    return treatment_data


def log_prediction_event(payload: Dict[str, Any]) -> None:
    """Persist lightweight prediction metadata for observability."""
    record = {
        "disease": payload.get("disease"),
        "confidence": payload.get("confidence"),
        "severity": payload.get("severity_level"),
        "latency_ms": payload.get("latency_ms"),
        "timestamp": time.time(),
        "quality": payload.get("image_quality", {}),
        "symptoms": payload.get("symptoms", {})
    }
    try:
        with PREDICTION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write prediction log: {exc}")


def log_feedback(payload: Dict[str, Any]) -> None:
    """Append user feedback to JSONL for later analysis."""
    record = {"timestamp": time.time(), **payload}
    try:
        with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write feedback log: {exc}")


@app.on_event("startup")
async def startup_event():
    """Load model on API startup"""
    logger.info("Starting Crop Disease Detection API...")
    load_model_and_grad_cam()
    logger.info("[OK] API ready for predictions")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "grad_cam_ready": grad_cam is not None,
        "classes": CLASSES
    }


@app.post("/select_crop")
async def select_crop(crop: str):
    """
    Select a crop model (tomato or potato)
    
    Endpoint: POST /select_crop
    Input:
        - crop: "tomato" or "potato"
    
    Output:
        - status: Success/error status
        - crop: Selected crop name
        - classes: Number of disease classes
        - accuracy: Model accuracy
    """
    try:
        load_model_for_crop(crop)
        
        accuracy_map = {
            "tomato": 0.89,
            "potato": 0.9583,
            "grape": 0.9753,
            "apple": 0.9432,
            "corn": 0.90,
        }
        
        return {
            "status": "success",
            "crop": crop,
            "classes": len(class_mapping),
            "accuracy": f"{accuracy_map.get(crop, 0)*100:.2f}%"
        }
    except Exception as e:
        logger.error(f"Error selecting crop: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# NOTE: The /crops endpoint is defined later to use config MODEL_PATHS.


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    yellowing_leaves: Optional[bool] = Form(False),
    brown_spots: Optional[bool] = Form(False),
    wilting: Optional[bool] = Form(False),
    white_fungal_growth: Optional[bool] = Form(False)
):
    """
    Main prediction endpoint
    
    Endpoint: POST /predict
    Input:
        - image: MultiPart form image file
        - yellowing_leaves: Optional symptom
        - brown_spots: Optional symptom
        - wilting: Optional symptom
        - white_fungal_growth: Optional symptom
    
    Output:
        - disease: Predicted disease name
        - confidence: Model confidence (0-1)
        - confidence_percent: Formatted percentage
        - severity_score: Confidence × 100
        - severity_level: Mild/Moderate/Severe
        - message: Human-readable prediction message
        - treatment_advice: Treatment recommendations
        - heatmap_base64: Grad-CAM visualization (base64)
        - explanation: Heatmap interpretation
        - symptom_analysis: Rule-based symptom fusion results
        - model_info: Model metadata
    """
    start_time = time.perf_counter()
    try:
        # Initialize Grad-CAM lazily on first prediction (only once per crop)
        init_grad_cam_lazy(current_crop)
        
        # Read image
        contents = await image.read()
        img_array, original_img = preprocess_image(contents)

        # Image quality assessment
        quality_info = assess_image_quality(original_img)

        # Step 1: Model prediction
        pred_data = predict_disease(img_array, original_img)
        disease = pred_data["disease"]
        confidence = pred_data["confidence"]
        pred_index = pred_data["pred_index"]
        
        # Step 2: Apply symptom fusion (rule-based)
        symptoms_dict = {
            "yellowing_leaves": yellowing_leaves,
            "brown_spots": brown_spots,
            "wilting": wilting,
            "white_fungal_growth": white_fungal_growth
        }
        adjusted_confidence, symptom_analysis = apply_symptom_fusion(
            disease, confidence, symptoms_dict
        )
        
        # Step 3: Calculate severity (now disease-dependent)
        severity_score, severity_level = calculate_severity(disease, adjusted_confidence)
        
        # Step 4: Generate message based on confidence and quality
        if adjusted_confidence < CONFIDENCE_THRESHOLD:
            message = LOW_CONFIDENCE_MESSAGE
        else:
            message = f"Detected: {disease} ({adjusted_confidence:.1%} confidence, {severity_level})"
        if quality_info["issues"]:
            message += " | Image quality issues detected: " + "; ".join(quality_info["issues"])
        
        # Step 5: Get treatment advice
        treatment_advice = get_treatment_advice(disease, adjusted_confidence)
        
        # Step 6: Generate Grad-CAM heatmap (OPTIMIZED - can disable for speed)
        heatmap_base64 = None
        if ENABLE_GRADCAM and grad_cam is not None:
            heatmap_base64 = generate_heatmap(pred_data["img_array"], original_img, pred_index)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        # Step 7: Build response
        response = PredictionResponse(
            disease=disease,
            confidence=adjusted_confidence,
            confidence_percent=f"{adjusted_confidence:.1%}",
            severity_score=severity_score,
            severity_level=severity_level,
            message=message,
            treatment_advice=treatment_advice,
            heatmap_base64=heatmap_base64,
            explanation="Saliency map shows which leaf regions most influenced the prediction",
            symptom_analysis=symptom_analysis,
            model_info={
                "architecture": "MobileNetV2 + Transfer Learning",
                "input_size": "224x224 RGB",
                "classes": str(len(CLASSES)),
                "confidence_threshold": str(CONFIDENCE_THRESHOLD),
                "grad_cam": "enabled (saliency)" if heatmap_base64 else "not_available"
            },
            image_quality=quality_info
        )

        log_prediction_event({
            "disease": disease,
            "confidence": adjusted_confidence,
            "severity_level": severity_level,
            "latency_ms": latency_ms,
            "image_quality": quality_info,
            "symptoms": symptoms_dict
        })
        logger.info(f"✓ Prediction: {disease} ({adjusted_confidence:.1%}) | latency {latency_ms:.1f} ms")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
async def get_classes():
    """Get list of supported disease classes"""
    return {
        "classes": CLASSES,
        "count": len(CLASSES)
    }


@app.get("/crops")
async def get_available_crops():
    """Get available crop models"""
    from src.config import MODEL_PATHS
    return {
        "crops": list(MODEL_PATHS.keys()),
        "current": current_crop
    }


@app.post("/select-crop")
async def select_crop(crop: str):
    """Switch to a different crop model"""
    try:
        load_model_for_crop(crop)
        from src.config import CLASSES_DICT
        classes = CLASSES_DICT.get(crop, [])
        accuracy_map = {"tomato": "89%", "potato": "95.43%", "grape": "97.53%", "apple": "94.32%"}
        
        return {
            "status": "success",
            "crop": crop,
            "classes": len(classes),
            "accuracy": accuracy_map.get(crop, "N/A"),
            "message": f"Switched to {crop} model"
        }
    except Exception as e:
        logger.error(f"Failed to switch to {crop} model: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to switch model: {str(e)}")


@app.get("/info")
async def get_info():
    """Get API and model information"""
    from src.config import IMAGE_SIZES
    crop_image_size = IMAGE_SIZES.get(current_crop, (224, 224))
    
    return {
        "name": "Crop Disease Detection API",
        "version": "1.0.0",
        "classes": CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_architecture": "MobileNetV2",
        "input_size": crop_image_size,
        "explainability": "Grad-CAM"
    }


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Collect user feedback for continuous improvement."""
    log_feedback(feedback.dict())
    return {"status": "received", "message": "Thank you for your feedback!"}


if __name__ == "__main__":
    print("=" * 70)
    print("CROP DISEASE DETECTION API")
    print("=" * 70)
    print(f"Starting server on {API_HOST}:{API_PORT}")
    print(f"API docs: http://localhost:{API_PORT}/docs")
    print(f"Alternative docs: http://localhost:{API_PORT}/redoc")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
