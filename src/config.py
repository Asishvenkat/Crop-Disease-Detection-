"""
Configuration and constants for Crop Disease Detection system
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model paths for each crop
MODEL_PATHS = {
    "potato": MODELS_DIR / "potato_model.keras",
    "grape": MODELS_DIR / "pdd_grape.keras",
    "apple": MODELS_DIR / "pdd_apple.keras",
    "corn": MODELS_DIR / "pdd_corn.keras",
}

# Image sizes for each crop (must match training)
IMAGE_SIZES = {
    "potato": (224, 224),
    "grape": (192, 192),   # MobileNetV2
    "apple": (192, 192),   # MobileNetV2
    "corn": (192, 192),    # MobileNetV2
}

# Grad-CAM target layers per crop (must exist in the loaded model)
GRAD_CAM_LAYERS = {
    "potato": "Conv_1",             # MobileNetV2 last conv block
    "grape": "Conv_1",              # MobileNetV2 last conv block
    "apple": "Conv_1",              # MobileNetV2 last conv block
    "corn": "Conv_1",               # MobileNetV2 last conv block
}

# Default model
DEFAULT_CROP = "potato"
MODEL_PATH = MODEL_PATHS[DEFAULT_CROP]
IMAGE_SIZE = IMAGE_SIZES[DEFAULT_CROP]  # Default image size

# Dataset configuration
DATASET_PATH = DATA_DIR / "plantvillage dataset" / "color"
# IMAGE_SIZE is now crop-specific (see IMAGE_SIZES dict above)
BATCH_SIZE = 8
# Fast hackathon mode: 3 epochs, smaller images
EPOCHS = 3
VALIDATION_SPLIT = 0.2

# Inference quality heuristics
BLUR_THRESHOLD = 120.0  # variance of Laplacian
BRIGHTNESS_BOUNDS = (60.0, 190.0)  # grayscale mean bounds
TEMPERATURE = 0.3  # temperature scaling - AGGRESSIVE: lower = much sharper, higher confidences

# Performance optimization
ENABLE_GRADCAM = True  # Set to False to disable heatmaps for speed
GRADCAM_RESOLUTION = (128, 128)  # Reduced resolution for faster heatmap generation

# Classes dictionary for each crop
CLASSES_DICT = {
    "potato": [
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight"
    ],
    "grape": [
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___healthy",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
    ],
    "apple": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy"
    ],
    "corn": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___healthy",
        "Corn_(maize)___Northern_Leaf_Blight",
    ],
}

# Default classes
CLASSES = CLASSES_DICT[DEFAULT_CROP]

# Confidence and decision thresholds
CONFIDENCE_THRESHOLD = 0.45  # Lowered: 45% is acceptable for practical use
LOW_CONFIDENCE_MESSAGE = "Low confidence – please capture a clearer image or consult expert"

# Severity scoring
SEVERITY_THRESHOLDS = {
    "Mild": (0, 40),
    "Moderate": (40, 70),
    "Severe": (70, 100)
}

# Disease metadata and treatment recommendations for all 38 classes
DISEASE_TREATMENTS = {
    "Apple___Apple_scab": {
        "cause": "Fungal infection (Venturia inaequalis)",
        "symptoms": "Dark, scabby lesions on leaves and fruit",
        "organic_treatment": ["Sulfur spray", "Remove infected leaves", "Improve air flow"],
        "chemical_treatment": ["Captan", "Dodine", "Myclobutanil"],
        "prevention": ["Disease-resistant varieties", "Sanitation", "Pruning for air circulation"],
        "confidence_boost": True
    },
    "Apple___Black_rot": {
        "cause": "Fungal infection (Botryosphaeria obtusa)",
        "symptoms": "Black, sunken lesions on fruit and shoots",
        "organic_treatment": ["Copper spray", "Remove infected branches", "Prune cankers"],
        "chemical_treatment": ["Thiophanate-methyl", "Captan"],
        "prevention": ["Prune infected wood", "Improve drainage", "Avoid wounding trees"],
        "confidence_boost": True
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Fungal infection (Gymnosporangium juniperi-virginianae)",
        "symptoms": "Yellow spots on leaves with tubular projections",
        "organic_treatment": ["Sulfur", "Remove infected leaves", "Distance from juniper"],
        "chemical_treatment": ["Propiconazole", "Myclobutanil"],
        "prevention": ["Resistant varieties", "Remove juniper hosts", "Fungicide spray"],
        "confidence_boost": True
    },
    "Apple___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Blueberry___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Cherry_(including_sour)___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "cause": "Fungal infection (Podosphaera clandestina)",
        "symptoms": "White powdery coating on leaves",
        "organic_treatment": ["Sulfur spray", "Baking soda solution", "Improve air circulation"],
        "chemical_treatment": ["Myclobutanil", "Sulfur", "Thiophanate-methyl"],
        "prevention": ["Good air flow", "Remove infected material", "Avoid overhead watering"],
        "confidence_boost": True
    },
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
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cause": "Bacterial infection (Candidatus Liberibacter asiaticus)",
        "symptoms": "Yellowing of leaves, fruit discoloration, blotchy appearance",
        "organic_treatment": ["Remove infected trees", "Control insect vectors", "Improve sanitation"],
        "chemical_treatment": ["No curative chemical treatment", "Preventative insect control"],
        "prevention": ["Remove infected trees", "Certified disease-free nursery stock"],
        "confidence_boost": True
    },
    "Peach___Bacterial_spot": {
        "cause": "Bacterial infection (Xanthomonas species)",
        "symptoms": "Small dark spots with yellow halos on leaves and fruit",
        "organic_treatment": ["Copper spray", "Remove infected branches", "Improve drainage"],
        "chemical_treatment": ["Copper-based bactericide", "Oxytetracycline"],
        "prevention": ["Resistant varieties", "Sanitation", "Avoid overhead watering"],
        "confidence_boost": True
    },
    "Peach___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Bacterial infection (Xanthomonas species)",
        "symptoms": "Small dark spots with yellow halo",
        "organic_treatment": ["Copper spray", "Remove infected leaves", "Improve drainage"],
        "chemical_treatment": ["Copper-based bactericide"],
        "prevention": ["Disease-free seeds", "Sanitation", "Avoid overhead watering"],
        "confidence_boost": True
    },
    "Pepper,_bell___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Potato___Early_blight": {
        "cause": "Fungal infection (Alternaria solani) - typically appears as brown concentric rings on lower leaves",
        "symptoms": "Brown spots with concentric rings, yellowing leaves",
        "organic_treatment": [
            "Remove infected leaves immediately",
            "Apply copper-based fungicide",
            "Ensure good air circulation",
            "Mulch to prevent soil splash"
        ],
        "chemical_treatment": [
            "Chlorothalonil spray (follow label directions)",
            "Mancozeb fungicide",
            "Apply every 7-10 days"
        ],
        "prevention": [
            "Crop rotation (3-year minimum)",
            "Remove plant debris",
            "Avoid overhead watering",
            "Space plants for air flow"
        ],
        "confidence_boost": True
    },



    "Potato___Early_blight": {
        "cause": "Fungal infection (Alternaria solani)",
        "symptoms": "Brown spots with concentric rings, yellowing leaves",
        "organic_treatment": ["Remove infected leaves", "Copper fungicide", "Good drainage"],
        "chemical_treatment": ["Chlorothalonil", "Mancozeb"],
        "prevention": ["Crop rotation", "Remove plant debris", "Resistant varieties"],
        "confidence_boost": True
    },
    "Potato___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Potato___Late_blight": {
        "cause": "Oomycete pathogen (Phytophthora infestans)",
        "symptoms": "Water-soaked lesions, white growth on underside, rapid spread",
        "organic_treatment": ["Copper spray", "Remove infected leaves", "Improve air flow"],
        "chemical_treatment": ["Metalaxyl", "Mancozeb"],
        "prevention": ["Resistant varieties", "Avoid overhead watering", "Mulch"],
        "confidence_boost": True
    },
    "Raspberry___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Soybean___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Squash___Powdery_mildew": {
        "cause": "Fungal infection (Podosphaera xanthii)",
        "symptoms": "White powdery coating on leaves and stems",
        "organic_treatment": ["Sulfur spray", "Baking soda", "Neem oil"],
        "chemical_treatment": ["Myclobutanil", "Sulfur"],
        "prevention": ["Good air circulation", "Remove infected parts", "Avoid overhead watering"],
        "confidence_boost": True
    },
    "Strawberry___healthy": {
        "cause": "No disease detected",
        "symptoms": "Healthy plant",
        "organic_treatment": ["Continue regular care"],
        "chemical_treatment": ["No treatment needed"],
        "prevention": ["Maintain good practices"],
        "confidence_boost": False
    },
    "Strawberry___Leaf_scorch": {
        "cause": "Fungal infection (Diplocarpon earlianum)",
        "symptoms": "Red to purple spots on leaves and runners",
        "organic_treatment": ["Remove infected leaves", "Copper sulfate", "Good drainage"],
        "chemical_treatment": ["Myclobutanil", "Sulphur"],
        "prevention": ["Remove infected runners", "Proper spacing", "Mulching"],
        "confidence_boost": True
    },

}

# Symptom checklist for decision fusion
SYMPTOM_DISEASE_MAPPING = {
    "yellowing_leaves": ["Potato___Early_blight"],
    "brown_spots": ["Potato___Early_blight"],
    "wilting": ["Potato___Early_blight"],
    "white_fungal_growth": []
}

# Confidence reduction for symptom contradictions
SYMPTOM_CONTRADICTION_PENALTY = 0.12  # 12% reduction

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = False

# Streamlit configuration
STREAMLIT_PAGE_TITLE = "Crop Disease Detection"
STREAMLIT_LAYOUT = "wide"

print("[OK] Configuration loaded successfully")
