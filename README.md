# 🌱 Crop Disease Detection

ML-powered disease detection for crops using deep learning and explainable AI. Trained on **10,00+ images** from Kaggle PlantVillage dataset.

## Features

- Explainable AI with Grad-CAM heatmaps
- Confidence-based predictions (>70% threshold)
- Symptom integration
- Treatment recommendations
- Multi-crop support ( Potato, Apple, Grape, Corn)
- Web interface with Streamlit

## Model Accuracy

| Crop | Accuracy     |
|------|----------    |
| 🥔 Potato | 95.43% |
| 🍎 Apple | 94.32%  |
| 🍇 Grape | 97.53%  |

## How to Run

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run backend (Terminal 1)
python main.py

# Run frontend (Terminal 2)
streamlit run app.py

# Open browser
http://localhost:8501
```

Test:
```bash
python test_prediction.py
python test_api.py

