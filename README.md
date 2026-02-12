# Crop Disease Prediction

ML-powered crop disease detection using deep learning, with an API, Streamlit UI, explainability, and per-crop training pipelines. This repository contains end-to-end training, evaluation, and inference for multiple crops based on the PlantVillage dataset.

## What This Project Does

This project trains and serves image classification models to identify crop diseases from leaf images. It includes:

- A Streamlit web app for interactive predictions.
- A backend API for programmatic access to predictions.
- Grad-CAM visual explanations to highlight model attention.
- Treatment recommendations per disease.
- Per-crop and multi-crop training pipelines.
- Accuracy validation scripts.

## Features

- Multi-crop disease prediction with dedicated models per crop.
- Support for multiple dataset variants (color, grayscale, segmented).
- Configurable confidence thresholds for safer predictions.
- Explainability via Grad-CAM heatmaps.
- Disease-to-treatment mapping stored in JSON files.
- API service for integration with other applications.
- Streamlit UI for image upload and results visualization.
- Automated verification and test scripts.
- Centralized configuration utilities.
- Logging of predictions to JSONL for auditability.

## Repository Structure

- [app.py](app.py) - Streamlit frontend for uploading images and viewing predictions.
- [main.py](main.py) - API backend entry point.
- [src/](src/) - Core logic, configuration, Grad-CAM utilities, and translations.
- [models/](models/) - Trained models and class mappings per crop.
- [data/](data/) - PlantVillage dataset structure (color, grayscale, segmented).
- [logs/](logs/) - Prediction logs.
- Training scripts (one per crop and combined):
	- [train.py](train.py)
	- [train_all_crops.py](train_all_crops.py)
	- [train_apple.py](train_apple.py)
	- [train_corn.py](train_corn.py)
	- [train_grape.py](train_grape.py)
	- [train_grape_corn.py](train_grape_corn.py)
	- [train_peach.py](train_peach.py)
	- [train_pepper.py](train_pepper.py)
	- [train_potato.py](train_potato.py)
	- [train_potato_enhanced.py](train_potato_enhanced.py)
	- [train_strawberry.py](train_strawberry.py)
	- [train_strawberry_fast.py](train_strawberry_fast.py)
	- [train_tomato_fast.py](train_tomato_fast.py)
	- [train_winning_model.py](train_winning_model.py)
- Testing and validation:
	- [test_prediction.py](test_prediction.py)
	- [test_api.py](test_api.py)
	- [test_apple_accuracy.py](test_apple_accuracy.py)
	- [test_corn_accuracy.py](test_corn_accuracy.py)

## Models and Metadata

The [models/](models/) folder includes:

- Trained model files (`.h5` / `.keras`) for each crop.
- Class mapping JSON files (label index to class name).
- Treatment recommendations per crop in JSON.

## Explainability (Grad-CAM)

Grad-CAM visualizations are supported to show which image regions most influenced the prediction. See [src/grad_cam.py](src/grad_cam.py) for the current implementation. Older or experimental versions are preserved in [src/grad_cam_old.py](src/grad_cam_old.py) and [src/grad_cam_broken.py](src/grad_cam_broken.py).

## Logging

Prediction events can be logged to [logs/predictions.jsonl](logs/predictions.jsonl) for auditing, monitoring, or dataset improvement.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

```bash
python main.py
```

You can also use [start_api.bat](start_api.bat) on Windows or [start_api.sh](start_api.sh) on Linux/macOS.

## Run the Streamlit App

```bash
streamlit run app.py
```

Open the app at `http://localhost:8501`.

## Tests and Verification

```bash
python test_prediction.py
python test_api.py
python test_apple_accuracy.py
python test_corn_accuracy.py
python verify_project.py
```

## Dataset

This project expects the PlantVillage dataset to be placed under [data/plantvillage dataset/](data/plantvillage%20dataset/), organized by crop and disease class folders.

## Supported Crops (Current Models)

- Apple
- Corn
- Grape
- Peach
- Pepper
- Potato
- Strawberry
- Tomato

## Notes

- If you retrain models, ensure class mapping JSON files are updated.
- For multi-crop classification, use the combined training script and mapping in [models/class_mapping_all_crops.json](models/class_mapping_all_crops.json).

