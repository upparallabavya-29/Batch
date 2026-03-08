from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DISEASE_INFO_PATH = REPO_ROOT / "utils" / "disease_info.json"

MODEL_URLS = {
    "vit": "https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "swin": "https://api-inference.huggingface.co/models/100xFORTUNE/plant_disease_classification"
}

@lru_cache(maxsize=1)
def load_disease_info() -> dict[str, dict[str, str]]:
    if DISEASE_INFO_PATH.exists():
        with DISEASE_INFO_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _split_label(raw_label: str) -> tuple[str, str]:
    if "___" in raw_label:
        plant, disease = raw_label.split("___", 1)
    elif "_" in raw_label:
        parts = raw_label.split("_", 1)
        if len(parts) > 1:
            plant, disease = parts[0], parts[1]
        else:
             plant, disease = "Unknown", raw_label
    else:
        plant, disease = "Unknown", raw_label
    return plant.replace("_", " "), disease.replace("_", " ")

def predict_image(image_bytes: bytes, model_type: str = "vit", model_path: str = "vit_plant_disease.pth") -> dict[str, Any]:
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise RuntimeError("HF_API_KEY environment variable is not set. Please add it to your .env file.")

    url = MODEL_URLS.get(model_type, MODEL_URLS["vit"])
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    response = requests.post(url, headers=headers, data=image_bytes)
    
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API returned status {response.status_code}: {response.text}")
        
    result = response.json()
    
    # HF usually returns a list of dicts: [{"label": "Healthy Apple", "score": 0.99}, ...]
    if isinstance(result, list) and len(result) > 0:
        top_prediction = result[0]
        label = top_prediction.get("label", "Unknown")
        conf = top_prediction.get("score", 0.0)
    elif isinstance(result, dict) and "error" in result:
         raise RuntimeError(f"Hugging Face API error: {result['error']}")
    else:
         raise RuntimeError(f"Unexpected response format from Hugging Face API: {result}")

    plant_name, disease = _split_label(label)

    info = load_disease_info()
    disease_key = "healthy" if "healthy" in disease.lower() else "default"
    details = info.get(disease_key, info.get("default", {}))

    return {
        "plant_name": plant_name,
        "disease": disease,
        "confidence": round(conf * 100, 2),
        "cause": details.get("cause", "No data available."),
        "cure": details.get("cure", "No data available."),
        "prevention": details.get("prevention", "No data available."),
    }
