from __future__ import annotations

import json
import os
import io
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
import requests
import torch
from PIL import Image
from torchvision import transforms

# --- Environment Configuration ---
# Redirect temporary files and model cache to E: drive to avoid C: drive space issues
REPO_ROOT = Path(__file__).resolve().parents[1]
E_TEMP_DIR = REPO_ROOT / "temp"
E_TEMP_DIR.mkdir(exist_ok=True)

os.environ["TMPDIR"] = str(E_TEMP_DIR)
os.environ["TEMP"] = str(E_TEMP_DIR)
os.environ["TMP"] = str(E_TEMP_DIR)
os.environ["TORCH_HOME"] = str(REPO_ROOT / "torch_cache")
tempfile.tempdir = str(E_TEMP_DIR)

# --- Path Definitions ---
DISEASE_INFO_PATH = REPO_ROOT / "utils" / "disease_info.json"
CLASS_NAMES_PATH = REPO_ROOT / "class_names.json"

# Import model definitions
from models.vit_model import get_vit_model
from models.swin_model import get_swin_model

MODEL_URLS = {
    "vit": "https://router.huggingface.co/hf-inference/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "swin": "https://router.huggingface.co/hf-inference/models/gianlab/swin-tiny-patch4-window7-224-finetuned-plantdisease"
}

@lru_cache(maxsize=1)
def load_disease_info() -> dict[str, dict[str, str]]:
    if DISEASE_INFO_PATH.exists():
        with DISEASE_INFO_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _detect_num_classes(checkpoint_path: Path) -> int:
    """Inspect checkpoint's head/classifier layer to get the true num_classes."""
    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Look for the output (bias) tensor of the classification head
    for key in ("head.bias", "swin.head.bias", "vit.heads.head.bias",
                "heads.head.bias", "classifier.bias", "fc.bias"):
        if key in sd:
            return sd[key].shape[0]
    raise ValueError(f"Cannot determine num_classes from checkpoint {checkpoint_path}")


def _load_checkpoint_into_model(model, checkpoint_path: Path, model_type: str, device):
    """Load checkpoint, handling wrapped vs unwrapped key prefixes."""
    sd = torch.load(checkpoint_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Determine whether the checkpoint already uses wrapper prefix
    prefix = model_type + "."  # e.g. 'swin.' or 'vit.'
    already_prefixed = any(k.startswith(prefix) for k in sd.keys())

    if already_prefixed:
        new_sd = sd
    else:
        # Checkpoint was saved WITHOUT wrapper – add the wrapper prefix
        new_sd = {f"{prefix}{k}": v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        # Critical if the head is missing — the predictions will be wrong
        head_missing = [k for k in missing if "head" in k or "classifier" in k]
        if head_missing:
            import warnings
            warnings.warn(
                f"[{model_type}] Head weights missing after load: {head_missing}. "
                "Predictions may be unreliable."
            )


_model_cache: dict = {}

def get_local_model(model_type: str, device: torch.device):
    """Build and cache a model, auto-detecting num_classes from the checkpoint."""
    cache_key = (model_type, str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoint_path = REPO_ROOT / (
        "vit_plant_disease.pth" if model_type == "vit" else "swin_plant_disease.pth"
    )
    if not checkpoint_path.exists():
        return None

    num_classes = _detect_num_classes(checkpoint_path)

    if model_type == "vit":
        model = get_vit_model(num_classes)
    else:
        model = get_swin_model(num_classes)

    _load_checkpoint_into_model(model, checkpoint_path, model_type, device)
    model.to(device)
    model.eval()
    _model_cache[cache_key] = model
    return model

def _split_label(raw_label: str) -> tuple[str, str]:
    """
    Splits labels like 'Healthy___Apple', 'Apple___Apple_scab', or 'Chilli___Healthy'.
    Returns (plant_name, disease_name).
    """
    # 1. Try triple underscore (primary separator)
    if "___" in raw_label:
        parts = raw_label.split("___", 1)
        part1 = parts[0]
        part2 = parts[1]
    
    # 2. Handle 'Healthy' as a special case in any format
    elif "healthy" in raw_label.lower():
        if raw_label.lower().startswith("healthy"):
            part1 = "Healthy"
            part2 = raw_label[7:].strip("_")
        elif raw_label.lower().endswith("healthy"):
            idx = raw_label.lower().rfind("healthy")
            part1 = raw_label[:idx].strip("_")
            part2 = "Healthy"
        else:
            # Contains 'healthy' somewhere in the middle
            part1, part2 = raw_label, "Healthy"
            
    # 3. Fallback: Treat as single plant name, default disease to 'Healthy'
    else:
        part1, part2 = raw_label, "Healthy"

    # Normalize strings
    p1_clean = part1.strip().replace("_", " ")
    p2_clean = part2.strip().replace("_", " ")

    # Identify which part is 'Healthy' for the UI display logic
    if p1_clean.lower() == "healthy":
        plant_name = p2_clean
        disease = "Healthy"
    elif p2_clean.lower() == "healthy":
        plant_name = p1_clean
        disease = "Healthy"
    else:
        plant_name = p1_clean
        disease = p2_clean

    return plant_name, disease

def predict_image(image_bytes: bytes, model_type: str = "vit", model_path: str = "vit_plant_disease.pth", target_plant_name: str = "") -> dict[str, Any]:
    # 1. Try Local Inference First
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names()
    warning_msg = None
    
    if class_names:
        # Get model — auto-detects num_classes from checkpoint internally
        model = get_local_model(model_type, device)
        
        if model:
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # Two-Stage Logic
                plant_probs = {}
                for idx, name in enumerate(class_names):
                    plant, _ = _split_label(name)
                    plant_probs[plant.lower()] = plant_probs.get(plant.lower(), 0.0) + probabilities[idx].item()
                
                predicted_plant_lower = max(plant_probs, key=plant_probs.get)
                predicted_plant = next((_split_label(name)[0] for name in class_names if _split_label(name)[0].lower() == predicted_plant_lower), "Unknown")

                valid_plants = set(_split_label(name)[0].lower() for name in class_names)
                target_lower = target_plant_name.strip().lower()
                
                if target_lower and target_lower in valid_plants:
                    selected_plant_lower = target_lower
                    if target_lower != predicted_plant_lower:
                        warning_msg = f"Warning: You selected {target_plant_name}, but AI detected {predicted_plant}."
                else:
                    selected_plant_lower = predicted_plant_lower

                restricted_probs = probabilities.clone()
                for idx, name in enumerate(class_names):
                    plant, _ = _split_label(name)
                    if plant.lower() != selected_plant_lower:
                        restricted_probs[idx] = -1.0
                
                _, index = torch.max(restricted_probs, 0)
                label = class_names[index.item()]
                conf = probabilities[index.item()].item()

            plant_name, disease = _split_label(label)
            info = load_disease_info()
            disease_key = "healthy" if "healthy" in disease.lower() else "default"
            details = info.get(disease_key, info.get("default", {}))

            message = "Low confidence prediction. Please upload a clearer image." if conf < 0.7 else None

            return {
                "plant_name": plant_name,
                "disease": disease,
                "confidence": round(conf * 100, 2),
                "cause": details.get("cause", "No data available."),
                "cure": details.get("cure", "No data available."),
                "prevention": details.get("prevention", "No data available."),
                "source": "local",
                "warning": warning_msg,
                "message": message
            }

    # 2. Fallback to HF API (if local model missing or fails)
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise RuntimeError("HF_API_KEY environment variable is not set and local model not found.")

    url = MODEL_URLS.get(model_type, MODEL_URLS["vit"])
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "image/jpeg"
    }

    response = requests.post(url, headers=headers, data=image_bytes)
    
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API returned status {response.status_code}: {response.text}")
        
    result = response.json()
    
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
        "source": "hf_api"
    }
