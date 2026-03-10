from __future__ import annotations

import json
import os
import io
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
import requests
import torch
from PIL import Image
from torchvision import transforms

# --- Environment Configuration ---
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

# HF API endpoints (used when selected plant is not in local training classes)
# HF API endpoints — router format (used when selected plant is not in local training classes)
# Both point to reliable full-PlantVillage models (38 classes)
MODEL_URLS = {
    "vit":  "https://router.huggingface.co/hf-inference/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    "swin": "https://router.huggingface.co/hf-inference/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
}

# Image preprocessing (standard ImageNet normalization for both ViT and Swin)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =============================================================================
#  DATA LOADERS
# =============================================================================

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


def get_local_plants() -> set[str]:
    """Return the set of plant names available in the local model."""
    return {_split_label(name)[0].lower() for name in load_class_names()}


# =============================================================================
#  MODEL LOADING — auto-detects num_classes from checkpoint
# =============================================================================

def _detect_num_classes(checkpoint_path: Path) -> int:
    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    for key in ("head.bias", "swin.head.bias", "vit.heads.head.bias",
                "heads.head.bias", "classifier.bias", "fc.bias"):
        if key in sd:
            return sd[key].shape[0]
    raise ValueError(f"Cannot determine num_classes from {checkpoint_path}")


def _load_checkpoint_into_model(model, checkpoint_path: Path, model_type: str, device):
    sd = torch.load(checkpoint_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    prefix = model_type + "."
    already_prefixed = any(k.startswith(prefix) for k in sd.keys())

    new_sd = sd if already_prefixed else {f"{prefix}{k}": v for k, v in sd.items()}
    missing, _ = model.load_state_dict(new_sd, strict=False)

    if missing:
        head_missing = [k for k in missing if any(x in k for x in ["head", "classifier", "fc"])]
        if head_missing:
            import warnings
            warnings.warn(
                f"[{model_type}] Head weights missing after load: {head_missing}. "
                "Predictions may be unreliable."
            )


_model_cache: dict = {}


def get_local_model(model_type: str, device: torch.device):
    cache_key = (model_type, str(device))
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    checkpoint_path = REPO_ROOT / (
        "vit_plant_disease.pth" if model_type == "vit" else "swin_plant_disease.pth"
    )
    if not checkpoint_path.exists():
        return None

    num_classes = _detect_num_classes(checkpoint_path)
    model = get_vit_model(num_classes) if model_type == "vit" else get_swin_model(num_classes)
    _load_checkpoint_into_model(model, checkpoint_path, model_type, device)
    model.to(device)
    model.eval()
    _model_cache[cache_key] = model
    return model


# =============================================================================
#  LABEL PARSING
# =============================================================================

def _split_label(raw_label: str) -> tuple[str, str]:
    """
    Parses labels in these formats (all used by HF and local models):
      - 'Tomato___Early_blight'     → ('Tomato', 'Early blight')
      - 'Apple___Healthy'           → ('Apple',  'Healthy')
      - 'Tomato Early_blight'       → ('Tomato', 'Early blight')
      - 'Tomato healthy'            → ('Tomato', 'Healthy')
    Returns (plant_name, disease_name).
    """
    # 1. Triple-underscore separator (local model / standard PlantVillage format)
    if "___" in raw_label:
        plant_raw, disease_raw = raw_label.split("___", 1)
        plant = plant_raw.strip().replace("_", " ")
        disease = disease_raw.strip().replace("_", " ")
        if disease.lower() == "healthy":
            disease = "Healthy"
        return plant, disease

    # 2. Space-separated 'Plant DiseaseName' (common in HF model labels)
    parts = raw_label.split(" ", 1)
    if len(parts) == 2:
        plant = parts[0].strip()
        disease = parts[1].strip().replace("_", " ")
        if disease.lower() == "healthy":
            disease = "Healthy"
        return plant, disease

    # 3. Fallback: treat whole string as disease, plant unknown
    label_clean = raw_label.strip().replace("_", " ")
    if "healthy" in label_clean.lower():
        return label_clean.replace("healthy", "").replace("Healthy", "").strip() or "Unknown", "Healthy"
    return "Unknown", label_clean


def _lookup_disease_info(plant_name: str, disease: str) -> dict[str, str]:
    """
    Lookup disease details. Tries progressively broader keys:
      1. 'Plant Disease' exact match
      2. 'Disease' alone
      3. 'healthy' or 'default' fallback
    """
    info = load_disease_info()

    if disease.lower() == "healthy":
        return info.get("healthy", info.get("default", {}))

    # Exact key: 'Plant Disease'
    key1 = f"{plant_name} {disease}"
    if key1 in info:
        return info[key1]

    # Disease only
    if disease in info:
        return info[disease]

    # Partial match: find the first key that contains the disease name
    disease_lower = disease.lower()
    for k, v in info.items():
        if disease_lower in k.lower():
            return v

    return info.get("default", {})


# =============================================================================
#  DISEASE VALIDATION — prevent healthy classification when lesions are visible
# =============================================================================

def _anti_healthy_override(
    probabilities: torch.Tensor,
    class_names: list[str],
    selected_plant_lower: str
) -> tuple[int, float]:
    """
    Within the plant's class slice, apply the disease-preference rule:
    If the sum of disease-class probabilities > healthy probability,
    return the top disease class instead of healthy.
    """
    plant_indices = [
        i for i, n in enumerate(class_names)
        if _split_label(n)[0].lower() == selected_plant_lower
    ]
    if not plant_indices:
        _, idx = torch.max(probabilities, 0)
        return idx.item(), probabilities[idx.item()].item()

    # Split into healthy and disease sub-groups
    healthy_idx = [i for i in plant_indices if "healthy" in class_names[i].lower()]
    disease_idx = [i for i in plant_indices if "healthy" not in class_names[i].lower()]

    healthy_prob = sum(probabilities[i].item() for i in healthy_idx)
    disease_prob = sum(probabilities[i].item() for i in disease_idx)

    if disease_idx and disease_prob > healthy_prob:
        # Return the highest-probability disease class
        best_disease = max(disease_idx, key=lambda i: probabilities[i].item())
        return best_disease, probabilities[best_disease].item()
    else:
        # Return the best plant-restricted class overall
        best = max(plant_indices, key=lambda i: probabilities[i].item())
        return best, probabilities[best].item()


# =============================================================================
#  MAIN INFERENCE ENTRY POINT
# =============================================================================

def predict_image(
    image_bytes: bytes,
    model_type: str = "vit",
    model_path: str = "vit_plant_disease.pth",
    target_plant_name: str = ""
) -> dict[str, Any]:
    """
    Two-stage pipeline:
      Stage 1 — Plant selection:
        • If user provides a plant name, always use it as ground truth.
        • If the plant exists in the local model classes, use local inference.
        • Otherwise, route to the HF API (which knows 38+ plants).
      Stage 2 — Disease classification:
        • Restrict predictions to diseases of the selected plant.
        • Anti-healthy override: if disease probabilities > healthy, return disease.
    """
    target_plant_name = target_plant_name.strip()
    target_lower = target_plant_name.lower()
    local_plants = get_local_plants()

    # ── Decide which inference path to use ───────────────────────────────────
    # Use local model only if:
    #   a) No plant name given (let model decide from known classes), OR
    #   b) The user-specified plant IS in the local training set
    use_local = (not target_lower) or (target_lower in local_plants)

    # ── LOCAL INFERENCE ───────────────────────────────────────────────────────
    if use_local:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = load_class_names()
        model = get_local_model(model_type, device) if class_names else None

        if model and class_names:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)

            # Stage 1: determine plant
            if target_lower and target_lower in local_plants:
                selected_plant_lower = target_lower
                # Compute which plant model predicted to check for mismatch
                plant_probs: dict[str, float] = {}
                for i, name in enumerate(class_names):
                    p, _ = _split_label(name)
                    plant_probs[p.lower()] = plant_probs.get(p.lower(), 0.0) + probs[i].item()
                ai_plant_lower = max(plant_probs, key=plant_probs.get)
                warning_msg = (
                    f"Note: You selected '{target_plant_name}', but the AI model detected "
                    f"'{ai_plant_lower.title()}'. Consider uploading a clearer image."
                    if ai_plant_lower != selected_plant_lower else None
                )
            else:
                # Let model decide
                plant_probs = {}
                for i, name in enumerate(class_names):
                    p, _ = _split_label(name)
                    plant_probs[p.lower()] = plant_probs.get(p.lower(), 0.0) + probs[i].item()
                selected_plant_lower = max(plant_probs, key=plant_probs.get)
                warning_msg = None

            # Stage 2: disease with anti-healthy override
            best_idx, conf = _anti_healthy_override(probs, class_names, selected_plant_lower)
            label = class_names[best_idx]
            plant_name, disease = _split_label(label)

            # Always display user-provided plant name if given
            if target_plant_name:
                plant_name = target_plant_name.title()

            details = _lookup_disease_info(plant_name, disease)
            message = "Low confidence — please upload a clearer, well-lit image." if conf < 0.70 else None

            return {
                "plant_name": plant_name,
                "disease": disease,
                "confidence": round(conf * 100, 2),
                "cause": details.get("cause", "No data available."),
                "cure": details.get("cure", "No data available."),
                "prevention": details.get("prevention", "No data available."),
                "source": "local",
                "warning": warning_msg,
                "message": message,
            }

    # ── HF API INFERENCE (plant not in local classes) ─────────────────────────
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise RuntimeError(
            f"Plant '{target_plant_name}' is not supported by the local model, "
            "and HF_API_KEY is not set. Cannot make a prediction."
        )

    url = MODEL_URLS.get(model_type, MODEL_URLS["swin"])
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "image/jpeg",
        "X-Wait-For-Model": "true",  # wait up to 30s for model warm-up
    }

    # Retry up to 3 times to handle cold-start 503s
    result = None
    last_error = ""
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=image_bytes, timeout=45)
            # Check for HTML response (error page)
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type or response.text.strip().startswith("<"):
                last_error = f"HF API returned an HTML error page (status {response.status_code})"
                import time
                time.sleep(5)
                continue
            if response.status_code == 503:
                last_error = f"HF API model loading (503), attempt {attempt+1}/3"
                import time
                time.sleep(8)
                continue
            if response.status_code != 200:
                last_error = f"HF API error {response.status_code}: {response.text[:200]}"
                break
            result = response.json()
            if isinstance(result, dict) and "error" in result:
                last_error = f"HF API error: {result['error']}"
                result = None
                import time
                time.sleep(5)
                continue
            break  # success
        except requests.exceptions.Timeout:
            last_error = f"HF API request timed out (attempt {attempt+1}/3)"
            import time
            time.sleep(3)

    if result is None:
        raise RuntimeError(f"HF API unavailable after 3 attempts: {last_error}")

    # Filter predictions to the user's selected plant
    matching = [
        r for r in result
        if target_lower in r.get("label", "").lower()
    ]
    # Fall back to all results if none match the plant name
    candidates = matching if matching else result

    # Apply anti-healthy preference: if disease candidates exist and their
    # cumulative probability > healthy, pick best disease
    healthy_entries = [r for r in candidates if "healthy" in r.get("label", "").lower()]
    disease_entries = [r for r in candidates if "healthy" not in r.get("label", "").lower()]

    healthy_prob = sum(r.get("score", 0) for r in healthy_entries)
    disease_prob = sum(r.get("score", 0) for r in disease_entries)

    if disease_entries and disease_prob > healthy_prob:
        top = max(disease_entries, key=lambda r: r.get("score", 0))
    else:
        top = candidates[0]  # Already sorted by score descending

    label = top.get("label", "Unknown")
    conf = top.get("score", 0.0)

    api_plant, disease = _split_label(label)
    # Always show user-requested plant name
    plant_name = target_plant_name.title() if target_plant_name else api_plant

    details = _lookup_disease_info(plant_name, disease)
    message = "Low confidence — please upload a clearer, well-lit image." if conf < 0.70 else None

    return {
        "plant_name": plant_name,
        "disease": disease,
        "confidence": round(conf * 100, 2),
        "cause": details.get("cause", "No data available."),
        "cure": details.get("cure", "No data available."),
        "prevention": details.get("prevention", "No data available."),
        "source": "hf_api",
        "warning": None,
        "message": message,
    }
