import torch
import torch.nn as nn
from torchvision.models import resnet50
import os
import urllib.request

# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model definition
# -----------------------------
num_classes = 10  # EuroSAT has 10 classes
model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# -----------------------------
# Auto-download weights if missing
# -----------------------------
WEIGHTS_PATH = "resnet50_eurosat.pth"
# 🔗 Replace this with your actual model URL (e.g. Google Drive direct link or Hugging Face)
MODEL_URL = "https://drive.google.com/uc?id=1WeyCyuPq3g8SOHM-2D24buqiy5IeS8Ow"

if not os.path.exists(WEIGHTS_PATH):
    print("📥 Model weights not found. Downloading...")
    try:
        urllib.request.urlretrieve(MODEL_URL, WEIGHTS_PATH)
        print("✅ Download complete.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download model weights.\nError: {e}")

# -----------------------------
# Load weights
# -----------------------------
try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print("✅ Model weights loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model weights.\nError: {e}")

# -----------------------------
# Move model to device
# -----------------------------
model.to(device)
model.eval()

# -----------------------------
# EuroSAT Class Names
# -----------------------------
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# -----------------------------
# Suitability Mapping
# -----------------------------
suitability_map = {
    "AnnualCrop": "Suitable",
    "HerbaceousVegetation": "Suitable",
    "Pasture": "Suitable",
    "Industrial": "Suitable",
    "PermanentCrop": "Suitable",
    "Forest": "Not Suitable",
    "Residential": "Not Suitable",
    "Highway": "Not Suitable",
    "River": "Not Suitable",
    "SeaLake": "Not Suitable"
}

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        suitability = suitability_map[predicted_class]
    return predicted_class, suitability
