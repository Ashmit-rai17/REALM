import torch
import torch.nn as nn

# Load your trained ResNet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10  # EuroSAT has 10 classes
model = torch.hub.load('pytorch/vision:v0.14.0', 'resnet50', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_eurosat.pth", map_location=device))
model.to(device)
model.eval()

# Class names from EuroSAT dataset
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# Define mapping to suitability
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

# Function to predict class + suitability
def predict(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        suitability = suitability_map[predicted_class]
    return predicted_class, suitability
