from flask import Flask, request, jsonify, send_file, render_template
import os
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
import torch
from torchvision import transforms
import cv2
import numpy as np
from model import model, class_names, suitability_map, device
from gradcam import GradCAM
import math

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ Utility: Fetch satellite image ------------------
def fetch_satellite_image(lat, lon):
    print("\n🛰 Fetching satellite image...")
    date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    base_url = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
    params = {
        "REQUEST": "GetSnapshot",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "BBOX": f"{lat-1.0},{lon-1.0},{lat+1.0},{lon+1.0}",
        "CRS": "EPSG:4326",
        "WIDTH": 512,
        "HEIGHT": 512,
        "FORMAT": "image/jpeg",
        "TIME": date
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
        image = Image.open(BytesIO(response.content))
        image_path = os.path.join(UPLOAD_FOLDER, f"satellite_{lat}_{lon}.jpg")
        image.save(image_path)
        print(f"✅ Saved satellite image at {image_path}")
        return image_path
    else:
        print("❌ Failed to fetch satellite image")
        print("Response:", response.text[:200])
        return None

# ------------------ Weather utility ------------------
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["current_weather"]
        return {
            "temperature": weather["temperature"],
            "windspeed": weather["windspeed"],
            "winddirection": weather["winddirection"]
        }
    else:
        return None

# ------------------ Inference Suitability Logic ------------------
def hub_height_wind(v_ref, ref_h=10.0, hub_h=80.0, alpha=0.143):
    """Convert 10m wind to 80m hub height (Hellman power law)."""
    return v_ref * (hub_h / ref_h) ** alpha

def check_wind_suitability(v_hub):
    if v_hub >= 6.5:
        return "Good"
    elif v_hub >= 5.5:
        return "Moderate"
    else:
        return "Low"

def check_solar_suitability(temp_c):
    """Temperature proxy for solar potential (rough heuristic)."""
    if 20 <= temp_c <= 35:
        return "Good"
    elif 15 <= temp_c < 20 or 35 < temp_c <= 40:
        return "Moderate"
    else:
        return "Low"

def evaluate_suitability(lat, lon):
    """Evaluate wind and solar suitability."""
    weather = get_weather(lat, lon)
    if not weather:
        return None
    temp_c = weather["temperature"]
    wind_speed = weather["windspeed"]
    wind_dir = weather["winddirection"]
    v_hub = hub_height_wind(wind_speed)
    wind_level = check_wind_suitability(v_hub)
    solar_level = check_solar_suitability(temp_c)
    if wind_level == "Good" and solar_level == "Good":
        label = "✅ Suitable for both Wind & Solar"
    elif wind_level == "Good":
        label = "💨 Suitable for Wind Energy"
    elif solar_level == "Good":
        label = "☀️ Suitable for Solar Energy"
    else:
        label = "⚠️ Not Suitable for Renewable Setup"
    return {
        "temp_c": temp_c,
        "wind_speed_10m": wind_speed,
        "wind_dir_deg": wind_dir,
        "wind_speed_80m": round(v_hub, 2),
        "wind_level": wind_level,
        "solar_level": solar_level,
        "label": label
    }

# ------------------ Transform & GradCAM utility ------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def generate_gradcam(image_path, save_path="gradcam_output.jpg"):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    suitability = suitability_map[predicted_class]
    gradcam = GradCAM(model, model.layer4[2].conv3)
    mask = gradcam.generate(input_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(img.resize((224,224))) / 255.0
    overlay = 0.5 * heatmap/255 + 0.5 * img_np
    result = np.uint8(255 * overlay)
    gradcam_filename = f"gradcam_{os.path.basename(image_path)}"
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return predicted_class, suitability, gradcam_path

# ------------------ Flask Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json if request.is_json else request.form
    lat = data.get("latitude")
    lon = data.get("longitude")
    if not lat or not lon:
        return jsonify({"error": "Please provide 'latitude' and 'longitude'"}), 400
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude format"}), 400
    # Step 1: Fetch satellite image
    image_path = fetch_satellite_image(lat, lon)
    if not image_path:
        return jsonify({"error": "Failed to retrieve satellite image"}), 500
    # Step 2: Run prediction and GradCAM
    predicted_class, suitability, gradcam_path = generate_gradcam(image_path)
    # Step 2.5: Fetch weather and suitability info
    weather_info = get_weather(lat, lon)
    if not weather_info:
        weather_info = {"temperature": None, "windspeed": None, "winddirection": None}
    suitability_info = evaluate_suitability(lat, lon)
    if not suitability_info:
        suitability_info = {
            "temp_c": None,
            "wind_speed_10m": None,
            "wind_dir_deg": None,
            "wind_speed_80m": None,
            "wind_level": "Unknown",
            "solar_level": "Unknown",
            "label": "⚠️ Unable to evaluate suitability"
        }
    # Step 3: Return response
    return jsonify({
        "latitude": lat,
        "longitude": lon,
        "predicted_class": predicted_class,
        "suitability": suitability,
        "satellite_image": f"/image/{os.path.basename(image_path)}",
        "gradcam_image": f"/gradcam/{os.path.basename(gradcam_path)}",
        "weather": weather_info,
        "suitability_info": suitability_info
    })

@app.route("/image/<filename>")
def get_image(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype="image/jpeg")

@app.route("/gradcam/<filename>")
def get_gradcam(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)