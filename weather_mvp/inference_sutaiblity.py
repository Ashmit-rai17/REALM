
"""
inference_suitability.py
---------------------------------
Takes latitude & longitude as input,
fetches weather (temp, wind speed, direction) using Open-Meteo API,
and outputs:
1. Satellite image of the area
2. Weather summary
3. Suitability label: Wind / Solar / Not Suitable

"""

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import math

# ---------- SETTINGS ----------
MAP_ZOOM = 12          # satellite zoom level (higher = more detailed)
MAP_SIZE = 500         # pixels per side
MAP_PROVIDER = "https://maps.geoapify.com/v1/staticmap"
API_KEY = "f874c3536d294eef8675ae0e974c6532"  

# ---------- WEATHER FETCH ----------
def get_weather(lat, lon):
    """Fetch temperature, wind speed, direction from Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    res = requests.get(url, timeout=10).json()
    weather = res["current_weather"]
    return {
        "temp_c": weather["temperature"],
        "wind_speed": weather["windspeed"],
        "wind_dir": weather["winddirection"]
    }

# ---------- SATELLITE IMAGE ----------
def get_satellite_image(lat, lon):
    """Fetch static satellite image from Geoapify with debugging info."""
    url = (f"{MAP_PROVIDER}?style=satellite&center=lonlat:{lon},{lat}"
           f"&zoom={MAP_ZOOM}&size={MAP_SIZE}x{MAP_SIZE}&apiKey={API_KEY}")
    print("🛰️ Fetching from:", url)
    res = requests.get(url, timeout=10)

    # Check if we actually got an image
    print("Status code:", res.status_code)
    print("Content-Type:", res.headers.get("Content-Type", ""))

    if res.status_code == 200 and res.headers.get("Content-Type", "").startswith("image"):
        return Image.open(BytesIO(res.content))
    else:
        print("⚠️ Geoapify returned non-image content:")
        print(res.text[:300])  # print start of the response
        return None


# ---------- SUITABILITY RULES ----------
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

def evaluate(lat, lon):
    """Full evaluation pipeline."""
    weather = get_weather(lat, lon)
    temp_c = weather["temp_c"]
    wind_speed = weather["wind_speed"]
    wind_dir = weather["wind_dir"]

    # Convert to hub height
    v_hub = hub_height_wind(wind_speed)

    # Get suitability levels
    wind_level = check_wind_suitability(v_hub)
    solar_level = check_solar_suitability(temp_c)

    # Final label logic
    if wind_level == "Good" and solar_level == "Good":
        label = "✅ Suitable for both Wind & Solar"
    elif wind_level == "Good":
        label = "💨 Suitable for Wind Energy"
    elif solar_level == "Good":
        label = "☀️ Suitable for Solar Energy"
    else:
        label = "⚠️ Not Suitable for Renewable Setup"

    return {
        "lat": lat,
        "lon": lon,
        "temp_c": temp_c,
        "wind_speed_10m": wind_speed,
        "wind_dir_deg": wind_dir,
        "wind_speed_80m": round(v_hub, 2),
        "wind_level": wind_level,
        "solar_level": solar_level,
        "label": label
    }

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    print("=== Renewable Energy Site Suitability Predictor ===")
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))

    print("\nFetching satellite image and weather data...")
    image = get_satellite_image(lat, lon)
    result = evaluate(lat, lon)
    
    # Display results
    print("\n--- Weather Info ---")
    print(f"Temperature: {result['temp_c']} °C")
    print(f"Wind Speed (10m): {result['wind_speed_10m']} m/s")
    print(f"Wind Speed (80m Hub): {result['wind_speed_80m']} m/s")
    print(f"Wind Direction: {result['wind_dir_deg']}°")

    print("\n--- Suitability ---")
    print(f"Wind: {result['wind_level']}")
    print(f"Solar: {result['solar_level']}")
    print(f"Final Verdict: {result['label']}")

    # Show image
    image.show()
