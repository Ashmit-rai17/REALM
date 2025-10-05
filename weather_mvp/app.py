import requests

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = data["current_weather"]
        print(f"🌡 Temperature: {weather['temperature']}°C")
        print(f"💨 Wind Speed: {weather['windspeed']} km/h")
        print(f"🧭 Wind Direction: {weather['winddirection']}°")
    else:
        print("❌ Error fetching weather data")

if __name__ == "__main__":
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))
    get_weather(lat, lon)
