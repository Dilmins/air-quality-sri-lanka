from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
from datetime import datetime
from typing import Optional, Dict
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import threading
import time
import os

# ----------------------------------------------------------------------------- 
# FLASK APP
# -----------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder="../Frontend/templates",
    static_folder="../Frontend/static"
)
CORS(app)

print("🚀 IAQ System booting")

# ----------------------------------------------------------------------------- 
# CONFIG (ENV ONLY — NO HARDCODED SECRETS)
# -----------------------------------------------------------------------------

API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not set")

CITIES = {
    "Colombo": {"lat": 6.9271, "lon": 79.8612},
    "Kandy": {"lat": 7.2906, "lon": 80.6337},
    "Galle": {"lat": 6.0535, "lon": 80.2210},
    "Jaffna": {"lat": 9.6615, "lon": 80.0255},
    "Negombo": {"lat": 7.2008, "lon": 79.8358},
    "Trincomalee": {"lat": 8.5874, "lon": 81.2152},
    "Batticaloa": {"lat": 7.7310, "lon": 81.6747},
    "Matara": {"lat": 5.9549, "lon": 80.5550},
    "Anuradhapura": {"lat": 8.3114, "lon": 80.4037},
    "Kurunegala": {"lat": 7.4863, "lon": 80.3623},
    "Ratnapura": {"lat": 6.6828, "lon": 80.3992},
    "Badulla": {"lat": 6.9934, "lon": 81.0550},
    "Nuwara Eliya": {"lat": 6.9497, "lon": 80.7891},
}

current_city = "Colombo"

# ----------------------------------------------------------------------------- 
# GLOBAL STATE (THREAD-SAFE)
# -----------------------------------------------------------------------------

latest_data = {}
data_lock = threading.Lock()

# ----------------------------------------------------------------------------- 
# DATA FETCH (HTTPS ONLY)
# -----------------------------------------------------------------------------

def fetch_outdoor_aqi(lat: float, lon: float) -> Optional[Dict]:
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()["list"][0]
    return {
        "aqi": d["main"]["aqi"],
        "pm25": d["components"]["pm2_5"],
        "pm10": d["components"]["pm10"],
        "no2": d["components"]["no2"],
        "o3": d["components"]["o3"],
        "co": d["components"]["co"],
    }

def fetch_weather(lat: float, lon: float) -> Optional[Dict]:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()
    return {
        "temp": d["main"]["temp"],
        "humidity": d["main"]["humidity"],
        "wind_speed": d["wind"]["speed"],
        "pressure": d["main"]["pressure"],
    }

# ----------------------------------------------------------------------------- 
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def engineer_features(aqi, weather, ts):
    hour = ts.hour
    dow = ts.weekday()
    return np.array([
        aqi["aqi"], aqi["pm25"], aqi["pm10"], aqi["no2"], aqi["o3"], aqi["co"],
        weather["temp"], weather["humidity"], weather["wind_speed"], weather["pressure"],
        hour, dow,
        1 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0,
        1 if dow >= 5 else 0,
        1 if hour < 6 or hour > 22 else 0,
        weather["temp"] * weather["humidity"] / 100,
        weather["wind_speed"] * aqi["aqi"],
        aqi["pm25"] + 0.5 * aqi["pm10"],
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ], dtype=np.float32)

# ----------------------------------------------------------------------------- 
# MODEL SYSTEM
# -----------------------------------------------------------------------------

class IAQSystem:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.reg = RandomForestRegressor(n_estimators=25, random_state=42)
        self.anom = IsolationForest(contamination=0.1, random_state=42)
        self._train()

    def _train(self):
        X, y = [], []
        for _ in range(400):
            oa = np.random.uniform(20, 150)
            ws = np.random.uniform(0, 8)
            f = np.random.rand(20)
            f[0] = oa
            f[8] = ws
            X.append(f)
            y.append(oa * (0.3 + 0.05 * ws))
        self.reg.fit(X, y)
        self.anom.fit(X)

    def update(self):
        aqi = fetch_outdoor_aqi(self.lat, self.lon)
        weather = fetch_weather(self.lat, self.lon)
        ts = datetime.now()
        feat = engineer_features(aqi, weather, ts)
        indoor = float(self.reg.predict([feat])[0])
        anomaly = self.anom.predict([feat])[0] == -1
        outdoor = aqi["aqi"] * 50
        return {
            "outdoor_aqi": outdoor,
            "temp": weather["temp"],
            "humidity": weather["humidity"],
            "wind_speed": weather["wind_speed"],
            "indoor_aqi": indoor,
            "indoor_risk": "Good" if indoor < 50 else "Moderate",
            "outdoor_risk": "Good" if outdoor < 50 else "Moderate",
            "is_anomaly": anomaly,
            "recommendation": "OPEN WINDOWS" if outdoor < indoor else "CLOSE WINDOWS",
            "explanation": "Auto decision based on AQI delta",
            "timestamp": ts.isoformat(),
            "city": current_city,
        }

# ----------------------------------------------------------------------------- 
# BACKGROUND WORKER (GUNICORN SAFE)
# -----------------------------------------------------------------------------

system = IAQSystem(**CITIES[current_city])
_bg_started = False

def background_loop():
    global latest_data
    print("✅ Background thread running")
    while True:
        try:
            data = system.update()
            with data_lock:
                latest_data = data
            print("🔄 Data updated")
        except Exception as e:
            print("❌ Background error:", e)
        time.sleep(60)

@app.before_first_request
def start_background():
    global _bg_started
    if not _bg_started:
        threading.Thread(target=background_loop, daemon=True).start()
        _bg_started = True

# ----------------------------------------------------------------------------- 
# ROUTES
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def data():
    with data_lock:
        return jsonify(latest_data or {})

@app.route("/api/cities")
def cities():
    return jsonify({"cities": list(CITIES.keys()), "current": current_city})

@app.route("/api/change-city", methods=["POST"])
def change_city():
    global system, current_city
    city = request.json.get("city")
    if city not in CITIES:
        return jsonify({"error": "Invalid city"}), 400
    current_city = city
    system = IAQSystem(**CITIES[city])
    return jsonify({"status": "ok", "city": city})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

# ----------------------------------------------------------------------------- 
# LOCAL DEV ONLY
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
