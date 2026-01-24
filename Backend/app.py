from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
from datetime import datetime
from typing import Optional, Dict, Tuple
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

print(f"✅ API Key loaded: {API_KEY[:8]}...")

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
    "Mirigama": {"lat": 7.2417, "lon": 80.1228},
    "Nalluruwa": {"lat": 6.8167, "lon": 79.8833},
    "Panadura": {"lat": 6.7133, "lon": 79.9026},
}

current_city = "Colombo"

# ----------------------------------------------------------------------------- 
# GLOBAL STATE (THREAD-SAFE)
# -----------------------------------------------------------------------------

latest_data = {
    "status": "initializing",
    "city": current_city,
    "outdoor_aqi": 0,
    "temp": 0,
    "humidity": 0,
    "wind_speed": 0,
    "indoor_aqi": 0,
    "indoor_risk": "Loading...",
    "outdoor_risk": "Loading...",
    "is_anomaly": False,
    "recommendation": "LOADING...",
    "explanation": "System initializing...",
    "timestamp": datetime.now().isoformat()
}
data_lock = threading.Lock()

# ----------------------------------------------------------------------------- 
# DATA FETCH (HTTPS ONLY)
# -----------------------------------------------------------------------------

def fetch_outdoor_aqi(lat: float, lon: float) -> Optional[Dict]:
    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"
        params = {"lat": lat, "lon": lon, "appid": API_KEY}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()["list"][0]
        result = {
            "aqi": d["main"]["aqi"],
            "pm25": d["components"].get("pm2_5", 0),
            "pm10": d["components"].get("pm10", 0),
            "no2": d["components"].get("no2", 0),
            "o3": d["components"].get("o3", 0),
            "co": d["components"].get("co", 0),
        }
        print(f"✅ AQI fetched: {result['aqi']}")
        return result
    except Exception as e:
        print(f"❌ AQI fetch error: {e}")
        raise

def fetch_weather(lat: float, lon: float) -> Optional[Dict]:
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        result = {
            "temp": d["main"]["temp"],
            "humidity": d["main"]["humidity"],
            "wind_speed": d["wind"]["speed"],
            "pressure": d["main"]["pressure"],
        }
        print(f"✅ Weather fetched: {result['temp']}°C")
        return result
    except Exception as e:
        print(f"❌ Weather fetch error: {e}")
        raise

# ----------------------------------------------------------------------------- 
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def engineer_features(aqi: Dict, weather: Dict, ts: datetime) -> np.ndarray:
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
# DECISION ENGINE
# -----------------------------------------------------------------------------

def get_health_risk_band(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def make_recommendation(
    indoor_aqi: float, 
    outdoor_aqi: float, 
    temp: float, 
    humidity: float, 
    wind_speed: float, 
    is_anomaly: bool
) -> Tuple[str, str]:
    outdoor_better = outdoor_aqi < indoor_aqi * 0.8
    temp_ok = 10 <= temp <= 30
    humidity_ok = 30 <= humidity <= 70
    wind_ok = wind_speed < 8.0
    
    if is_anomaly:
        if outdoor_better and temp_ok:
            return "OPEN WINDOWS", "Anomaly detected. Outdoor air is cleaner."
        return "CLOSE WINDOWS", "Anomaly detected. Keep windows closed."
    
    if outdoor_aqi > 150:
        return "CLOSE WINDOWS", f"Outdoor air is unhealthy (AQI {outdoor_aqi:.0f})."
    
    if indoor_aqi > 100 and outdoor_better and temp_ok and humidity_ok:
        return "OPEN WINDOWS", f"Indoor air quality is poor. Outdoor air is better."
    
    if outdoor_better and temp_ok and humidity_ok and wind_ok:
        return "OPEN WINDOWS", "Outdoor conditions favorable for ventilation."
    
    if not temp_ok:
        return "CLOSE WINDOWS", f"Temperature {temp:.1f}°C outside comfort range."
    
    if not humidity_ok:
        return "CLOSE WINDOWS", f"Humidity {humidity:.0f}% outside comfort range."
    
    if wind_speed >= 8.0:
        return "CLOSE WINDOWS", f"Wind speed {wind_speed:.1f} m/s too strong."
    
    return "CLOSE WINDOWS", "Maintain current indoor conditions."

# ----------------------------------------------------------------------------- 
# MODEL SYSTEM
# -----------------------------------------------------------------------------

class IAQSystem:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
        self.reg = RandomForestRegressor(
            n_estimators=50, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=2,
            random_state=42
        )
        self.anom = IsolationForest(
            n_estimators=50, 
            max_samples=256,
            contamination=0.1, 
            random_state=42,
            n_jobs=2
        )
        self._train()

    def _train(self) -> None:
        print("🔧 Training models...")
        X = []
        y = []
        
        for _ in range(1000):
            oa = np.random.uniform(10, 150)
            ws = np.random.uniform(0, 10)
            temp = np.random.uniform(10, 35)
            hum = np.random.uniform(30, 90)
            
            f = np.zeros(20, dtype=np.float32)
            f[0] = oa
            f[6] = temp
            f[7] = hum
            f[8] = ws
            f[9] = np.random.uniform(990, 1030)
            
            X.append(f)
            indoor = oa * min(0.7, 0.3 + 0.05 * ws) + np.random.uniform(0, 10)
            y.append(indoor)
        
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)
        
        self.reg.fit(X_arr, y_arr)
        self.anom.fit(X_arr)
        print("✅ Models trained")

    def update(self) -> Dict:
        print(f"\n🔄 Updating data for {current_city}...")
        try:
            aqi = fetch_outdoor_aqi(self.lat, self.lon)
            weather = fetch_weather(self.lat, self.lon)
            ts = datetime.now()
            feat = engineer_features(aqi, weather, ts)
            
            indoor = float(self.reg.predict([feat])[0])
            indoor = max(1.0, min(500.0, indoor))
            
            anomaly = self.anom.predict([feat])[0] == -1
            outdoor = aqi["aqi"] * 50
            
            indoor_risk = get_health_risk_band(indoor)
            outdoor_risk = get_health_risk_band(outdoor)
            
            recommendation, explanation = make_recommendation(
                indoor, outdoor, weather["temp"], 
                weather["humidity"], weather["wind_speed"], anomaly
            )
            
            print(f"✅ Indoor: {indoor:.0f}, Outdoor: {outdoor:.0f}, Rec: {recommendation}")
            
            return {
                "status": "active",
                "outdoor_aqi": outdoor,
                "temp": weather["temp"],
                "humidity": weather["humidity"],
                "wind_speed": weather["wind_speed"],
                "indoor_aqi": indoor,
                "indoor_risk": indoor_risk,
                "outdoor_risk": outdoor_risk,
                "is_anomaly": anomaly,
                "recommendation": recommendation,
                "explanation": explanation,
                "timestamp": ts.isoformat(),
                "city": current_city,
            }
        except Exception as e:
            print(f"❌ Update failed: {e}")
            import traceback
            traceback.print_exc()
            raise

# ----------------------------------------------------------------------------- 
# INITIALIZE SYSTEM AND START BACKGROUND THREAD
# -----------------------------------------------------------------------------

print(f"🌍 Initializing system for {current_city}...")
system = IAQSystem(**CITIES[current_city])

# Perform initial update
try:
    print("📊 Fetching initial data...")
    initial_data = system.update()
    with data_lock:
        latest_data = initial_data
    print("✅ Initial data loaded successfully")
except Exception as e:
    print(f"⚠️ Initial data fetch failed: {e}")
    print("Will retry in background thread")

def background_loop() -> None:
    global latest_data
    print("✅ Background thread started")
    
    time.sleep(10)
    
    while True:
        try:
            data = system.update()
            with data_lock:
                latest_data = data
            print(f"🔄 Background update complete at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"❌ Background error: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(60)

bg_thread = threading.Thread(target=background_loop, daemon=True)
bg_thread.start()
print("✅ Background updater started")

# ----------------------------------------------------------------------------- 
# ROUTES
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data")
def data():
    with data_lock:
        return jsonify(latest_data)

@app.route("/api/refresh")
def refresh():
    try:
        global latest_data
        print("🔄 Manual refresh triggered")
        data = system.update()
        with data_lock:
            latest_data = data
        return jsonify(data)
    except Exception as e:
        print(f"❌ Refresh error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/cities")
def cities():
    return jsonify({"cities": list(CITIES.keys()), "current": current_city})

@app.route("/api/change-city", methods=["POST"])
def change_city():
    global system, current_city, latest_data
    try:
        city = request.json.get("city")
        if city not in CITIES:
            return jsonify({"error": "Invalid city"}), 400
        
        print(f"🌍 Changing city to {city}")
        current_city = city
        system = IAQSystem(**CITIES[city])
        
        data = system.update()
        with data_lock:
            latest_data = data
        
        return jsonify({"status": "ok", "city": city, "data": data})
    except Exception as e:
        print(f"❌ City change error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    with data_lock:
        return jsonify({
            "status": "healthy",
            "data_status": latest_data.get("status", "unknown"),
            "city": current_city,
            "api_configured": bool(API_KEY),
            "timestamp": datetime.now().isoformat()
        })

# ----------------------------------------------------------------------------- 
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("=" * 70)
    print("🌬️  Indoor Air Quality System - Web Version")
    print("=" * 70)
    print(f"📍 City: {current_city}")
    print(f"🌐 Port: {port}")
    print(f"🔑 API: {API_KEY[:8]}...")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=False)