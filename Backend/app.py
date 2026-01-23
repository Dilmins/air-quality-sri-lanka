from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import requests
from datetime import datetime
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import threading
import time
import os

app = Flask(__name__, 
            template_folder='../Frontend/templates',
            static_folder='../Frontend/static')
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = "892e9461d30e3702e6976bfe327d69f7"

# Major cities in Sri Lanka with coordinates
CITIES = {
    'Colombo': {'lat': 6.9271, 'lon': 79.8612},
    'Kandy': {'lat': 7.2906, 'lon': 80.6337},
    'Galle': {'lat': 6.0535, 'lon': 80.2210},
    'Jaffna': {'lat': 9.6615, 'lon': 80.0255},
    'Negombo': {'lat': 7.2008, 'lon': 79.8358},
    'Trincomalee': {'lat': 8.5874, 'lon': 81.2152},
    'Batticaloa': {'lat': 7.7310, 'lon': 81.6747},
    'Matara': {'lat': 5.9549, 'lon': 80.5550},
    'Anuradhapura': {'lat': 8.3114, 'lon': 80.4037},
    'Kurunegala': {'lat': 7.4863, 'lon': 80.3623},
    'Ratnapura': {'lat': 6.6828, 'lon': 80.3992},
    'Badulla': {'lat': 6.9934, 'lon': 81.0550},
    'Nuwara Eliya': {'lat': 6.9497, 'lon': 80.7891},
    'Matale': {'lat': 7.4675, 'lon': 80.6234},
    'Gampaha': {'lat': 7.0914, 'lon': 79.9990},
    'Nalluruwa': {'lat': 6.7000, 'lon': 79.9167},
    'Mirigama': {'lat': 7.2382, 'lon': 80.1262}
}

# Default city
current_city = 'Colombo'
LATITUDE = CITIES[current_city]['lat']
LONGITUDE = CITIES[current_city]['lon']

# Global cache for latest data
latest_data = {
    'outdoor_aqi': 0,
    'temp': 0,
    'humidity': 0,
    'wind_speed': 0,
    'indoor_aqi': 0,
    'indoor_risk': 'Unknown',
    'outdoor_risk': 'Unknown',
    'is_anomaly': False,
    'recommendation': 'Loading...',
    'explanation': 'Fetching data...',
    'timestamp': datetime.now().isoformat()
}

# ============================================================================
# DATA INGESTION
# ============================================================================

def fetch_outdoor_aqi(lat: float, lon: float, api_key: str) -> Optional[Dict]:
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'list' not in data or len(data['list']) == 0:
            return None
            
        current = data['list'][0]
        components = current['main']
        pollutants = current.get('components', {})
        
        return {
            'aqi': components.get('aqi', 1),
            'pm25': pollutants.get('pm2_5', 0),
            'pm10': pollutants.get('pm10', 0),
            'no2': pollutants.get('no2', 0),
            'o3': pollutants.get('o3', 0),
            'co': pollutants.get('co', 0)
        }
    except Exception as e:
        print(f"AQI fetch error: {e}")
        return None

def fetch_weather(lat: float, lon: float, api_key: str) -> Optional[Dict]:
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'temp': data['main'].get('temp', 20),
            'humidity': data['main'].get('humidity', 50),
            'wind_speed': data['wind'].get('speed', 0),
            'pressure': data['main'].get('pressure', 1013)
        }
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(aqi_data: Optional[Dict], weather_data: Optional[Dict], timestamp: datetime) -> Optional[np.ndarray]:
    if aqi_data is None or weather_data is None:
        return None
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    is_night = 1 if hour < 6 or hour > 22 else 0
    
    outdoor_aqi = aqi_data['aqi']
    pm25 = aqi_data['pm25']
    pm10 = aqi_data['pm10']
    no2 = aqi_data['no2']
    o3 = aqi_data['o3']
    co = aqi_data['co']
    
    temp = weather_data['temp']
    humidity = weather_data['humidity']
    wind_speed = weather_data['wind_speed']
    pressure = weather_data['pressure']
    
    temp_humidity_interaction = temp * humidity / 100.0
    wind_aqi_interaction = wind_speed * outdoor_aqi
    pollutant_mix = pm25 + 0.5 * pm10 + 0.3 * no2
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    features = np.array([
        outdoor_aqi, pm25, pm10, no2, o3, co,
        temp, humidity, wind_speed, pressure,
        hour, day_of_week, is_rush_hour, is_weekend, is_night,
        temp_humidity_interaction, wind_aqi_interaction, pollutant_mix,
        hour_sin, hour_cos
    ], dtype=np.float32)
    
    return features

# ============================================================================
# MODELS
# ============================================================================

class IAQRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=30, max_depth=8, min_samples_split=15,
            min_samples_leaf=8, max_features='sqrt', n_jobs=1, random_state=42
        )
        self.is_trained = False
        self.feature_importance = None
    
    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> Optional[float]:
        if not self.is_trained:
            return self._fallback_prediction(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        prediction = self.model.predict(X)[0]
        return float(max(1.0, min(500.0, prediction)))
    
    def _fallback_prediction(self, X: np.ndarray) -> float:
        outdoor_aqi = X[0]
        wind_speed = X[8]
        is_night = X[14]
        infiltration_factor = min(0.7, 0.3 + 0.05 * wind_speed)
        offset = 5.0 if is_night == 0 else 2.0
        indoor_aqi = outdoor_aqi * infiltration_factor + offset
        return float(max(1.0, min(500.0, indoor_aqi)))

class IAQAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=30, max_samples=128, contamination=0.1,
            random_state=42, n_jobs=1
        )
        self.is_trained = False
        self.threshold = -0.5
    
    def train(self, X: np.ndarray):
        self.model.fit(X)
        self.is_trained = True
    
    def detect(self, X: np.ndarray) -> bool:
        if not self.is_trained:
            return self._fallback_detection(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        score = self.model.decision_function(X)[0]
        return bool(score < self.threshold)
    
    def _fallback_detection(self, X: np.ndarray) -> bool:
        outdoor_aqi, pm25, temp, humidity = X[0], X[1], X[6], X[7]
        return bool(outdoor_aqi > 150 or pm25 > 55 or temp < 0 or 
                    temp > 40 or humidity < 20 or humidity > 90)

# ============================================================================
# DECISION ENGINE
# ============================================================================

class WindowRecommender:
    @staticmethod
    def get_health_risk_band(aqi: float) -> str:
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"
    
    @staticmethod
    def recommend(indoor_aqi: float, outdoor_aqi: float, temp: float,
                  humidity: float, wind_speed: float, is_anomaly: bool) -> Tuple[str, str, Dict]:
        outdoor_better = outdoor_aqi < indoor_aqi * 0.8
        temp_ok = 10 <= temp <= 30
        humidity_ok = 30 <= humidity <= 70
        wind_moderate = wind_speed < 8.0
        
        indoor_risk = WindowRecommender.get_health_risk_band(indoor_aqi)
        outdoor_risk = WindowRecommender.get_health_risk_band(outdoor_aqi)
        
        metadata = {
            'indoor_risk': indoor_risk, 'outdoor_risk': outdoor_risk,
            'outdoor_better': outdoor_better, 'temp_ok': temp_ok,
            'humidity_ok': humidity_ok, 'wind_moderate': wind_moderate,
            'is_anomaly': is_anomaly
        }
        
        if is_anomaly:
            if outdoor_better and temp_ok:
                return "OPEN WINDOWS", "Anomaly detected. Outdoor air is cleaner.", metadata
            else:
                return "CLOSE WINDOWS", "Anomaly detected. Keep windows closed.", metadata
        
        if outdoor_aqi > 150:
            return "CLOSE WINDOWS", f"Outdoor air is {outdoor_risk}. Keep windows closed.", metadata
        
        if indoor_aqi > 100 and outdoor_better and temp_ok and humidity_ok:
            return "OPEN WINDOWS", f"Indoor air is {indoor_risk}. Outdoor air is cleaner. Ventilate.", metadata
        
        if outdoor_better and temp_ok and humidity_ok and wind_moderate:
            return "OPEN WINDOWS", "Outdoor conditions favorable for ventilation.", metadata
        
        if not temp_ok:
            return "CLOSE WINDOWS", f"Temperature {temp:.1f}°C is outside comfort range.", metadata
        
        if not humidity_ok:
            return "CLOSE WINDOWS", f"Humidity {humidity:.0f}% is outside comfort range.", metadata
        
        if wind_speed >= 8.0:
            return "CLOSE WINDOWS", f"Wind speed {wind_speed:.1f} m/s is too strong.", metadata
        
        return "CLOSE WINDOWS", "Maintain current indoor conditions.", metadata

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

class IAQSystem:
    def __init__(self, api_key: str, lat: float, lon: float):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.regressor = IAQRegressor()
        self.anomaly_detector = IAQAnomalyDetector()
        self.recommender = WindowRecommender()
        self._initialize_models()
        
    def _initialize_models(self):
        print("Initializing models...")
        n_samples = 500  # Reduced for lower CPU
        X_train, y_train = [], []
        
        for _ in range(n_samples):
            outdoor_aqi = np.random.uniform(10, 150)
            pm25 = outdoor_aqi * np.random.uniform(0.3, 0.8)
            pm10 = pm25 * np.random.uniform(1.2, 2.0)
            no2, o3, co = np.random.uniform(10, 80), np.random.uniform(20, 100), np.random.uniform(200, 1000)
            temp, humidity = np.random.uniform(10, 30), np.random.uniform(30, 80)
            wind_speed, pressure = np.random.uniform(0, 10), np.random.uniform(990, 1030)
            hour, day_of_week = np.random.randint(0, 24), np.random.randint(0, 7)
            is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            is_weekend = 1 if day_of_week >= 5 else 0
            is_night = 1 if hour < 6 or hour > 22 else 0
            
            features = np.array([
                outdoor_aqi, pm25, pm10, no2, o3, co, temp, humidity, wind_speed, pressure,
                hour, day_of_week, is_rush_hour, is_weekend, is_night,
                temp * humidity / 100.0, wind_speed * outdoor_aqi, pm25 + 0.5 * pm10 + 0.3 * no2,
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
            ])
            
            indoor_aqi = outdoor_aqi * min(0.7, 0.3 + 0.05 * wind_speed) + np.random.uniform(0, 10)
            X_train.append(features)
            y_train.append(indoor_aqi)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        self.regressor.train(X_train, y_train)
        self.anomaly_detector.train(X_train)
        print("Models initialized successfully.")
        
    def update(self) -> Optional[Dict]:
        timestamp = datetime.now()
        aqi_data = fetch_outdoor_aqi(self.lat, self.lon, self.api_key)
        weather_data = fetch_weather(self.lat, self.lon, self.api_key)
        
        if aqi_data is None or weather_data is None:
            print("Failed to fetch data")
            return None
        
        features = engineer_features(aqi_data, weather_data, timestamp)
        if features is None:
            return None
        
        indoor_aqi = self.regressor.predict(features)
        is_anomaly = self.anomaly_detector.detect(features)
        outdoor_aqi = float(aqi_data['aqi'] * 50)
        
        recommendation, explanation, metadata = self.recommender.recommend(
            indoor_aqi, outdoor_aqi, weather_data['temp'],
            weather_data['humidity'], weather_data['wind_speed'], is_anomaly
        )
        
        return {
            'outdoor_aqi': float(outdoor_aqi),
            'temp': float(weather_data['temp']),
            'humidity': float(weather_data['humidity']),
            'wind_speed': float(weather_data['wind_speed']),
            'indoor_aqi': float(indoor_aqi),
            'indoor_risk': str(metadata['indoor_risk']),
            'outdoor_risk': str(metadata['outdoor_risk']),
            'is_anomaly': bool(is_anomaly),
            'recommendation': str(recommendation),
            'explanation': str(explanation),
            'timestamp': timestamp.isoformat(),
            'city': current_city
        }

# Initialize system
system = IAQSystem(API_KEY, LATITUDE, LONGITUDE)

# Background update thread
def background_updater():
    global latest_data
    while True:
        try:
            data = system.update()
            if data:
                latest_data = data
                print(f"✓ Updated at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Update error: {e}")
        time.sleep(60)

thread = threading.Thread(target=background_updater, daemon=True)
thread.start()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify(latest_data)

@app.route('/api/cities')
def get_cities():
    return jsonify({
        'cities': list(CITIES.keys()),
        'current': current_city
    })

@app.route('/api/change-city', methods=['POST'])
def change_city():
    global current_city, LATITUDE, LONGITUDE, system
    from flask import request
    
    data = request.json
    city = data.get('city')
    
    if city not in CITIES:
        return jsonify({'error': 'Invalid city'}), 400
    
    current_city = city
    LATITUDE = CITIES[city]['lat']
    LONGITUDE = CITIES[city]['lon']
    
    # Reinitialize system with new coordinates
    system = IAQSystem(API_KEY, LATITUDE, LONGITUDE)
    
    # Get fresh data
    updated_data = system.update()
    if updated_data:
        global latest_data
        latest_data = updated_data
        return jsonify(updated_data)
    
    return jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/refresh')
def refresh():
    data = system.update()
    if data:
        global latest_data
        latest_data = data
        return jsonify(data)
    return jsonify({'error': 'Failed to fetch data'}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("  Indoor Air Quality Intelligence System - Web Version")
    print("=" * 70)
    print(f" Default Location: {current_city}, Sri Lanka")
    print(" Server starting at http://localhost:5000")
    print(" Auto-refresh: Every 60 seconds")
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)