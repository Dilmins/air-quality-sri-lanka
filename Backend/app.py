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

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

API_KEY = os.getenv("OPENWEATHER_API_KEY", "892e9461d30e3702e6976bfe327d69f7")

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
    "Matale": {"lat": 7.4675, "lon": 80.6234},
    "Gampaha": {"lat": 7.0914, "lon": 79.9990},
    "Nalluruwa": {"lat": 6.7000, "lon": 79.9167},
    "Mirigama": {"lat": 7.2382, "lon": 80.1262},
    "Panadura": {"lat": 6.7133, "lon": 79.9026},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503},
    "Osaka": {"lat": 34.6937, "lon": 135.5023},
    "Hiroshima": {"lat": 34.3853, "lon": 132.4553}
}

current_city = "Colombo"
LATITUDE = CITIES[current_city]['lat']
LONGITUDE = CITIES[current_city]['lon']

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
    'timestamp': datetime.now().isoformat(),
    'city': current_city
}

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
        return None

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

class IAQRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', n_jobs=1, random_state=42
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
            n_estimators=50, max_samples=256, contamination=0.05,
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
        # Temperature and humidity ranges suitable for Sri Lanka's tropical climate
        temp_ok = 20 <= temp <= 32
        humidity_ok = 40 <= humidity <= 85
        wind_moderate = wind_speed < 10.0
        
        indoor_risk = WindowRecommender.get_health_risk_band(indoor_aqi)
        outdoor_risk = WindowRecommender.get_health_risk_band(outdoor_aqi)
        
        # Outdoor is better if it's at least 15 points lower than indoor
        outdoor_better = outdoor_aqi < indoor_aqi - 15
        
        metadata = {
            'indoor_risk': indoor_risk, 'outdoor_risk': outdoor_risk,
            'outdoor_better': outdoor_better, 'temp_ok': temp_ok,
            'humidity_ok': humidity_ok, 'wind_moderate': wind_moderate,
            'is_anomaly': is_anomaly
        }
        
        # Handle anomalies first
        if is_anomaly:
            if outdoor_better and temp_ok:
                return "OPEN WINDOWS", "Anomaly detected. Outdoor air is cleaner - ventilate immediately.", metadata
            else:
                return "CLOSE WINDOWS", "Anomaly detected. Keep windows closed for safety.", metadata
        
        # When both indoor and outdoor are GOOD - open windows for fresh air
        if indoor_aqi <= 50 and outdoor_aqi <= 50:
            if temp_ok and humidity_ok and wind_moderate:
                return "OPEN WINDOWS", "Excellent air quality both indoors and outdoors. Enjoy natural ventilation.", metadata
            elif not temp_ok:
                return "CLOSE WINDOWS", f"Air quality is good, but temperature ({temp:.1f}°C) is outside comfort range.", metadata
            elif not humidity_ok:
                return "CLOSE WINDOWS", f"Air quality is good, but humidity ({humidity:.0f}%) is outside comfort range.", metadata
            else:
                return "CLOSE WINDOWS", f"Air quality is good, but wind is too strong ({wind_speed:.1f} m/s).", metadata
        
        # When outdoor is hazardous or very unhealthy - always close
        if outdoor_aqi > 200:
            return "CLOSE WINDOWS", f"Outdoor air is {outdoor_risk}. Keep windows closed and use air purifier.", metadata
        
        # When indoor is good but outdoor is moderate - close to maintain good indoor air
        if indoor_aqi <= 50 and outdoor_aqi > 50:
            return "CLOSE WINDOWS", f"Indoor air is excellent ({indoor_risk}). Keep windows closed to maintain quality.", metadata
        
        # When indoor is moderate/unhealthy and outdoor is significantly better
        if indoor_aqi > 50 and outdoor_better and temp_ok and humidity_ok:
            return "OPEN WINDOWS", f"Indoor air is {indoor_risk}, outdoor is {outdoor_risk}. Ventilate to improve indoor quality.", metadata
        
        # When outdoor is moderately better but weather conditions aren't ideal
        if outdoor_better:
            if not temp_ok:
                return "CLOSE WINDOWS", f"Outdoor air is cleaner, but temperature ({temp:.1f}°C) makes ventilation uncomfortable.", metadata
            elif not humidity_ok:
                return "CLOSE WINDOWS", f"Outdoor air is cleaner, but humidity ({humidity:.0f}%) makes ventilation uncomfortable.", metadata
            elif not wind_moderate:
                return "CLOSE WINDOWS", f"Outdoor air is cleaner, but wind ({wind_speed:.1f} m/s) is too strong.", metadata
        
        # When indoor is acceptable (moderate) - maintain status
        if indoor_aqi <= 100:
            return "CLOSE WINDOWS", f"Indoor air is {indoor_risk}. Maintain current conditions.", metadata
        
        # When both are similar quality - close to avoid unnecessary exchange
        if abs(indoor_aqi - outdoor_aqi) < 15:
            return "CLOSE WINDOWS", f"Indoor and outdoor air quality are similar ({indoor_risk}). Keep windows closed.", metadata
        
        # Default case - close windows
        return "CLOSE WINDOWS", f"Indoor: {indoor_risk}, Outdoor: {outdoor_risk}. Keep windows closed.", metadata

def generate_realistic_training_data():
    """
    Generate training data based on real-world research on indoor/outdoor air quality relationships.
    
    Based on scientific literature:
    - Indoor/outdoor PM2.5 ratios typically range from 0.2 to 1.2
    - Infiltration rates depend on building characteristics, ventilation, and outdoor conditions
    - Temperature and humidity affect particle deposition and resuspension
    - Time of day affects occupancy and activities (cooking, cleaning)
    """
    
    X_train, y_train = [], []
    n_samples = 2000
    
    for _ in range(n_samples):
        # Outdoor conditions based on real-world distributions
        outdoor_aqi_raw = np.random.choice([1, 2, 3, 4, 5], p=[0.35, 0.40, 0.15, 0.07, 0.03])
        
        if outdoor_aqi_raw == 1:  # Good
            outdoor_aqi = np.random.uniform(10, 50)
            pm25 = np.random.uniform(5, 25)
        elif outdoor_aqi_raw == 2:  # Moderate
            outdoor_aqi = np.random.uniform(50, 100)
            pm25 = np.random.uniform(25, 50)
        elif outdoor_aqi_raw == 3:  # Unhealthy for sensitive
            outdoor_aqi = np.random.uniform(100, 150)
            pm25 = np.random.uniform(50, 75)
        elif outdoor_aqi_raw == 4:  # Unhealthy
            outdoor_aqi = np.random.uniform(150, 200)
            pm25 = np.random.uniform(75, 100)
        else:  # Very unhealthy
            outdoor_aqi = np.random.uniform(200, 300)
            pm25 = np.random.uniform(100, 150)
        
        pm10 = pm25 * np.random.uniform(1.5, 2.5)
        no2 = np.random.uniform(10, 100)
        o3 = np.random.uniform(20, 120)
        co = np.random.uniform(200, 1500)
        
        # Weather conditions realistic for tropical/temperate climates
        temp = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 90)
        wind_speed = np.random.gamma(2, 1.5)  # Realistic wind distribution
        pressure = np.random.uniform(990, 1030)
        
        # Temporal features
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour < 6 or hour > 22 else 0
        
        # Calculate indoor AQI based on realistic physics and building science
        
        # Base infiltration rate (0.2-0.7 depending on building tightness)
        building_tightness = np.random.uniform(0.2, 0.7)
        
        # Wind increases infiltration
        wind_factor = 1 + (wind_speed / 20)
        
        # Temperature difference affects stack effect
        temp_diff_factor = 1 + abs(temp - 24) / 100
        
        # Combined infiltration rate
        infiltration = building_tightness * wind_factor * temp_diff_factor
        infiltration = min(1.2, max(0.15, infiltration))
        
        # Indoor sources contribution
        # Higher during cooking hours (6-8am, 6-8pm) and cleaning
        indoor_source = 0
        if 6 <= hour <= 8 or 18 <= hour <= 20:
            indoor_source = np.random.uniform(5, 25)  # Cooking emissions
        elif 9 <= hour <= 17 and not is_weekend:
            indoor_source = np.random.uniform(0, 8)  # Lower during work hours
        else:
            indoor_source = np.random.uniform(0, 12)  # General activity
        
        # Deposition and removal
        # Higher humidity increases particle deposition
        removal_rate = 0.05 + (humidity / 1000)
        
        # Natural ventilation when windows open (assumed open when outdoor is much better)
        ventilation_bonus = 0
        if outdoor_aqi < 60 and temp > 20 and temp < 30:
            if np.random.random() < 0.3:  # 30% chance windows are open
                ventilation_bonus = -np.random.uniform(5, 15)
        
        # Calculate indoor AQI
        # I/O ratio based on infiltration, plus indoor sources, minus removal, plus ventilation
        indoor_aqi = (pm25 * infiltration) + indoor_source - (pm25 * removal_rate) + ventilation_bonus
        
        # Add realistic noise
        indoor_aqi += np.random.normal(0, 3)
        
        # Ensure reasonable bounds
        indoor_aqi = max(5, min(400, indoor_aqi))
        
        # Feature engineering
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
        
        X_train.append(features)
        y_train.append(indoor_aqi)
    
    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

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
        """Initialize models with realistic training data based on building science research"""
        print("Initializing ML models with realistic building physics data...")
        X_train, y_train = generate_realistic_training_data()
        
        print(f"Training Random Forest Regressor on {len(X_train)} samples...")
        self.regressor.train(X_train, y_train)
        
        print(f"Training Isolation Forest Anomaly Detector on {len(X_train)} samples...")
        self.anomaly_detector.train(X_train)
        
        print("ML models initialized successfully!")
        
    def update(self) -> Optional[Dict]:
        timestamp = datetime.now()
        aqi_data = fetch_outdoor_aqi(self.lat, self.lon, self.api_key)
        weather_data = fetch_weather(self.lat, self.lon, self.api_key)
        
        if aqi_data is None or weather_data is None:
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

system = IAQSystem(API_KEY, LATITUDE, LONGITUDE)

def background_updater():
    global latest_data
    while True:
        try:
            data = system.update()
            if data:
                latest_data = data
        except Exception as e:
            pass
        time.sleep(60)

thread = threading.Thread(target=background_updater, daemon=True)
thread.start()

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
    
    data = request.json
    city = data.get('city')
    
    if city not in CITIES:
        return jsonify({'error': 'Invalid city'}), 400
    
    current_city = city
    LATITUDE = CITIES[city]['lat']
    LONGITUDE = CITIES[city]['lon']
    
    system = IAQSystem(API_KEY, LATITUDE, LONGITUDE)
    
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
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)