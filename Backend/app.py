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
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')
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
    'rain_probability_24h': 0.0,
    'rain_risk_curve': [],
    'recommendation': 'Loading...',
    'explanation': 'Fetching data...',
    'timestamp': datetime.now().isoformat(),
    'city': current_city
}

# Load rain model
rain_model = None
try:
    with open('rain_model.pkl', 'rb') as f:
        rain_model = pickle.load(f)
        print("✅ Rain model loaded successfully")
except Exception as e:
    print(f"⚠️ Rain model not found: {e}")

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
        print(f"AQI error: {e}")
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
            'pressure': data['main'].get('pressure', 1013),
            'clouds': data.get('clouds', {}).get('all', 50)
        }
    except Exception as e:
        print(f"Weather error: {e}")
        return None

def fetch_24h_forecast(lat: float, lon: float, api_key: str) -> Optional[list]:
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        forecast_list = data.get('list', [])[:8]  # Next 24 hours (8 x 3-hour intervals)
        return forecast_list if forecast_list else None
    except Exception as e:
        print(f"Forecast error: {e}")
        return None

def calculate_dew_point(temp: float, humidity: float) -> float:
    """Calculate dew point temperature"""
    return temp - ((100 - humidity) / 5.0)

def extract_rain_features(weather: Dict, forecast: list) -> Optional[np.ndarray]:
    """Extract features for rain prediction from current weather and 24h forecast"""
    if not weather or not forecast:
        return None
    
    try:
        temp = weather['temp']
        humidity = weather['humidity']
        pressure = weather['pressure']
        wind_speed = weather['wind_speed']
        clouds = weather['clouds']
        dew_point = calculate_dew_point(temp, humidity)
        
        # Extract forecast statistics
        humidities = [f['main']['humidity'] for f in forecast]
        pressures = [f['main']['pressure'] for f in forecast]
        cloud_covers = [f.get('clouds', {}).get('all', 50) for f in forecast]
        rains = [f.get('rain', {}).get('3h', 0) for f in forecast]
        
        mean_humidity_24h = np.mean(humidities)
        max_humidity_24h = np.max(humidities)
        mean_pressure_24h = np.mean(pressures)
        pressure_trend_24h = pressures[-1] - pressures[0] if len(pressures) > 1 else 0
        mean_clouds_24h = np.mean(cloud_covers)
        max_clouds_24h = np.max(cloud_covers)
        total_rain_24h = sum(rains)
        rain_steps_24h = sum(1 for r in rains if r > 0)
        
        features = np.array([
            temp, humidity, pressure, wind_speed, clouds, dew_point,
            mean_humidity_24h, max_humidity_24h, mean_pressure_24h, pressure_trend_24h,
            mean_clouds_24h, max_clouds_24h, total_rain_24h, rain_steps_24h
        ], dtype=np.float32)
        
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def generate_rain_risk_curve(forecast: list, ml_probability: float) -> list:
    """Generate hour-by-hour rain risk curve from forecast data"""
    if not forecast:
        return []
    
    risk_curve = []
    
    for i, f in enumerate(forecast):
        hour = i * 3  # 3-hour intervals
        
        try:
            humidity = f['main']['humidity']
            pressure = f['main']['pressure']
            clouds = f.get('clouds', {}).get('all', 50)
            has_rain = f.get('rain', {}).get('3h', 0) > 0
            
            # Calculate risk score based on conditions
            risk = 0.0
            
            # Humidity contribution (0-35%)
            if humidity > 85:
                risk += 30
            elif humidity > 75:
                risk += 20
            elif humidity > 65:
                risk += 10
            
            # Pressure contribution (0-25%)
            if pressure < 1005:
                risk += 20
            elif pressure < 1010:
                risk += 10
            
            # Cloud cover contribution (0-25%)
            risk += (clouds / 100) * 25
            
            # Actual rain detection (adds 20%)
            if has_rain:
                risk += 20
            
            # Blend with ML model prediction
            risk = (risk * 0.6) + (ml_probability * 100 * 0.4)
            
            risk = max(0, min(100, risk))
            risk_curve.append({'hour': hour, 'risk': round(risk, 1)})
        except Exception as e:
            print(f"Risk curve error at hour {hour}: {e}")
            risk_curve.append({'hour': hour, 'risk': round(ml_probability * 100, 1)})
    
    return risk_curve

def engineer_features(aqi_data: Optional[Dict], weather_data: Optional[Dict], timestamp: datetime) -> Optional[np.ndarray]:
    if aqi_data is None or weather_data is None:
        return None
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    is_night = 1 if hour < 6 or hour > 22 else 0
    
    features = np.array([
        aqi_data['aqi'], aqi_data['pm25'], aqi_data['pm10'], aqi_data['no2'], aqi_data['o3'], aqi_data['co'],
        weather_data['temp'], weather_data['humidity'], weather_data['wind_speed'], weather_data['pressure'],
        hour, day_of_week, is_rush_hour, is_weekend, is_night,
        weather_data['temp'] * weather_data['humidity'] / 100.0,
        weather_data['wind_speed'] * aqi_data['aqi'],
        aqi_data['pm25'] + 0.5 * aqi_data['pm10'] + 0.3 * aqi_data['no2'],
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24)
    ], dtype=np.float32)
    
    return features

class IAQRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=30, max_depth=8, min_samples_split=15,
            min_samples_leaf=8, max_features='sqrt', n_jobs=1, random_state=42
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> float:
        if not self.is_trained:
            return self._fallback_prediction(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return float(max(1.0, min(500.0, self.model.predict(X)[0])))
    
    def _fallback_prediction(self, X: np.ndarray) -> float:
        outdoor_aqi = X[0]
        wind_speed = X[8]
        infiltration = min(0.7, 0.3 + 0.05 * wind_speed)
        return float(max(1.0, min(500.0, outdoor_aqi * infiltration + 5.0)))

class IAQAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=30, max_samples=128, contamination=0.1,
            random_state=42, n_jobs=1
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray):
        self.model.fit(X)
        self.is_trained = True
    
    def detect(self, X: np.ndarray) -> bool:
        if not self.is_trained:
            return bool(X[0] > 150 or X[1] > 55)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return bool(self.model.decision_function(X)[0] < -0.5)

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
                  humidity: float, wind_speed: float, is_anomaly: bool,
                  rain_probability: float) -> Tuple[str, str, Dict]:
        
        temp_ok = 15 <= temp <= 32
        humidity_ok = 35 <= humidity <= 80
        wind_moderate = wind_speed < 8.0
        high_rain_risk = rain_probability > 0.5
        
        indoor_risk = WindowRecommender.get_health_risk_band(indoor_aqi)
        outdoor_risk = WindowRecommender.get_health_risk_band(outdoor_aqi)
        outdoor_better = outdoor_aqi < indoor_aqi - 10
        
        metadata = {
            'indoor_risk': indoor_risk,
            'outdoor_risk': outdoor_risk,
            'outdoor_better': outdoor_better,
            'temp_ok': temp_ok,
            'humidity_ok': humidity_ok,
            'wind_moderate': wind_moderate,
            'is_anomaly': is_anomaly
        }
        
        if is_anomaly:
            return ("KEEP CLOSED", "Anomaly detected. Keep windows closed for safety.", metadata)
        
        if high_rain_risk:
            return ("KEEP CLOSED", f"Rain probability {rain_probability*100:.0f}%. Keep windows closed.", metadata)
        
        if outdoor_aqi > 150:
            return ("KEEP CLOSED", f"Outdoor air is {outdoor_risk}. Keep windows closed.", metadata)
        
        if indoor_aqi <= 50 and outdoor_aqi <= 50 and temp_ok and humidity_ok:
            return ("OPEN WINDOWS", "Excellent air quality. Safe to ventilate.", metadata)
        
        if indoor_aqi > 100 and outdoor_better and temp_ok and humidity_ok:
            return ("OPEN WINDOWS", f"Indoor air is {indoor_risk}. Outdoor air is cleaner.", metadata)
        
        if outdoor_better and temp_ok and humidity_ok and wind_moderate:
            return ("OPEN WINDOWS", "Outdoor conditions favorable for ventilation.", metadata)
        
        return ("KEEP CLOSED", f"Indoor: {indoor_risk}. Maintain current conditions.", metadata)

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
        print("🔧 Initializing IAQ models...")
        n = 500
        X, y = [], []
        for _ in range(n):
            oa = np.random.uniform(10, 150)
            ws = np.random.uniform(0, 10)
            f = np.random.rand(20)
            f[0], f[8] = oa, ws
            X.append(f)
            y.append(oa * min(0.7, 0.3 + 0.05 * ws) + np.random.uniform(0, 10))
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        self.regressor.train(X, y)
        self.anomaly_detector.train(X)
        print("✅ IAQ models ready")
        
    def update(self) -> Optional[Dict]:
        timestamp = datetime.now()
        
        print(f"\n🔄 Updating data for {current_city}...")
        
        aqi_data = fetch_outdoor_aqi(self.lat, self.lon, self.api_key)
        weather_data = fetch_weather(self.lat, self.lon, self.api_key)
        forecast_data = fetch_24h_forecast(self.lat, self.lon, self.api_key)
        
        if not aqi_data or not weather_data:
            print("❌ Failed to fetch basic data")
            return None
        
        features = engineer_features(aqi_data, weather_data, timestamp)
        if features is None:
            return None
        
        indoor_aqi = self.regressor.predict(features)
        is_anomaly = self.anomaly_detector.detect(features)
        outdoor_aqi = float(aqi_data['aqi'] * 50)
        
        # Rain prediction
        rain_probability = 0.0
        rain_risk_curve = []
        
        if rain_model and forecast_data:
            try:
                rain_features = extract_rain_features(weather_data, forecast_data)
                if rain_features is not None:
                    rain_probability = float(rain_model.predict_proba([rain_features])[0][1])
                    rain_risk_curve = generate_rain_risk_curve(forecast_data, rain_probability)
                    print(f"🌧️  Rain probability: {rain_probability*100:.1f}%")
                else:
                    print("⚠️  Could not extract rain features")
            except Exception as e:
                print(f"❌ Rain prediction error: {e}")
        
        # Fallback: generate basic curve if prediction failed
        if not rain_risk_curve:
            print("⚠️  Using fallback rain curve")
            base_risk = (weather_data['humidity'] - 50) / 50 * 100
            base_risk = max(10, min(80, base_risk))
            rain_risk_curve = [
                {'hour': i*3, 'risk': round(base_risk + np.random.uniform(-10, 10), 1)}
                for i in range(8)
            ]
        
        recommendation, explanation, metadata = self.recommender.recommend(
            indoor_aqi, outdoor_aqi, weather_data['temp'],
            weather_data['humidity'], weather_data['wind_speed'],
            is_anomaly, rain_probability
        )
        
        print(f"✅ Update complete: Indoor AQI={indoor_aqi:.0f}, Rain={rain_probability*100:.1f}%")
        
        return {
            'outdoor_aqi': float(outdoor_aqi),
            'temp': float(weather_data['temp']),
            'humidity': float(weather_data['humidity']),
            'wind_speed': float(weather_data['wind_speed']),
            'indoor_aqi': float(indoor_aqi),
            'indoor_risk': str(metadata['indoor_risk']),
            'outdoor_risk': str(metadata['outdoor_risk']),
            'is_anomaly': bool(is_anomaly),
            'rain_probability_24h': float(rain_probability),
            'rain_risk_curve': rain_risk_curve,
            'recommendation': str(recommendation),
            'explanation': str(explanation),
            'timestamp': timestamp.isoformat(),
            'city': current_city
        }

system = IAQSystem(API_KEY, LATITUDE, LONGITUDE)

def background_updater():
    global latest_data
    time.sleep(5)  # Initial delay
    while True:
        try:
            data = system.update()
            if data:
                latest_data = data
        except Exception as e:
            print(f"Background update error: {e}")
        time.sleep(60)

thread = threading.Thread(target=background_updater, daemon=True)
thread.start()

# Perform initial update
initial_data = system.update()
if initial_data:
    latest_data = initial_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify(latest_data)

@app.route('/api/cities')
def get_cities():
    return jsonify({'cities': list(CITIES.keys()), 'current': current_city})

@app.route('/api/change-city', methods=['POST'])
def change_city():
    global current_city, LATITUDE, LONGITUDE, system
    
    city = request.json.get('city')
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

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'rain_model_loaded': rain_model is not None})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 Starting server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)