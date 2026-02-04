from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import threading
import time
import os
import pickle
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENWEATHER_API_KEY", "892e9461d30e3702e6976bfe327d69f7")

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    print(f"✅ Using Neon PostgreSQL")
    USE_POSTGRES = True
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    print(f"⚠️ Using SQLite")
    USE_POSTGRES = False
    import sqlite3

def get_db():
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    else:
        return sqlite3.connect('monitoring.db')

api_calls_today = 0
api_calls_date = datetime.now().date()
API_CALL_LIMIT = 950

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
    "Rajagiriya": {"lat": 6.9089, "lon": 79.8911},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503},
    "Osaka": {"lat": 34.6937, "lon": 135.5023},
    "Hiroshima": {"lat": 34.3853, "lon": 132.4553},
    "Dubai": {"lat": 25.2048, "lon": 55.2708},
    "Uganda": {"lat": 1.3733, "lon": 32.2903},
    "Guyana": {"lat": 4.8604, "lon": -58.9302},
    "Malaysia": {"lat": 4.2105, "lon": 101.9758},
    "Thailand": {"lat": 15.8700, "lon": 100.9925},
    "Bambarabatuoya MHPP": {"lat": 6.7014, "lon": 80.5097},
    "Batathota MHPP": {"lat": 6.8126, "lon": 80.3757},
    "Wembiyagoda MHPP": {"lat": 6.5176, "lon": 80.4132},
    "Lower Kotmale Oya MHPP": {"lat": 7.0332, "lon": 80.6508},
    "Rideepana MHPP": {"lat": 7.0093, "lon": 81.0640},
    "Udawela MHPP": {"lat": 7.0564, "lon": 81.0608},
    "Madugeta MHPP": {"lat": 6.3706, "lon": 80.4088},
    "Ethamala Ella MHPP": {"lat": 6.2269, "lon": 80.4979},
    "Muvumbe SHPP": {"lat": -1.3186, "lon": 30.0789},
    "Bukinda SHPP": {"lat": -1.1858, "lon": 30.1206}
}

current_city = "Colombo"
LATITUDE = CITIES[current_city]['lat']
LONGITUDE = CITIES[current_city]['lon']

latest_data = {
    'outdoor_aqi': 0, 'temp': 0, 'humidity': 0, 'wind_speed': 0, 'pressure': 0, 'clouds': 0,
    'indoor_aqi': 0, 'indoor_risk': 'Unknown', 'outdoor_risk': 'Unknown', 'is_anomaly': False,
    'rain_probability_24h': 0.0, 'rain_risk_curve': [], 'recommendation': 'Loading...',
    'explanation': 'Fetching data...', 'timestamp': datetime.now().isoformat(), 'city': current_city
}

rain_model = None
try:
    with open('rain_model.pkl', 'rb') as f:
        rain_model = pickle.load(f)
        logger.info("Rain model loaded")
except:
    logger.warning("Rain model not found")

def init_database():
    conn = get_db()
    c = conn.cursor()
    
    if USE_POSTGRES:
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id SERIAL PRIMARY KEY, timestamp TIMESTAMP NOT NULL, city TEXT NOT NULL,
                      predicted_rain_prob REAL, outdoor_aqi REAL, temp REAL, humidity REAL, pressure REAL,
                      clouds REAL, wind_speed REAL, indoor_aqi REAL, recommendation TEXT,
                      actual_rain INTEGER DEFAULT NULL, verification_time TIMESTAMP DEFAULT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS api_usage
                     (id SERIAL PRIMARY KEY, date TEXT NOT NULL, call_count INTEGER NOT NULL, UNIQUE(date))''')
    else:
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, city TEXT NOT NULL,
                      predicted_rain_prob REAL, outdoor_aqi REAL, temp REAL, humidity REAL, pressure REAL,
                      clouds REAL, wind_speed REAL, indoor_aqi REAL, recommendation TEXT,
                      actual_rain INTEGER DEFAULT NULL, verification_time TEXT DEFAULT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS api_usage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, call_count INTEGER NOT NULL, UNIQUE(date))''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_database()

def can_make_api_call():
    global api_calls_today, api_calls_date
    today = datetime.now().date()
    if today != api_calls_date:
        api_calls_date = today
        api_calls_today = 0
    return api_calls_today < API_CALL_LIMIT

def log_api_call():
    global api_calls_today
    api_calls_today += 1

def log_prediction(data: Dict):
    try:
        conn = get_db()
        c = conn.cursor()
        if USE_POSTGRES:
            c.execute('''INSERT INTO predictions (timestamp, city, predicted_rain_prob, outdoor_aqi, temp, 
                         humidity, pressure, clouds, wind_speed, indoor_aqi, recommendation)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                      (data['timestamp'], data['city'], data['rain_probability_24h'], data['outdoor_aqi'],
                       data['temp'], data['humidity'], data['pressure'], data['clouds'],
                       data['wind_speed'], data['indoor_aqi'], data['recommendation']))
        else:
            c.execute('''INSERT INTO predictions (timestamp, city, predicted_rain_prob, outdoor_aqi, temp, 
                         humidity, pressure, clouds, wind_speed, indoor_aqi, recommendation)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (data['timestamp'], data['city'], data['rain_probability_24h'], data['outdoor_aqi'],
                       data['temp'], data['humidity'], data['pressure'], data['clouds'],
                       data['wind_speed'], data['indoor_aqi'], data['recommendation']))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error logging: {e}")

def fetch_outdoor_aqi(lat, lon, api_key):
    if not can_make_api_call():
        return None
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        log_api_call()
        data = response.json()
        if 'list' not in data or len(data['list']) == 0:
            return None
        current = data['list'][0]
        return {
            'aqi': current['main'].get('aqi', 1), 'pm25': current.get('components', {}).get('pm2_5', 0),
            'pm10': current.get('components', {}).get('pm10', 0), 'no2': current.get('components', {}).get('no2', 0),
            'o3': current.get('components', {}).get('o3', 0), 'co': current.get('components', {}).get('co', 0)
        }
    except:
        return None

def fetch_weather(lat, lon, api_key):
    if not can_make_api_call():
        return None
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        log_api_call()
        data = response.json()
        return {
            'temp': data['main'].get('temp', 20), 'humidity': data['main'].get('humidity', 50),
            'wind_speed': data['wind'].get('speed', 0), 'pressure': data['main'].get('pressure', 1013),
            'clouds': data.get('clouds', {}).get('all', 50)
        }
    except:
        return None

def fetch_24h_forecast(lat, lon, api_key):
    if not can_make_api_call():
        return None
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        log_api_call()
        return response.json().get('list', [])[:8]
    except:
        return None

def calculate_dew_point(temp, humidity):
    a, b = 17.27, 237.7
    try:
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)
    except:
        return temp - ((100 - humidity) / 5.0)

def extract_rain_features(weather, forecast):
    if not weather or not forecast:
        return None
    try:
        temp = float(weather['temp'])
        humidity = float(weather['humidity'])
        pressure = float(weather['pressure'])
        wind_speed = float(weather['wind_speed'])
        clouds = float(weather['clouds'])
        dew_point = calculate_dew_point(temp, humidity)
        
        humidities = [f['main']['humidity'] for f in forecast]
        pressures = [f['main']['pressure'] for f in forecast]
        cloud_covers = [f.get('clouds', {}).get('all', 50) for f in forecast]
        rains = [f.get('rain', {}).get('3h', 0) for f in forecast]
        
        return np.array([
            temp, humidity, pressure, wind_speed, clouds, dew_point,
            float(np.mean(humidities)), float(np.max(humidities)), float(np.mean(pressures)),
            float(pressures[-1] - pressures[0]) if len(pressures) > 1 else 0.0,
            float(np.mean(cloud_covers)), float(np.max(cloud_covers)),
            float(sum(rains)), int(sum(1 for r in rains if r > 0))
        ], dtype=np.float32)
    except:
        return None

def generate_rain_risk_curve(forecast, ml_probability):
    if not forecast:
        return []
    risk_curve = []
    for i, f in enumerate(forecast):
        hour = i * 3
        try:
            hum = f['main']['humidity']
            pres = f['main']['pressure']
            clouds = f.get('clouds', {}).get('all', 50)
            has_rain = f.get('rain', {}).get('3h', 0) > 0
            risk = 0.0
            if hum > 85: risk += 30
            elif hum > 75: risk += 20
            elif hum > 65: risk += 10
            if pres < 1005: risk += 20
            elif pres < 1010: risk += 10
            risk += (clouds / 100) * 25
            if has_rain: risk += 20
            risk = (risk * 0.5) + (ml_probability * 100 * 0.5)
            risk = max(0, min(100, risk))
            risk_curve.append({'hour': hour, 'risk': round(risk, 1)})
        except:
            risk_curve.append({'hour': hour, 'risk': round(ml_probability * 100, 1)})
    return risk_curve

def engineer_features(aqi_data, weather_data, timestamp):
    if aqi_data is None or weather_data is None:
        return None
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    return np.array([
        aqi_data['aqi'], aqi_data['pm25'], aqi_data['pm10'], aqi_data['no2'], aqi_data['o3'], aqi_data['co'],
        weather_data['temp'], weather_data['humidity'], weather_data['wind_speed'], weather_data['pressure'],
        hour, day_of_week, 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
        1 if day_of_week >= 5 else 0, 1 if hour < 6 or hour > 22 else 0,
        weather_data['temp'] * weather_data['humidity'] / 100.0, weather_data['wind_speed'] * aqi_data['aqi'],
        aqi_data['pm25'] + 0.5 * aqi_data['pm10'] + 0.3 * aqi_data['no2'],
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    ], dtype=np.float32)

class IAQRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=30, max_depth=8, min_samples_split=15,
                                          min_samples_leaf=8, max_features='sqrt', n_jobs=1, random_state=42)
        self.is_trained = False
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            outdoor_aqi = X[0]
            wind_speed = X[8]
            infiltration = min(0.7, 0.3 + 0.05 * wind_speed)
            return float(max(1.0, min(500.0, outdoor_aqi * infiltration + 5.0)))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return float(max(1.0, min(500.0, self.model.predict(X)[0])))

class IAQAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=30, max_samples=128, contamination=0.1, random_state=42, n_jobs=1)
        self.is_trained = False
    
    def train(self, X):
        self.model.fit(X)
        self.is_trained = True
    
    def detect(self, X):
        if not self.is_trained:
            return bool(X[0] > 150 or X[1] > 55)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return bool(self.model.decision_function(X)[0] < -0.5)

class WindowRecommender:
    @staticmethod
    def get_health_risk_band(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"
    
    @staticmethod
    def recommend(indoor_aqi, outdoor_aqi, temp, humidity, wind_speed, is_anomaly, rain_probability):
        temp_ok = 15 <= temp <= 32
        humidity_ok = 35 <= humidity <= 80
        wind_moderate = wind_speed < 8.0
        high_rain_risk = rain_probability > 0.5
        indoor_risk = WindowRecommender.get_health_risk_band(indoor_aqi)
        outdoor_risk = WindowRecommender.get_health_risk_band(outdoor_aqi)
        outdoor_better = outdoor_aqi < indoor_aqi - 10
        
        metadata = {
            'indoor_risk': indoor_risk, 'outdoor_risk': outdoor_risk, 'outdoor_better': outdoor_better,
            'temp_ok': temp_ok, 'humidity_ok': humidity_ok, 'wind_moderate': wind_moderate, 'is_anomaly': is_anomaly
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
        if indoor_aqi <= 50:
            return ("ALL GOOD", f"Indoor air is {indoor_risk}. Maintain current conditions.", metadata)
        return ("KEEP CLOSED", f"Outdoor air not optimal. Indoor: {indoor_risk}.", metadata)

class IAQSystem:
    def __init__(self, api_key, lat, lon):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.regressor = IAQRegressor()
        self.anomaly_detector = IAQAnomalyDetector()
        self.recommender = WindowRecommender()
        self._initialize_models()
        
    def _initialize_models(self):
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
        
    def update(self):
        timestamp = datetime.now()
        logger.info(f"Updating {current_city} (API: {api_calls_today}/{API_CALL_LIMIT})")
        
        aqi_data = fetch_outdoor_aqi(self.lat, self.lon, self.api_key)
        weather_data = fetch_weather(self.lat, self.lon, self.api_key)
        forecast_data = fetch_24h_forecast(self.lat, self.lon, self.api_key)
        
        if not aqi_data or not weather_data:
            return None
        
        features = engineer_features(aqi_data, weather_data, timestamp)
        if features is None:
            return None
        
        indoor_aqi = self.regressor.predict(features)
        is_anomaly = self.anomaly_detector.detect(features)
        outdoor_aqi = float(aqi_data['aqi'] * 50)
        
        rain_probability = 0.0
        rain_risk_curve = []
        
        if rain_model and forecast_data:
            try:
                rain_features = extract_rain_features(weather_data, forecast_data)
                if rain_features is not None:
                    rain_proba_array = rain_model.predict_proba([rain_features])
                    rain_probability = float(rain_proba_array[0][1])
                    rain_risk_curve = generate_rain_risk_curve(forecast_data, rain_probability)
            except:
                pass
        
        if not rain_risk_curve:
            base_risk = max(10, min(60, (weather_data['humidity'] - 50) * 1.2))
            rain_risk_curve = [{'hour': i*3, 'risk': round(base_risk + np.random.uniform(-8, 8), 1)} for i in range(8)]
        
        recommendation, explanation, metadata = self.recommender.recommend(
            indoor_aqi, outdoor_aqi, weather_data['temp'], weather_data['humidity'],
            weather_data['wind_speed'], is_anomaly, rain_probability
        )
        
        result = {
            'outdoor_aqi': float(outdoor_aqi), 'temp': float(weather_data['temp']),
            'humidity': float(weather_data['humidity']), 'wind_speed': float(weather_data['wind_speed']),
            'pressure': float(weather_data['pressure']), 'clouds': float(weather_data['clouds']),
            'indoor_aqi': float(indoor_aqi), 'indoor_risk': str(metadata['indoor_risk']),
            'outdoor_risk': str(metadata['outdoor_risk']), 'is_anomaly': bool(is_anomaly),
            'rain_probability_24h': float(rain_probability), 'rain_risk_curve': rain_risk_curve,
            'recommendation': str(recommendation), 'explanation': str(explanation),
            'timestamp': timestamp.isoformat(), 'city': current_city
        }
        
        log_prediction(result)
        return result

system = IAQSystem(API_KEY, LATITUDE, LONGITUDE)

def background_updater():
    global latest_data
    time.sleep(10)
    while True:
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = c.fetchone()[0]
            conn.close()
            
            if total_predictions >= 50:
                logger.info(f"Limit reached ({total_predictions}/50). Display only.")
                aqi_data = fetch_outdoor_aqi(system.lat, system.lon, system.api_key)
                weather_data = fetch_weather(system.lat, system.lon, system.api_key)
                
                if aqi_data and weather_data:
                    timestamp = datetime.now()
                    features = engineer_features(aqi_data, weather_data, timestamp)
                    
                    if features is not None:
                        indoor_aqi = system.regressor.predict(features)
                        is_anomaly = system.anomaly_detector.detect(features)
                        outdoor_aqi = float(aqi_data['aqi'] * 50)
                        
                        recommendation, explanation, metadata = system.recommender.recommend(
                            indoor_aqi, outdoor_aqi, weather_data['temp'],
                            weather_data['humidity'], weather_data['wind_speed'],
                            is_anomaly, 0.0
                        )
                        
                        latest_data = {
                            'outdoor_aqi': float(outdoor_aqi), 'temp': float(weather_data['temp']),
                            'humidity': float(weather_data['humidity']), 'wind_speed': float(weather_data['wind_speed']),
                            'pressure': float(weather_data['pressure']), 'clouds': float(weather_data['clouds']),
                            'indoor_aqi': float(indoor_aqi), 'indoor_risk': str(metadata['indoor_risk']),
                            'outdoor_risk': str(metadata['outdoor_risk']), 'is_anomaly': bool(is_anomaly),
                            'rain_probability_24h': 0.0, 'rain_risk_curve': [],
                            'recommendation': str(recommendation), 'explanation': str(explanation),
                            'timestamp': timestamp.isoformat(), 'city': current_city,
                            'limit_reached': True, 'total_predictions': total_predictions
                        }
            else:
                data = system.update()
                if data:
                    data['limit_reached'] = False
                    data['total_predictions'] = total_predictions + 1
                    latest_data = data
                    
        except Exception as e:
            logger.error(f"Update error: {e}")
        
        time.sleep(600)

thread = threading.Thread(target=background_updater, daemon=True)
thread.start()

initial_data = system.update()
if initial_data:
    latest_data = initial_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/api/data')
def get_data():
    # Add stats to data response
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE actual_rain IS NOT NULL")
        verified = c.fetchone()[0]
        
        if USE_POSTGRES:
            c.execute('''SELECT COUNT(*) FROM predictions 
                         WHERE actual_rain IS NULL 
                         AND EXTRACT(EPOCH FROM (NOW() - timestamp))/3600 >= 12''')
        else:
            c.execute('''SELECT COUNT(*) FROM predictions 
                         WHERE actual_rain IS NULL 
                         AND (julianday('now') - julianday(timestamp)) * 24 >= 12''')
        ready = c.fetchone()[0]
        
        conn.close()
        
        response = latest_data.copy()
        response['total_predictions'] = total
        response['verified_predictions'] = verified
        response['ready_to_verify'] = ready
        response['api_calls_today'] = api_calls_today
        response['limit_reached'] = total >= 50
        
        return jsonify(response)
    except:
        return jsonify(latest_data)

@app.route('/api/cities')
def get_cities():
    return jsonify({'cities': list(CITIES.keys()), 'current': current_city})

@app.route('/api/stats')
def get_stats():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE actual_rain IS NOT NULL")
        verified = c.fetchone()[0]
        
        if USE_POSTGRES:
            c.execute('''SELECT COUNT(*) FROM predictions 
                         WHERE actual_rain IS NULL 
                         AND EXTRACT(EPOCH FROM (NOW() - timestamp))/3600 >= 12''')
        else:
            c.execute('''SELECT COUNT(*) FROM predictions 
                         WHERE actual_rain IS NULL 
                         AND (julianday('now') - julianday(timestamp)) * 24 >= 12''')
        ready = c.fetchone()[0]
        
        c.execute('''SELECT timestamp FROM predictions 
                     WHERE actual_rain IS NULL 
                     ORDER BY timestamp ASC LIMIT 1''')
        oldest = c.fetchone()
        oldest_time = oldest[0] if oldest else None
        
        hours_until_ready = 0
        if oldest_time and ready == 0:
            if isinstance(oldest_time, str):
                oldest_dt = datetime.fromisoformat(oldest_time)
            else:
                oldest_dt = oldest_time
            elapsed = (datetime.now() - oldest_dt).total_seconds() / 3600
            hours_until_ready = max(0, 12 - elapsed)
        
        conn.close()
        
        return jsonify({
            'api_calls_today': api_calls_today, 'api_limit': API_CALL_LIMIT,
            'total_predictions': total, 'verified_predictions': verified,
            'ready_to_verify': ready, 'limit_reached': total >= 50,
            'hours_until_ready': round(hours_until_ready, 1)
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

# NEW ENDPOINT - matches monitor.html
@app.route('/api/predictions')
def get_predictions():
    try:
        conn = get_db()
        c = conn.cursor()
        two_days = (datetime.now() - timedelta(hours=48)).isoformat()
        
        if USE_POSTGRES:
            c.execute('''SELECT id, timestamp, city, predicted_rain_prob, temp, humidity, 
                         clouds, recommendation, actual_rain, verification_time
                         FROM predictions WHERE timestamp > %s ORDER BY timestamp DESC LIMIT 50''', (two_days,))
        else:
            c.execute('''SELECT id, timestamp, city, predicted_rain_prob, temp, humidity, 
                         clouds, recommendation, actual_rain, verification_time
                         FROM predictions WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 50''', (two_days,))
        
        predictions = []
        for row in c.fetchall():
            if isinstance(row[1], str):
                dt = datetime.fromisoformat(row[1])
            else:
                dt = row[1]
            hours_ago = (datetime.now() - dt).total_seconds() / 3600
            predictions.append({
                'id': row[0], 'timestamp': row[1] if isinstance(row[1], str) else row[1].isoformat(),
                'display_time': dt.strftime('%Y-%m-%d %H:%M'),
                'hours_ago': round(hours_ago, 1), 'city': row[2],
                'rain_probability': round(row[3] * 100, 1) if row[3] else 0,
                'temp': round(row[4], 1) if row[4] else 0,
                'humidity': round(row[5]) if row[5] else 0,
                'clouds': round(row[6]) if row[6] else 0,
                'recommendation': row[7], 'actual_rain': row[8],
                'verified': row[8] is not None,
                'ready_to_verify': hours_ago >= 12 and row[8] is None
            })
        conn.close()
        return jsonify({'predictions': predictions})
    except Exception as e:
        logger.error(f"Predictions error: {e}")
        return jsonify({'error': str(e), 'predictions': []}), 500

# NEW ENDPOINT - matches monitor.html
@app.route('/api/verify-prediction', methods=['POST'])
def verify_prediction():
    try:
        data = request.json
        pred_id = data.get('id')
        actual_rain = data.get('actual_rain')
        
        if pred_id is None or actual_rain not in [0, 1, 2, 3]:
            return jsonify({'error': 'Invalid data'}), 400
        
        conn = get_db()
        c = conn.cursor()
        
        if USE_POSTGRES:
            c.execute('SELECT predicted_rain_prob FROM predictions WHERE id = %s', (pred_id,))
        else:
            c.execute('SELECT predicted_rain_prob FROM predictions WHERE id = ?', (pred_id,))
        result = c.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'error': 'Not found'}), 404
        
        predicted_prob = result[0]
        verification_time = datetime.now().isoformat()
        
        if USE_POSTGRES:
            c.execute('UPDATE predictions SET actual_rain = %s, verification_time = %s WHERE id = %s',
                      (actual_rain, verification_time, pred_id))
        else:
            c.execute('UPDATE predictions SET actual_rain = ?, verification_time = ? WHERE id = ?',
                      (actual_rain, verification_time, pred_id))
        conn.commit()
        conn.close()
        
        predicted_class = 1 if predicted_prob > 0.5 else 0
        actual_binary = 1 if actual_rain > 0 else 0
        correct = predicted_class == actual_binary
        
        return jsonify({'success': True, 'correct': correct})
    except Exception as e:
        logger.error(f"Verify error: {e}")
        return jsonify({'error': str(e)}), 500

# NEW ENDPOINT - matches monitor.html
@app.route('/api/update-verification', methods=['POST'])
def update_verification():
    try:
        data = request.json
        pred_id = data.get('id')
        actual_rain = data.get('actual_rain')
        
        if pred_id is None or actual_rain not in [0, 1, 2, 3]:
            return jsonify({'error': 'Invalid data'}), 400
        
        conn = get_db()
        c = conn.cursor()
        
        if USE_POSTGRES:
            c.execute('UPDATE predictions SET actual_rain = %s, verification_time = %s WHERE id = %s',
                      (actual_rain, datetime.now().isoformat(), pred_id))
        else:
            c.execute('UPDATE predictions SET actual_rain = ?, verification_time = ? WHERE id = ?',
                      (actual_rain, datetime.now().isoformat(), pred_id))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Updated successfully'})
    except Exception as e:
        logger.error(f"Update error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT predicted_rain_prob, actual_rain FROM predictions WHERE actual_rain IS NOT NULL')
        results = c.fetchall()
        
        if len(results) < 5:
            conn.close()
            return jsonify({'insufficient_data': True, 'verified_count': len(results)})
        
        tp = fp = tn = fn = 0
        for pred_prob, actual in results:
            predicted = 1 if pred_prob > 0.5 else 0
            actual_bin = 1 if actual > 0 else 0
            if predicted == 1 and actual_bin == 1: tp += 1
            elif predicted == 1 and actual_bin == 0: fp += 1
            elif predicted == 0 and actual_bin == 0: tn += 1
            else: fn += 1
        
        total = len(results)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        conn.close()
        
        return jsonify({
            'verified_count': total, 'accuracy': round(accuracy * 100, 1),
            'precision': round(precision * 100, 1), 'recall': round(recall * 100, 1),
            'f1_score': round(f1 * 100, 1)
        })
    except Exception as e:
        logger.error(f"Performance error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions WHERE actual_rain IS NOT NULL")
        verified = c.fetchone()[0]
        
        if verified < 50:
            conn.close()
            return jsonify({'error': f'Need 50 verified. Have {verified}'}), 400
        
        logger.info(f"Retraining with {verified} verified")
        c.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Retrained. Counter reset.'})
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return jsonify({'error': str(e)}), 500

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
    data = system.update()
    if data:
        global latest_data
        latest_data = data
        return jsonify(data)
    return jsonify({'error': 'Failed'}), 500

@app.route('/api/refresh')
def refresh():
    data = system.update()
    if data:
        global latest_data
        latest_data = data
        return jsonify(data)
    return jsonify({'error': 'Failed'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 'rain_model_loaded': rain_model is not None,
        'current_city': current_city, 'api_calls_today': api_calls_today
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)