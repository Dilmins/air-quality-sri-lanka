import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle

print("🌧️ Creating Realistic Rain Prediction Model...")

np.random.seed(42)

# Generate 20,000 realistic weather samples based on actual meteorological patterns
n = 20000
X = np.zeros((n, 14))
y = np.zeros(n)

for i in range(n):
    # Generate realistic weather scenarios
    # Scenario distribution: 70% clear, 20% cloudy, 10% rainy
    scenario = np.random.choice(['clear', 'cloudy', 'rainy'], p=[0.70, 0.20, 0.10])
    
    if scenario == 'clear':
        temp = np.random.uniform(20, 35)
        hum = np.random.uniform(40, 70)
        pres = np.random.uniform(1010, 1025)
        wind = np.random.uniform(1, 6)
        clouds = np.random.uniform(0, 40)
    elif scenario == 'cloudy':
        temp = np.random.uniform(18, 30)
        hum = np.random.uniform(60, 85)
        pres = np.random.uniform(1005, 1015)
        wind = np.random.uniform(2, 8)
        clouds = np.random.uniform(50, 85)
    else:  # rainy
        temp = np.random.uniform(15, 28)
        hum = np.random.uniform(75, 98)
        pres = np.random.uniform(990, 1008)
        wind = np.random.uniform(3, 12)
        clouds = np.random.uniform(80, 100)
    
    # Calculate dew point (approximation)
    dew_point = temp - ((100 - hum) / 5.0)
    
    # 24h forecast simulation
    if scenario == 'rainy':
        mean_hum_24h = np.random.uniform(80, 95)
        max_hum_24h = np.random.uniform(85, 98)
        mean_pres_24h = pres + np.random.uniform(-3, 2)
        pres_trend = np.random.uniform(-8, -2)
        mean_clouds_24h = np.random.uniform(75, 95)
        max_clouds_24h = np.random.uniform(85, 100)
        total_rain = np.random.uniform(5, 30)
        rain_steps = np.random.randint(3, 8)
    elif scenario == 'cloudy':
        mean_hum_24h = np.random.uniform(65, 80)
        max_hum_24h = np.random.uniform(70, 88)
        mean_pres_24h = pres + np.random.uniform(-2, 2)
        pres_trend = np.random.uniform(-4, 1)
        mean_clouds_24h = np.random.uniform(60, 80)
        max_clouds_24h = np.random.uniform(70, 90)
        total_rain = np.random.uniform(0, 5)
        rain_steps = np.random.randint(0, 3)
    else:  # clear
        mean_hum_24h = np.random.uniform(45, 65)
        max_hum_24h = np.random.uniform(55, 75)
        mean_pres_24h = pres + np.random.uniform(-1, 3)
        pres_trend = np.random.uniform(-2, 4)
        mean_clouds_24h = np.random.uniform(20, 50)
        max_clouds_24h = np.random.uniform(30, 60)
        total_rain = 0
        rain_steps = 0
    
    X[i] = [
        temp, hum, pres, wind, clouds, dew_point,
        mean_hum_24h, max_hum_24h, mean_pres_24h, pres_trend,
        mean_clouds_24h, max_clouds_24h, total_rain, rain_steps
    ]
    
    # Label: 1 if rain scenario, 0 otherwise
    y[i] = 1 if scenario == 'rainy' else 0

y = y.astype(int)

print(f"📊 Dataset: {n} samples, {y.sum()} rain ({y.sum()/n*100:.1f}%), {n-y.sum()} no-rain ({(n-y.sum())/n*100:.1f}%)")

# Train model
print("🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)

print("📈 Calibrating probabilities...")
model = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
model.fit(X, y)

# Test
print("\n🧪 Test Cases:")
tests = [
    ("Sunny", [28, 55, 1018, 4, 25, 19, 58, 65, 1018, 1, 30, 40, 0, 0]),
    ("Cloudy", [25, 72, 1012, 5, 70, 20, 75, 82, 1011, -2, 72, 85, 0.5, 1]),
    ("Rainy", [23, 88, 1002, 8, 95, 21, 90, 96, 1000, -6, 92, 98, 15, 5]),
]
for name, feat in tests:
    prob = model.predict_proba([feat])[0][1] * 100
    print(f"   {name:10s} → {prob:5.1f}%")

with open('rain_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model saved to rain_model.pkl")