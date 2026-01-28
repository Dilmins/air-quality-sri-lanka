import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import os

np.random.seed(42)

print("="*70)
print("RAIN PREDICTION MODEL TRAINING")
print("="*70)

# Generate realistic meteorological dataset with noise and edge cases
n_samples = 20000  # Reduced for your hardware
X = np.zeros((n_samples, 14))
y = np.zeros(n_samples)

print("\nGenerating realistic weather scenarios with noise...")

# More nuanced scenario distributions with overlapping conditions
scenarios = {
    'clear': {'weight': 0.42, 'rain_prob': 0.02},  # Sometimes light rain even when clear
    'partly_cloudy': {'weight': 0.26, 'rain_prob': 0.05},
    'cloudy': {'weight': 0.16, 'rain_prob': 0.15},
    'pre_rain': {'weight': 0.11, 'rain_prob': 0.60},  # Not always rain
    'rain': {'weight': 0.05, 'rain_prob': 0.95}  # Not perfect
}

for i in range(n_samples):
    scenario_type = np.random.choice(
        list(scenarios.keys()),
        p=[s['weight'] for s in scenarios.values()]
    )
    
    # Add random noise to make data less deterministic
    noise_factor = np.random.uniform(0.85, 1.15)
    
    if scenario_type == 'clear':
        temp = np.random.normal(28, 5) * noise_factor
        hum = np.random.normal(55, 12)
        pres = np.random.normal(1018, 6)
        wind = np.random.uniform(0, 6)
        clouds = np.random.uniform(0, 35)
        
    elif scenario_type == 'partly_cloudy':
        temp = np.random.normal(26, 4) * noise_factor
        hum = np.random.normal(65, 10)
        pres = np.random.normal(1013, 5)
        wind = np.random.uniform(1, 8)
        clouds = np.random.uniform(25, 65)
        
    elif scenario_type == 'cloudy':
        temp = np.random.normal(24, 4) * noise_factor
        hum = np.random.normal(73, 9)
        pres = np.random.normal(1008, 6)
        wind = np.random.uniform(2, 10)
        clouds = np.random.uniform(55, 90)
        
    elif scenario_type == 'pre_rain':
        temp = np.random.normal(22, 4) * noise_factor
        hum = np.random.normal(82, 8)
        pres = np.random.normal(1002, 6)
        wind = np.random.uniform(3, 12)
        clouds = np.random.uniform(70, 100)
        
    else:  # rain
        temp = np.random.normal(21, 4) * noise_factor
        hum = np.random.normal(90, 6)
        pres = np.random.normal(996, 7)
        wind = np.random.uniform(4, 15)
        clouds = np.random.uniform(80, 100)
    
    # Ensure realistic bounds
    temp = np.clip(temp, 15, 40)
    hum = np.clip(hum, 30, 100)
    pres = np.clip(pres, 980, 1030)
    wind = np.clip(wind, 0, 20)
    clouds = np.clip(clouds, 0, 100)
    
    # Calculate dew point with slight errors
    a, b = 17.27, 237.7
    try:
        alpha = ((a * temp) / (b + temp)) + np.log(hum / 100.0)
        dew_point = (b * alpha) / (a - alpha)
    except:
        dew_point = temp - ((100 - hum) / 5.0)
    
    # Determine rain label with probabilistic approach
    rain_label = np.random.choice([0, 1], p=[1-scenarios[scenario_type]['rain_prob'], 
                                              scenarios[scenario_type]['rain_prob']])
    
    # Generate forecast features with MORE noise and less perfect correlation
    forecast_noise = np.random.uniform(0.9, 1.1)
    measurement_error = np.random.normal(0, 2)
    
    if rain_label == 1:
        mean_hum_24h = hum + np.random.normal(4, 6) + measurement_error
        max_hum_24h = mean_hum_24h + np.random.uniform(2, 10)
        mean_pres_24h = pres + np.random.normal(-3, 4)
        pres_trend = np.random.normal(-6, 5)
        mean_clouds_24h = clouds + np.random.normal(3, 6)
        max_clouds_24h = mean_clouds_24h + np.random.uniform(1, 10)
        total_rain = np.random.uniform(5, 30) if np.random.random() > 0.2 else np.random.uniform(0, 5)
        rain_steps = np.random.randint(2, 8) if total_rain > 5 else np.random.randint(0, 3)
    else:
        mean_hum_24h = hum + np.random.normal(0, 8) + measurement_error
        max_hum_24h = mean_hum_24h + np.random.uniform(0, 8)
        mean_pres_24h = pres + np.random.normal(0, 4)
        pres_trend = np.random.normal(0, 6)
        mean_clouds_24h = clouds + np.random.normal(0, 8)
        max_clouds_24h = mean_clouds_24h + np.random.uniform(0, 7)
        total_rain = np.random.uniform(0, 3) if np.random.random() > 0.85 else 0
        rain_steps = 1 if total_rain > 1 else 0
    
    # Ensure bounds
    mean_hum_24h = np.clip(mean_hum_24h, 25, 100)
    max_hum_24h = np.clip(max_hum_24h, mean_hum_24h, 100)
    mean_clouds_24h = np.clip(mean_clouds_24h, 0, 100)
    max_clouds_24h = np.clip(max_clouds_24h, mean_clouds_24h, 100)
    total_rain = max(0, total_rain)
    
    X[i] = [
        temp, hum, pres, wind, clouds, dew_point,
        mean_hum_24h, max_hum_24h, mean_pres_24h, pres_trend,
        mean_clouds_24h, max_clouds_24h, total_rain, rain_steps
    ]
    
    y[i] = rain_label

y = y.astype(int)

# Shuffle the data to ensure randomness
shuffle_idx = np.random.permutation(n_samples)
X = X[shuffle_idx]
y = y[shuffle_idx]


rain_count = y.sum()
no_rain_count = len(y) - rain_count

print(f"\nDataset Statistics:")
print(f"  Total samples: {n_samples:,}")
print(f"  Rain samples: {rain_count:,} ({rain_count/n_samples*100:.1f}%)")
print(f"  No-rain samples: {no_rain_count:,} ({no_rain_count/n_samples*100:.1f}%)")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y, shuffle=True
)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Train Random Forest with proper regularization
print("\nTraining Random Forest Classifier with regularization...")
rf = RandomForestClassifier(
    n_estimators=100,           # Moderate number of trees
    max_depth=12,               # Limit tree depth
    min_samples_split=20,       # Require more samples to split
    min_samples_leaf=10,        # Require more samples in leaves
    max_features='sqrt',        # Use subset of features
    max_samples=0.8,           # Bootstrap with 80% of data
    class_weight='balanced',    # Handle class imbalance
    bootstrap=True,
    oob_score=True,            # Out-of-bag score for validation
    n_jobs=-1,                 # Use all CPU cores
    random_state=42,
    verbose=0
)

rf.fit(X_train, y_train)

print(f"Out-of-bag score: {rf.oob_score_:.4f}")

# Evaluate on test set
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("\nModel Performance on Test Set:")
print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'], digits=3))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:,}    False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}    True Positives:  {tp:,}")
print(f"\nAccuracy: {(tn+tp)/(tn+fp+fn+tp):.3f}")
print(f"Precision (Rain): {tp/(tp+fp) if (tp+fp) > 0 else 0:.3f}")
print(f"Recall (Rain): {tp/(tp+fn) if (tp+fn) > 0 else 0:.3f}")

# Calibrate probabilities for better probability estimates
print("\nCalibrating probability predictions...")
calibrated_model = CalibratedClassifierCV(
    rf, 
    method='isotonic',  # Better for non-parametric calibration
    cv=3,              # Reduced for smaller dataset
    n_jobs=-1
)

calibrated_model.fit(X_train, y_train)

# Test calibrated model
y_cal_proba = calibrated_model.predict_proba(X_test)[:, 1]
cal_roc_auc = roc_auc_score(y_test, y_cal_proba)
print(f"Calibrated ROC AUC Score: {cal_roc_auc:.4f}")

# Test realistic scenarios
print("\n" + "="*70)
print("TESTING MODEL ON REALISTIC SCENARIOS")
print("="*70)

test_scenarios = [
    ("Clear sunny day", [30, 50, 1020, 3, 15, 20, 52, 60, 1020, 2, 18, 28, 0, 0]),
    ("Partly cloudy", [27, 65, 1015, 4, 45, 22, 68, 75, 1015, 0, 50, 65, 0, 0]),
    ("Overcast conditions", [25, 75, 1010, 5, 80, 23, 78, 85, 1009, -2, 82, 90, 0.5, 1]),
    ("Pre-rain conditions", [23, 82, 1005, 7, 88, 22, 85, 92, 1003, -5, 88, 95, 3, 2]),
    ("Light rain expected", [21, 88, 1000, 8, 92, 20, 90, 95, 998, -7, 92, 97, 12, 5]),
    ("Heavy rain conditions", [20, 93, 995, 10, 97, 19, 94, 97, 992, -10, 95, 99, 25, 7]),
]

for scenario_name, features in test_scenarios:
    prob = calibrated_model.predict_proba([features])[0][1]
    print(f"  {scenario_name:25s} → {prob*100:5.1f}% rain probability")

# Feature importance
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)

feature_names = [
    "Temperature", "Humidity", "Pressure", "Wind Speed", "Cloud Cover",
    "Dew Point", "24h Mean Humidity", "24h Max Humidity", "24h Mean Pressure",
    "24h Pressure Trend", "24h Mean Clouds", "24h Max Clouds",
    "Total Forecast Rain", "Rain Steps Count"
]

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices, 1):
    print(f"  {i:2d}. {feature_names[idx]:25s} {importances[idx]:.4f}")

# Save model
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

with open('rain_model.pkl', 'wb') as f:
    pickle.dump(calibrated_model, f)

print("Model saved to: rain_model.pkl")
print(f"Model size: {os.path.getsize('rain_model.pkl') / 1024:.1f} KB")
print("\nTraining complete!")
print("="*70)