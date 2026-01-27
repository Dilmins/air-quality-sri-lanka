# Run this ONCE locally to generate the model file
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle

# Generate synthetic training data (replace with real data in production)
n_samples = 5000
X_train = np.random.randn(n_samples, 14)
y_train = (X_train[:, 1] > 0.5) & (X_train[:, 6] > 100)  # Simplified rule

# Train model
rf = RandomForestClassifier(
    n_estimators=60,
    max_depth=8,
    min_samples_leaf=25,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# Calibrate
calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
calibrated.fit(X_train, y_train)

# Save
with open('Backend/rain_model.pkl', 'wb') as f:
    pickle.dump(calibrated, f)

print("✓ Model saved to Backend/rain_model.pkl")
