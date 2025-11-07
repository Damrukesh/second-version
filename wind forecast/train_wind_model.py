# # train_wind_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

# ==============================
# Load & Clean Data
# ==============================
df = pd.read_csv("wind_data.csv")

# Fix column name casing and timestamp parsing
df.rename(columns=lambda x: x.strip().title(), inplace=True)

# Add a year to parse dates correctly
df["Timestamp"] = pd.to_datetime("2025 " + df["Timestamp"], errors="coerce")

# Drop rows with bad timestamps
df = df.dropna(subset=["Timestamp"])

# Sort by time
df = df.sort_values("Timestamp")

# ==============================
# Features & Target
# ==============================
# Check the actual columns
print("Columns available:", df.columns.tolist())

X = df[["Wind Speed", "Wind Direction", "Pressure", "Air Temperature"]]
y = df["Windfarm Power"]

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Scale Data
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Try Multiple Models
# ==============================
models = {
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.1),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=300)
}

best_model = None
best_score = -np.inf

print("\n🔍 Model Performance Summary:")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name:15} → R²: {r2:.4f}, RMSE: {rmse:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

print("\n✅ Best model selected:", best_model.__class__.__name__)

# ==============================
# Save Model and Scaler
# ==============================
with open("wind_forecast_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🎯 Model saved as wind_forecast_model.pkl and scaler.pkl")
