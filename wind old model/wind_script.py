# ============================================================
# PHASE 1: TRAINING SCRIPT (Run Once)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv("Wind data_texas.csv")

df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.sort_values("Timestamp").reset_index(drop=True)
features = ["Wind_speed", "Wind_direction", "Pressure", "Air_temperature"]
target = "windfarm_power"
data = df[features + [target]].copy()

# Scale all features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled, columns=data.columns)

SEQ_LEN = 24
def create_sequences(dataset, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(dataset) - seq_len):
        X.append(dataset.iloc[i:i+seq_len][features].values)
        y.append(dataset.iloc[i+seq_len][target])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_df)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# -----------------------------
# Build LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=40, batch_size=16, callbacks=[early_stop], verbose=1)

# -----------------------------
# Save Model and Scaler
# -----------------------------
model.save("wind_forecast_model.keras")
joblib.dump(scaler, "wind_scaler.pkl")

print("✅ Model and scaler saved successfully!")
