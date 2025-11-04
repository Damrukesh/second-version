import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
CSV_PATH = 'demand_data_texas.csv'
FEATURE_COLS = ['Temperature', 'Humidity']
TARGET_COL = 'Demand'
LOOKBACK = 24          # past 24 hours to predict next hour
TEST_SIZE = 0.2        # last 20% as test
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MODEL_OUT = 'demand_forecast_lstm.h5'


# ----------------------------
# Load and clean data
# ----------------------------
df = pd.read_csv(CSV_PATH)

# Build true timestamp: date (in 'Timestamp') + hour column
# If your file already has full datetimes, this will still work as long as 'hour' exists
if 'hour' in df.columns:
    # Coerce to datetime using your format; adjust if needed
    # Example format in your dataset is like '01-Jan-20'
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%b-%y', errors='coerce') + pd.to_timedelta(df['hour'], unit='h')
else:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Force numeric for features and target
for col in FEATURE_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sort and drop rows with missing core fields
df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.dropna(subset=['Timestamp'] + FEATURE_COLS + [TARGET_COL])

# Remove inf/-inf if any
df = df[np.isfinite(df[FEATURE_COLS + [TARGET_COL]]).all(axis=1)]

# Final sanity check
if len(df) < LOOKBACK + 10:
    raise ValueError("Not enough cleaned rows after preprocessing to build sequences.")


# ----------------------------
# Build sequences BEFORE scaling
# ----------------------------
features_all = df[FEATURE_COLS].values.astype('float32')
target_all = df[[TARGET_COL]].values.astype('float32')

def create_dataset(features, target, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])  # shape (1,)
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

X_all, y_all = create_dataset(features_all, target_all, LOOKBACK)

# Time-ordered split
split = int(len(X_all) * (1 - TEST_SIZE))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]


# ----------------------------
# Scale with train-only fit
# ----------------------------
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

n_steps, n_feats = X_train.shape[1], X_train.shape[2]

# Flatten time dimension to fit scaler
X_train_2d = X_train.reshape(-1, n_feats)
X_test_2d = X_test.reshape(-1, n_feats)

X_train_scaled_2d = feature_scaler.fit_transform(X_train_2d)
X_test_scaled_2d = feature_scaler.transform(X_test_2d)

X_train = X_train_scaled_2d.reshape(-1, n_steps, n_feats)
X_test = X_test_scaled_2d.reshape(-1, n_steps, n_feats)

y_train = target_scaler.fit_transform(y_train)
y_test = target_scaler.transform(y_test)

# Safety checks
assert np.isfinite(X_train).all() and np.isfinite(y_train).all(), "NaN/inf in train data"
assert np.isfinite(X_test).all() and np.isfinite(y_test).all(), "NaN/inf in test data"


# ----------------------------
# Build and train model
# ----------------------------
tf.keras.backend.set_floatx('float32')

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(n_steps, n_feats)),
    Dropout(0.2),
    Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ],
    verbose=1
)

# Save model
model.save(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")


# ----------------------------
# Evaluate
# ----------------------------
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_true = target_scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.3f}")


# ----------------------------
# Plots
# ----------------------------
# Loss curves
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Actual vs Predicted (sample)
plt.figure(figsize=(12, 4))
window = min(200, len(y_true))
plt.plot(y_true[:window], label='Actual')
plt.plot(y_pred[:window], label='Predicted')
plt.title('LSTM: Actual vs Predicted (sample)')
plt.xlabel('Sample'); plt.ylabel('Demand (MW)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()