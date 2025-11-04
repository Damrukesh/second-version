import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Same config as training
CSV_PATH = 'demand_data_texas.csv'
FEATURE_COLS = ['Temperature', 'Humidity']
TARGET_COL = 'Demand'
LOOKBACK = 24
TEST_SIZE = 0.2

# Load and preprocess (same as training script)
df = pd.read_csv(CSV_PATH)

if 'hour' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%b-%y', errors='coerce') + pd.to_timedelta(df['hour'], unit='h')
else:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

for col in FEATURE_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.dropna(subset=['Timestamp'] + FEATURE_COLS + [TARGET_COL])
df = df[np.isfinite(df[FEATURE_COLS + [TARGET_COL]]).all(axis=1)]

# Build sequences (same as training)
features_all = df[FEATURE_COLS].values.astype('float32')
target_all = df[[TARGET_COL]].values.astype('float32')

def create_dataset(features, target, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

X_all, y_all = create_dataset(features_all, target_all, LOOKBACK)

# Split (same as training)
split = int(len(X_all) * (1 - TEST_SIZE))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# Create and fit scalers (same as training)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

n_steps, n_feats = X_train.shape[1], X_train.shape[2]
X_train_2d = X_train.reshape(-1, n_feats)

feature_scaler.fit(X_train_2d)
target_scaler.fit(y_train)

# Save scalers
joblib.dump(target_scaler, 'target_scaler.pkl')
joblib.dump(feature_scaler, 'feature_scaler.pkl')
print("✅ Saved scalers: target_scaler.pkl and feature_scaler.pkl")