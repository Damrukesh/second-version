import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# ----------------------------
# Load Model and Scalers
# ----------------------------
MODEL_PATH = "demand_forecast_lstm.keras"  # or .keras if you converted it
TARGET_SCALER_PATH = "target_scaler.pkl"
FEATURE_SCALER_PATH = "feature_scaler.pkl"

@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    return model, target_scaler, feature_scaler

try:
    model, target_scaler, feature_scaler = load_model_and_scalers()
except FileNotFoundError as e:
    st.error(f"Missing required files: {e}. Please run the training script first to generate model and scalers.")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("⚡ Wind Energy Demand Forecasting")
st.write("Upload your Temperature & Humidity CSV file to forecast electricity demand using the trained LSTM model.")

st.markdown("""
**Expected CSV format:**  
`Timestamp, hour, Temperature, Humidity`  
(24 or more hourly records recommended for sequence creation)
""")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        # Check required columns
        required_cols = ["Temperature", "Humidity"]
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain 'Temperature' and 'Humidity' columns.")
        else:
            # Preprocess: build true timestamp if needed
            if 'hour' in df.columns and 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%b-%y', errors='coerce') + pd.to_timedelta(df['hour'], unit='h')
            
            # Clean data
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            df = df[np.isfinite(df[required_cols]).all(axis=1)]
            
            if len(df) < 24:
                st.warning("Need at least 24 hours of data for LSTM prediction. Using available data.")
            
            # Use saved feature scaler (don't fit a new one!)
            features_scaled = feature_scaler.transform(df[required_cols].values.astype('float32'))
            
            # Reshape for LSTM: [1, timesteps, features]
            # For single prediction, use last 24 hours
            lookback = 24
            if len(features_scaled) >= lookback:
                X_input = features_scaled[-lookback:].reshape(1, lookback, len(required_cols))
            else:
                # Pad with first value if not enough data
                padding = np.tile(features_scaled[0], (lookback - len(features_scaled), 1))
                X_input = np.vstack([padding, features_scaled]).reshape(1, lookback, len(required_cols))
            
            # Predict (gives normalized 0-1)
            y_pred_scaled = model.predict(X_input, verbose=0)
            
            # Inverse transform to get actual MW
            y_pred_mw = target_scaler.inverse_transform(y_pred_scaled)
            
            st.success(f"🔮 Predicted Next Hour Demand: **{y_pred_mw[0][0]:.2f} MW**")
            
            # Optional visualization
            st.line_chart(df[required_cols])
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Developed for Final Year Project — Wind Power Forecasting & Energy Management")