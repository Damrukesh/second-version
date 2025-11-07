import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("wind_forecast_model.keras", compile=False)
    scaler = joblib.load("wind_scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("💨 Wind Farm Power Prediction")
st.write("Upload a CSV containing recent **wind condition data** to predict the expected windfarm power output (MW).")

st.markdown("""
**Expected CSV Columns:**  
`Timestamp`, `Wind_speed`, `Wind_direction`, `Pressure`, `Air_temperature`
""")

uploaded_file = st.file_uploader("📂 Upload your wind feature CSV", type=["csv"])

if uploaded_file:
    try:
        # ------------------------------------------------------------
        # Load and preprocess data
        # ------------------------------------------------------------
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
        
        # optional timestamp parsing
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        else:
            df["Timestamp"] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")

        features = ["Wind_speed", "Wind_direction", "Pressure", "Air_temperature"]
        if not all(f in df.columns for f in features):
            st.error(f"❌ Missing required columns. Found: {list(df.columns)}")
            st.stop()

        # scale only features
        scaled_features = scaler.transform(df[features])
        scaled_features = np.expand_dims(scaled_features, axis=1) if len(scaled_features.shape) == 2 else scaled_features

        # ------------------------------------------------------------
        # Predict
        # ------------------------------------------------------------
        X_input = np.expand_dims(scaled_features, axis=0) if len(scaled_features.shape) == 2 else scaled_features
        preds_scaled = model.predict(X_input)
        preds_scaled = preds_scaled.reshape(-1, 1)

        # inverse-transform to MW
        dummy = np.zeros((preds_scaled.shape[0], len(features) + 1))
        dummy[:, -1] = preds_scaled.flatten()
        inv = scaler.inverse_transform(dummy)
        preds_MW = inv[:, -1]

        # ------------------------------------------------------------
        # Display results
        # ------------------------------------------------------------
        result_df = pd.DataFrame({
            "Timestamp": df["Timestamp"],
            "Predicted_Windfarm_Power_MW": preds_MW.round(2)
        })

        st.subheader("🔮 Predicted Windfarm Power Output (MW)")
        st.line_chart(result_df.set_index("Timestamp"))
        st.dataframe(result_df)

        # Download option
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name="predicted_windfarm_power.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
    st.write("📋 All columns in file:", list(df.columns))
    st.write("🎯 Feature columns extracted:", features)
    st.write("🧮 Shape of df[features]:", df[features].shape)
    

st.markdown("---")
st.caption("Developed for Final Year Project — Wind Energy Forecasting & Carbon-Aware Management")
