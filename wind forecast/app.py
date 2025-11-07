import streamlit as st
import pandas as pd
import pickle

# ==============================
# Load model & scaler
# ==============================
with open("wind_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="Wind Power Forecasting", page_icon="💨", layout="centered")

st.title("💨 Wind Farm Power Prediction")
st.write("Upload a CSV file containing the following columns:")
st.markdown("**Wind Speed, Wind Direction, Pressure, Air Temperature**")

# File uploader
uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)

    # Normalize column names
    data.rename(columns=lambda x: x.strip().title(), inplace=True)

    required_columns = ["Wind Speed", "Wind Direction", "Pressure", "Air Temperature"]
    missing_cols = [col for col in required_columns if col not in data.columns]

    if missing_cols:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
    else:
        # ✅ Show only the required input columns
        st.write("### ✅ Uploaded Input Data (for prediction)")
        st.dataframe(data[required_columns].head())

        # Scale features
        scaled_data = scaler.transform(data[required_columns])

        # Predict
        predictions = model.predict(scaled_data)
        data["Predicted Windfarm Power (MW)"] = predictions

        # ⚡ Show results
        st.write("### ⚡ Predicted Power Output (MW)")
        st.dataframe(data[required_columns + ["Predicted Windfarm Power (MW)"]])

        # Download button
        csv = data[required_columns + ["Predicted Windfarm Power (MW)"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="wind_power_predictions.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Please upload a CSV file to begin.")
