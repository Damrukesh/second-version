from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import os, joblib, numpy as np, pandas as pd, json, pickle, sys
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Add parent directory to Python path to import demand_predict_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demand_predict_helper import predict_demand  # Import the demand prediction function

app = Flask(__name__)
app.secret_key = "dev-secret-key"  # change for production

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(os.path.dirname(BASE_DIR), "datasets")

# Load Texas energy portfolio data
def load_texas_energy_data():
    try:
        texas_data = pd.read_csv(os.path.join(DATA_FOLDER, "texas_energy_portfolio.csv"))
        texas_data['Timestamp'] = pd.to_datetime(texas_data['Timestamp'])
        return texas_data
    except Exception as e:
        print(f"Error loading Texas energy data: {e}")
        return None

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper: safe model loader
def load_keras_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print("Model load error:", e)
        return None

# Simple fallback predictor if model not available: repeat last value or mean
def fallback_predict_series(values, steps=24):
    # values: pandas Series of recent target values
    last = float(values.iloc[-24:].mean()) if len(values) >= 24 else float(values.mean())
    return [last]*steps

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_type = request.form.get('input_type', 'csv')
    
    if input_type == 'csv':
        # Handle CSV file upload
        wind_file = request.files.get("wind_csv")
        demand_file = request.files.get("demand_csv")
        
        if not wind_file or not demand_file:
            flash("Please upload both Wind and Demand CSV files.", "danger")
            return redirect(url_for("index"))

        # Save uploaded files
        wind_path = os.path.join(UPLOAD_FOLDER, "uploaded_wind.csv")
        demand_path = os.path.join(UPLOAD_FOLDER, "uploaded_demand.csv")
        wind_file.save(wind_path)
        demand_file.save(demand_path)

        # Load as DataFrames (robust parsing)
        try:
            wind_df = pd.read_csv(wind_path)
            demand_df = pd.read_csv(demand_path)
        except Exception as e:
            flash(f"Error reading CSVs: {e}", "danger")
            return redirect(url_for("index"))
            
    elif input_type == 'manual':
        # Handle manual input (single set of slider values with date and hour selection)
        try:
            # Get slider values
            wind_speed = float(request.form.get('manual_wind_speed', 5.5))
            wind_dir = float(request.form.get('manual_wind_dir', 180))
            pressure = float(request.form.get('manual_pressure', 1013))
            air_temp = float(request.form.get('manual_air_temp', 20))
            demand_temp = float(request.form.get('manual_temp', 20))
            humidity = float(request.form.get('manual_humidity', 65))
            selected_date = request.form.get('manual_date')
            selected_hour = int(request.form.get('manual_hour', 12))
            
            if not selected_date:
                flash("Please select a date", "danger")
                return redirect(url_for("index"))
            
            # Parse the selected date and add the selected hour
            selected_datetime = pd.to_datetime(selected_date) + pd.Timedelta(hours=selected_hour)
            
            # Create 24 hourly entries with the same values (models need 24 entries)
            wind_data = []
            demand_data = []
            
            for hour in range(24):
                timestamp = selected_datetime.replace(hour=hour)
                
                wind_data.append({
                    'Wind Speed': wind_speed,
                    'Wind Direction': wind_dir,
                    'Pressure': pressure,
                    'Air Temperature': air_temp,
                })
                
                demand_data.append({
                    'Temperature': demand_temp,
                    'Humidity': humidity,
                    'Timestamp': timestamp
                })
            
            # Convert to DataFrames
            wind_df = pd.DataFrame(wind_data)
            demand_df = pd.DataFrame(demand_data)

            # Store selected datetime and hour for later use
            request.selected_datetime = selected_datetime
            request.selected_hour = selected_hour
                
        except Exception as e:
            flash(f"Error processing manual input: {str(e)}", "danger")
            print(f"Manual input error: {e}")
            return redirect(url_for("index"))
    else:
        flash("Invalid input type", "danger")
        return redirect(url_for("index"))

    # Basic column normalization: replace spaces with underscores and lowercase
    wind_df.columns = [c.strip().replace(" ", "_").lower() for c in wind_df.columns]
    demand_df.columns = [c.strip().replace(" ", "_").lower() for c in demand_df.columns]
    
    # Log the columns for debugging
    print(f"Wind CSV columns: {list(wind_df.columns)}")
    print(f"Demand CSV columns: {list(demand_df.columns)}")
    print(f"Wind CSV shape: {wind_df.shape}")
    print(f"Demand CSV shape: {demand_df.shape}")

    # Ensure there are at least 24 rows for inference
    if len(wind_df) < 24 or len(demand_df) < 24:
        flash("Each uploaded CSV must contain at least 24 hourly records (past 24 hours).", "danger")
        return redirect(url_for("index"))

    # Load Texas energy portfolio data
    texas_data = load_texas_energy_data()
    if texas_data is None:
        flash("Error loading Texas energy portfolio data. Using fallback predictions.", "warning")
    
    # Get the latest timestamp from the data to align with forecast
    # For testing with 2024 data, use 2024-01-01 as the forecast start
    forecast_start_time = datetime(2024, 1, 1, 0, 0, 0)
    if texas_data is not None and not texas_data.empty:
        # Use the first timestamp from the dataset for testing
        forecast_start_time = texas_data['Timestamp'].min()
    
    # Generate timestamps for the forecast period
    forecast_hours = 24
    forecast_timestamps = [forecast_start_time + timedelta(hours=i) for i in range(forecast_hours)]
    
    # Initialize forecast results
    results = []
    
    # 1. WIND PREDICTION
    try:
        # Load wind model and scaler
        wind_model_path = os.path.join(os.path.dirname(BASE_DIR), "wind forecast", "wind_forecast_model.pkl")
        wind_scaler_path = os.path.join(os.path.dirname(BASE_DIR), "wind forecast", "scaler.pkl")
        
        if os.path.exists(wind_model_path) and os.path.exists(wind_scaler_path):
            with open(wind_model_path, 'rb') as f:
                wind_model = pickle.load(f)
            with open(wind_scaler_path, 'rb') as f:
                wind_scaler = pickle.load(f)
            
            # Prepare input features for wind prediction
            # The wind model expects columns: wind_speed, wind_direction, pressure, air_temperature
            required_wind_cols = ['wind_speed', 'wind_direction', 'pressure', 'air_temperature']
            
            # Check if all required columns exist
            if all(col in wind_df.columns for col in required_wind_cols):
                # Get all 24 rows for prediction (use last 24 hours of data)
                wind_input = wind_df[required_wind_cols].iloc[-24:].values
                
                # If we have less than 24 rows, repeat the last row
                if len(wind_input) < 24:
                    last_row = wind_input[-1]
                    wind_input = np.vstack([wind_input] + [last_row] * (24 - len(wind_input)))
                
                # Scale features
                wind_input_scaled = wind_scaler.transform(wind_input)
                
                # Predict wind power for each of the 24 hours
                wind_predictions = wind_model.predict(wind_input_scaled)
                wind_forecast = [float(x) for x in wind_predictions.flatten()[:forecast_hours]]
                
                print(f"Wind predictions: min={min(wind_forecast):.2f}, max={max(wind_forecast):.2f}, avg={sum(wind_forecast)/len(wind_forecast):.2f}")
            else:
                missing = [col for col in required_wind_cols if col not in wind_df.columns]
                print(f"Missing wind columns: {missing}")
                raise ValueError(f"Missing required wind columns: {missing}")
        else:
            # Fallback: Use the last available wind value
            last_wind = wind_df.iloc[-1]['wind_speed'] if 'wind_speed' in wind_df.columns else 100
            wind_forecast = [float(last_wind)] * forecast_hours
            
    except Exception as e:
        print(f"Wind prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: Use a default value
        wind_forecast = [100.0] * forecast_hours
    
    # 2. DEMAND PREDICTION
    try:
        # Prepare input for demand prediction
        # Assuming the demand CSV has columns: temperature, humidity, timestamp
        if 'temperature' in demand_df.columns and 'humidity' in demand_df.columns:
            # Use the last row for prediction (most recent data)
            last_row = demand_df.iloc[-1]
            temp = float(last_row['temperature'])
            humidity = float(last_row['humidity'])
            
            # Generate demand predictions for the next 24 hours
            demand_forecast = []
            for i in range(forecast_hours):
                # Adjust temperature slightly for each hour (optional)
                # This simulates temperature changes throughout the day
                hour_of_day = (forecast_start_time.hour + i) % 24
                temp_adjusted = temp + 5 * np.sin(hour_of_day * np.pi / 12)  # Vary temp by ±5°C
                
                # Call the demand prediction function
                try:
                    # Provide absolute paths to model files in parent directory
                    parent_dir = os.path.dirname(BASE_DIR)
                    pred = predict_demand(
                        Temperature=temp_adjusted,
                        Humidity=humidity,
                        Timestamp=forecast_timestamps[i],
                        model_path=os.path.join(parent_dir, 'demand_forecast_xgboost.pkl'),
                        feature_scaler_path=os.path.join(parent_dir, 'demand_feature_scaler_xgb.pkl'),
                        target_scaler_path=os.path.join(parent_dir, 'demand_target_scaler_xgb.pkl'),
                        year_params_path=os.path.join(parent_dir, 'demand_year_params.json')
                    )
                    demand_forecast.append(float(pred))
                    if i == 0:  # Log first prediction
                        print(f"First demand prediction: {pred} for temp={temp_adjusted}, humidity={humidity}")
                except Exception as pred_error:
                    # Fallback if prediction fails
                    print(f"Demand prediction failed for hour {i}: {pred_error}")
                    demand_forecast.append(5000.0)  # Default value
        else:
            # Fallback: Use the last available demand value or a default
            last_demand = demand_df.iloc[-1].iloc[-1] if not demand_df.empty else 5000.0
            demand_forecast = [float(last_demand)] * forecast_hours
            
    except Exception as e:
        print(f"Demand prediction error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: Use a default value
        demand_forecast = [5000.0] * forecast_hours
    
    # 3. CALCULATE FOSSIL FUEL NEEDED AND PREPARE RESULTS
    results = []
    
    for i in range(forecast_hours):
        # Get forecasted wind and demand
        wind = wind_forecast[i]
        demand = demand_forecast[i]
        
        # Get other renewables from Texas data if available
        solar = 0.0
        hydro = 0.0
        nuclear = 0.0
        
        if texas_data is not None and not texas_data.empty:
            # Get the same hour from historical data (e.g., same hour of the day)
            hour_of_day = forecast_timestamps[i].hour
            historical_data = texas_data[texas_data['Timestamp'].dt.hour == hour_of_day]
            
            if not historical_data.empty:
                # Use median values for the hour
                solar = historical_data['Solar_MW'].median() if 'Solar_MW' in historical_data.columns else 0.0
                hydro = historical_data['Hydro_MW'].median() if 'Hydro_MW' in historical_data.columns else 0.0
                nuclear = historical_data['Nuclear_MW'].median() if 'Nuclear_MW' in historical_data.columns else 0.0
        
        # Calculate fossil fuel needed
        # Nuclear is NOT subtracted because it's non-renewable
        # Only renewable sources (wind, solar, hydro) are subtracted from demand
        fossil = max(0, demand - wind - solar - hydro)
        
        # Append result for this hour
        results.append({
            "hour": i + 1,
            "timestamp": forecast_timestamps[i].strftime("%Y-%m-%d %H:%M"),
            "wind_mw": round(wind, 2),
            "demand_mw": round(demand, 2),
            "solar_mw": round(solar, 2),
            "hydro_mw": round(hydro, 2),
            "nuclear_mw": round(nuclear, 2),
            "fossil_needed_mw": round(fossil, 2)
        })
    
    # Save results temporarily as JSON for analysis page
    out_json_path = os.path.join(UPLOAD_FOLDER, "latest_results.json")
    with open(out_json_path, "w") as fh:
        json.dump({
            "results": results,
            "forecast_start": forecast_start_time.isoformat(),
            "forecast_end": forecast_timestamps[-1].isoformat()
        }, fh, default=str)

    # If manual input, show comparison page with predicted vs actual for single hour
    if input_type == 'manual':
        # Get the specific hour's prediction from the 24-hour forecast
        selected_hour = request.selected_hour if hasattr(request, 'selected_hour') else 12
        wind_predicted = [wind_forecast[selected_hour]]  # Get the specific hour's prediction
        demand_predicted = [demand_forecast[selected_hour]]  # Get the specific hour's prediction
        
        wind_actual = [0]  # Default value
        demand_actual = [0]  # Default value
        
        selected_datetime = request.selected_datetime if hasattr(request, 'selected_datetime') else datetime.now()
        
        # Load wind actual data for the specific hour
        try:
            wind_data_path = os.path.join(DATA_FOLDER, "Wind data_texas.csv")
            if os.path.exists(wind_data_path):
                wind_actual_df = pd.read_csv(wind_data_path)
                # Parse timestamp - handle format like "Jan 1, 12:00 am"
                wind_actual_df['Timestamp'] = pd.to_datetime(wind_actual_df['Timestamp'], format='%b %d, %I:%M %p', errors='coerce')
                
                # Find the closest timestamp to our selected datetime
                time_diff = (wind_actual_df['Timestamp'] - selected_datetime).abs()
                closest_idx = time_diff.idxmin()
                wind_actual = [wind_actual_df.loc[closest_idx, 'windfarm power']]
        except Exception as e:
            print(f"Error loading wind actual data: {e}")
            wind_actual = wind_predicted
        
        # Load demand actual data for the specific hour
        try:
            demand_data_path = os.path.join(DATA_FOLDER, "demand_data_texas.csv")
            if os.path.exists(demand_data_path):
                demand_actual_df = pd.read_csv(demand_data_path)
                # Parse timestamp - format like "01-Jan-20"
                demand_actual_df['Timestamp'] = pd.to_datetime(demand_actual_df['Timestamp'], format='%d-%b-%y', errors='coerce')
                
                # Find the closest timestamp to our selected datetime
                time_diff = (demand_actual_df['Timestamp'] - selected_datetime).abs()
                closest_idx = time_diff.idxmin()
                demand_actual = [demand_actual_df.loc[closest_idx, 'Demand']]
        except Exception as e:
            print(f"Error loading demand actual data: {e}")
            demand_actual = demand_predicted
        
        return render_template(
            "manual_results.html",
            selected_datetime=selected_datetime,
            now=datetime.now(),
            wind_speed=request.form.get('manual_wind_speed', 5.5),
            wind_dir=request.form.get('manual_wind_dir', 180),
            pressure=request.form.get('manual_pressure', 1013),
            air_temp=request.form.get('manual_air_temp', 20),
            demand_temp=request.form.get('manual_temp', 20),
            humidity=request.form.get('manual_humidity', 65),
            wind_predicted=wind_predicted,
            wind_actual=wind_actual,
            demand_predicted=demand_predicted,
            demand_actual=demand_actual
        )
    
    return render_template("results.html", results=results)

@app.route("/analysis")
def analysis():
    out_json_path = os.path.join(UPLOAD_FOLDER, "latest_results.json")
    if not os.path.exists(out_json_path):
        flash("No results available. Please upload CSVs and run prediction first.", "warning")
        return redirect(url_for("index"))
    
    with open(out_json_path) as fh:
        data = json.load(fh)
    
    results = data["results"]
    forecast_start = datetime.fromisoformat(data.get("forecast_start")) if "forecast_start" in data else None
    
    # Get actual non-renewable energy (Fossil + Nuclear) from Texas energy portfolio for the SAME timestamps
    texas_data = load_texas_energy_data()
    
    if texas_data is not None and forecast_start:
        # Get the actual non-renewable energy usage for the same 24-hour period we're forecasting
        forecast_end = forecast_start + timedelta(hours=23)
        actual_data = texas_data[
            (texas_data['Timestamp'] >= forecast_start) & 
            (texas_data['Timestamp'] <= forecast_end)
        ]
        
        if not actual_data.empty and len(actual_data) >= 24:
            # Use actual non-renewable energy (Fossil + Nuclear) for comparison
            actual_data = actual_data.iloc[:24]
            # Historical non-renewable = Fossil + Nuclear (both are non-renewable)
            f1 = (actual_data['Fossil_MW'] + actual_data['Nuclear_MW']).tolist()
            print(f"Using actual non-renewable (Fossil+Nuclear) data: {len(f1)} hours, avg={sum(f1)/len(f1):.2f} MW")
        else:
            # If not enough data, use a baseline from the dataset
            # Use the average non-renewable energy for each hour of the day from the entire dataset
            f1 = []
            for i in range(24):
                hour_data = texas_data[texas_data['Timestamp'].dt.hour == i]
                if not hour_data.empty:
                    # Average of Fossil + Nuclear for this hour
                    f1.append((hour_data['Fossil_MW'] + hour_data['Nuclear_MW']).mean())
                else:
                    f1.append(4500.0)  # Default baseline
            print(f"Using average hourly non-renewable baseline from dataset")
    else:
        # Fallback: Use a baseline from the dataset
        if texas_data is not None:
            f1 = [(texas_data['Fossil_MW'] + texas_data['Nuclear_MW']).mean()] * 24
        else:
            f1 = [4500.0] * 24
    
    # Get the forecasted non-renewable energy (Fossil + Nuclear)
    # Since we're comparing with historical non-renewable (Fossil + Nuclear), 
    # we need to add nuclear to the forecasted fossil fuel
    f2 = [r["fossil_needed_mw"] + r.get("nuclear_mw", 0) for r in results]
    
    # Calculate energy savings (difference between historical and forecasted non-renewable energy usage)
    energy_saved = [max(0, round(a - b, 2)) for a, b in zip(f1, f2)]
    total_energy_saved_mwh = round(sum(energy_saved), 2)
    
    # Calculate CO2 savings (assuming 0.95 kg CO2 per kWh from fossil fuels)
    co2_saved_kg = round(total_energy_saved_mwh * 1000 * 0.95, 2)
    
    # Calculate RECs (1 REC per MWh of renewable energy used)
    # For simplicity, we'll assume the difference in fossil fuel usage is due to increased renewables
    recs = round(total_energy_saved_mwh, 2)
    
    # Prepare data for the chart
    timestamps = [r["timestamp"] for r in results] if "timestamp" in results[0] else list(range(1, 25))
    
    # Get other energy sources for the stacked area chart
    wind_forecast = [r["wind_mw"] for r in results]
    solar_forecast = [r.get("solar_mw", 0) for r in results]
    hydro_forecast = [r.get("hydro_mw", 0) for r in results]
    nuclear_forecast = [r.get("nuclear_mw", 0) for r in results]
    
    return render_template(
        "analysis.html",
        timestamps=timestamps,
        f1=f1,
        f2=f2,
        wind_forecast=wind_forecast,
        solar_forecast=solar_forecast,
        hydro_forecast=hydro_forecast,
        nuclear_forecast=nuclear_forecast,
        energy_saved=energy_saved,
        total_energy_saved_mwh=total_energy_saved_mwh,
        co2_saved_kg=co2_saved_kg,
        recs=recs
    )

@app.route("/download_results")
def download_results():
    path = os.path.join(UPLOAD_FOLDER, "latest_results.json")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="latest_results.json")
    else:
        flash("No results to download.", "warning")
        return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
