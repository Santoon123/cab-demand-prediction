from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, timezone
import holidays
import os
import requests
import json
import traceback
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_ml/")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "19588a43a473ec2b97e25d5758972d61")
WEATHER_API_ENDPOINT = "https://api.openweathermap.org/data/2.5/forecast"
NYC_LAT = 40.7128
NYC_LON = -74.0060
OWM_FORECAST_MAP = {
    "main.temp": "temperature",
    "main.humidity": "humidity",
    "wind.speed": "wind_speed",
    "rain.3h": "precipitation",
    "snow.3h": "snowfall",
}

print("--- API Server Starting (Using 2-Year Model) ---")
print(f"Loading assets from: {os.path.abspath(MODEL_DIR)}")
try:
    model_path = os.path.join(MODEL_DIR, "lgbm_demand_model_2yr.joblib")
    model = joblib.load(model_path)
    print(f"Model loaded.")

    feature_scaler_path = os.path.join(MODEL_DIR, "feature_scaler_ml_2yr.joblib")
    feature_scaler = joblib.load(feature_scaler_path)
    print(f"Feature scaler loaded.")

    zone_order_path = os.path.join(MODEL_DIR, "zone_order_ml.joblib")
    ZONE_ORDER = joblib.load(zone_order_path)
    NUM_ZONES = len(ZONE_ORDER)
    print(f"Zone order loaded ({NUM_ZONES} zones).")

    feature_names_path = os.path.join(MODEL_DIR, "feature_names_ml_2yr.joblib")
    EXPECTED_FEATURES = joblib.load(feature_names_path)
    print(f"Model expects {len(EXPECTED_FEATURES)} features.")
    time_feature_names_list = [
        "year",
        "hour",
        "dayofweek",
        "dayofmonth",
        "month",
        "quarter",
        "dayofyear",
        "weekofyear",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]
    EXPECTED_TIME_FEATURES_API = [
        f for f in EXPECTED_FEATURES if f in time_feature_names_list
    ]
    EXPECTED_WEATHER_FEATURES_API = [
        f for f in EXPECTED_FEATURES if f not in EXPECTED_TIME_FEATURES_API
    ]
    print(f"Identified {len(EXPECTED_TIME_FEATURES_API)} time features (incl. year).")
    print(f"Identified {len(EXPECTED_WEATHER_FEATURES_API)} weather features.")
except FileNotFoundError as fnf_error:
    print(
        f"FATAL ERROR loading asset: {fnf_error}. Check paths & ensure 2yr training ran."
    )
    exit()
except Exception as e:
    print(f"FATAL: Could not load assets: {e}")
    traceback.print_exc()
    exit()


def get_weather_forecast(target_dt_utc):
    print(f"Fetching weather forecast closest to: {target_dt_utc} (UTC)")
    target_unix_timestamp = int(target_dt_utc.timestamp())
    params = {
        "lat": NYC_LAT,
        "lon": NYC_LON,
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }
    try:
        response = requests.get(WEATHER_API_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        forecast_data = response.json()
        if "list" not in forecast_data or not forecast_data["list"]:
            return {key: 0.0 for key in EXPECTED_WEATHER_FEATURES_API}
        closest_forecast = min(
            forecast_data["list"], key=lambda x: abs(x["dt"] - target_unix_timestamp)
        )
        forecast_time = datetime.fromtimestamp(closest_forecast["dt"], tz=timezone.utc)
        print(f"Closest forecast block: {forecast_time}")
        weather_values = {}
        for owm_key, feature_name in OWM_FORECAST_MAP.items():
            keys = owm_key.split(".")
            value = closest_forecast
            try:
                for key in keys:
                    value = value[key]
                weather_values[feature_name] = float(value)
            except (KeyError, TypeError):
                weather_values[feature_name] = 0.0
        if "precipitation" in weather_values:
            weather_values["precipitation"] /= 3.0
        model_expects_snowfall = "snowfall" in EXPECTED_WEATHER_FEATURES_API
        model_expects_snow_depth = "snow_depth" in EXPECTED_WEATHER_FEATURES_API
        if model_expects_snowfall:
            weather_values["snowfall"] = weather_values.get("snowfall", 0.0) / 3.0
        elif model_expects_snow_depth:
            weather_values["snow_depth"] = 0.0
        final_weather_dict = {
            key: weather_values.get(key, 0.0) for key in EXPECTED_WEATHER_FEATURES_API
        }
        print(f"Weather fetched: {final_weather_dict}")
        return final_weather_dict
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Weather API fail: {e}")
        return {key: 0.0 for key in EXPECTED_WEATHER_FEATURES_API}
    except Exception as e:
        print(f"ERROR: Weather processing error: {e}")
        traceback.print_exc()
        return {key: 0.0 for key in EXPECTED_WEATHER_FEATURES_API}


def generate_time_features(target_dt_utc, expected_time_cols):
    """Generates ONLY time features for the target_dt (assumed UTC), including Year."""
    df_time_features = pd.DataFrame(index=[target_dt_utc])
    df_time_features["year"] = df_time_features.index.year
    df_time_features["hour"] = df_time_features.index.hour
    df_time_features["dayofweek"] = df_time_features.index.dayofweek
    df_time_features["dayofmonth"] = df_time_features.index.day
    df_time_features["month"] = df_time_features.index.month
    df_time_features["quarter"] = df_time_features.index.quarter
    df_time_features["dayofyear"] = df_time_features.index.dayofyear
    df_time_features["weekofyear"] = df_time_features.index.isocalendar().week.astype(
        int
    )
    df_time_features["is_weekend"] = (
        df_time_features["dayofweek"].isin([5, 6]).astype(int)
    )
    target_date = target_dt_utc.date()
    us_holidays_api = holidays.US(years=[target_date.year])
    df_time_features["is_holiday"] = 1 if target_date in us_holidays_api else 0
    seconds_in_day = 24 * 60 * 60
    seconds_in_year = 365.2425 * seconds_in_day
    timestamp_sec = target_dt_utc.timestamp()
    df_time_features["hour_sin"] = np.sin(timestamp_sec * (2 * np.pi / seconds_in_day))
    df_time_features["hour_cos"] = np.cos(timestamp_sec * (2 * np.pi / seconds_in_day))
    df_time_features["month_sin"] = np.sin(
        timestamp_sec * (2 * np.pi / seconds_in_year)
    )
    df_time_features["month_cos"] = np.cos(
        timestamp_sec * (2 * np.pi / seconds_in_year)
    )
    time_cols_present = [
        col for col in expected_time_cols if col in df_time_features.columns
    ]
    return df_time_features[time_cols_present]


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["GET"])
def predict():
    global model, feature_scaler, ZONE_ORDER, EXPECTED_FEATURES, EXPECTED_TIME_FEATURES_API, EXPECTED_WEATHER_FEATURES_API
    date_str = request.args.get("date")
    time_str = request.args.get("time")
    if not date_str or not time_str:
        return jsonify({"error": "Missing 'date' or 'time'."}), 400
    try:
        local_tz_name = "America/New_York"
        target_dt_naive = pd.Timestamp(f"{date_str} {time_str}")
        target_dt_local = target_dt_naive.tz_localize(
            local_tz_name, ambiguous=False, nonexistent="shift_forward"
        )
        target_dt_utc = target_dt_local.tz_convert("UTC")
        df_time = generate_time_features(target_dt_utc, EXPECTED_TIME_FEATURES_API)
        if df_time is None or df_time.empty:
            raise ValueError("Time feature generation failed.")
        weather_data_dict = get_weather_forecast(target_dt_utc)
        df_weather = pd.DataFrame([weather_data_dict], index=[target_dt_utc])
        df_time.index.name = "ts_t"
        df_weather.index.name = "ts_w"
        df_combined_features = df_time.join(df_weather, how="left")
        df_combined_features.index.name = "timestamp"
        df_combined_features = df_combined_features.reindex(columns=EXPECTED_FEATURES)
        if df_combined_features.isnull().values.any():
            df_combined_features.fillna(0.0, inplace=True)
        scaled_input_features = feature_scaler.transform(df_combined_features)
        if hasattr(model, "estimators_"):
            all_predictions = [
                est.predict(scaled_input_features)[0] for est in model.estimators_
            ]
            prediction_flat = np.array(all_predictions)
        else:
            raise RuntimeError("Loaded model missing 'estimators_'.")
        predictions = {
            str(zone_id): max(0, int(round(prediction_flat[i])))
            for i, zone_id in enumerate(ZONE_ORDER)
        }

    except ValueError as ve:
        print(f"Value Error: {ve}")
        return jsonify({"error": f"Invalid input/feature error: {ve}"}), 400
    except Exception as e:
        print(f"---!!! UNEXPECTED ERROR !!!---")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed (server error)."}), 500
    return jsonify(predictions)


if __name__ == "__main__":
    print("\n--- Starting Flask Server (2-Year Model) ---")
    app.run(debug=False, host="0.0.0.0", port=5000)
