import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import MinMaxScaler
import holidays
import os
import traceback
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from meteostat import Point, Hourly
from datetime import timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(CODE_DIR, "Dataset")
BACKEND_DIR = os.path.join(CODE_DIR, "Backend")
MODEL_DIR = os.path.join(BACKEND_DIR, "models_ml/")
EVAL_DATA_PATH = os.path.join(SCRIPT_DIR, "jan-2025.parquet")
ZONE_LOOKUP_PATH = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
WEATHER_COLS_TO_USE = [
    "temperature",
    "precipitation",
    "snow_depth",
    "humidity",
    "wind_speed",
]
NYC_LAT = 40.7128
NYC_LON = -74.0060
NYC_ALTITUDE = 10
print("--- Model Evaluation Script (Evaluating 2-Year Model) ---")
print(f"Evaluating using Taxi Data: {EVAL_DATA_PATH}")
print(f"Loading 2-Year assets from: {MODEL_DIR}")
print("\nLoading 2-Year model assets...")
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
except FileNotFoundError as fnf_error:
    print(
        f"FATAL ERROR loading asset: {fnf_error}. Check paths, ensure 2-year training ran."
    )
    exit()
except Exception as e:
    print(f"FATAL: Could not load assets: {e}")
    traceback.print_exc()
    exit()
print(f"\nLoading evaluation taxi data (Jan 2025)...")
try:
    zone_lookup_df = pd.read_csv(ZONE_LOOKUP_PATH)
    zone_lookup_df["LocationID"] = (
        zone_lookup_df["LocationID"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    VALID_ZONE_IDS = sorted(list(zone_lookup_df["LocationID"].unique()))
    print(f"Using {len(VALID_ZONE_IDS)} zones.")
    parquet_cols_to_read = ["pickup_datetime", "PULocationID"]
    df_eval_raw = pd.read_parquet(EVAL_DATA_PATH, columns=parquet_cols_to_read)
    df_eval_raw.dropna(subset=parquet_cols_to_read, inplace=True)
    df_eval_raw["PULocationID"] = pd.to_numeric(
        df_eval_raw["PULocationID"], errors="coerce"
    )
    df_eval_raw.dropna(subset=["PULocationID"], inplace=True)
    df_eval_raw["PULocationID"] = df_eval_raw["PULocationID"].astype(int)
    df_eval_raw = df_eval_raw[df_eval_raw["PULocationID"].isin(VALID_ZONE_IDS)]
    df_eval_raw["pickup_datetime"] = pd.to_datetime(df_eval_raw["pickup_datetime"])
    start_eval_dt = pd.Timestamp("2025-01-01 00:00:00")
    end_eval_dt = pd.Timestamp("2025-01-31 23:59:59")
    df_eval_raw = df_eval_raw[
        (df_eval_raw["pickup_datetime"] >= start_eval_dt)
        & (df_eval_raw["pickup_datetime"] <= end_eval_dt)
    ]
    if df_eval_raw.empty:
        print("ERROR: No valid trip data found for Jan 2025.")
        exit()
    print(f"Loaded {len(df_eval_raw)} valid trips for Jan 2025.")
    df_eval_raw["pickup_hour"] = df_eval_raw["pickup_datetime"].dt.floor("h")
    df_eval_demand = (
        df_eval_raw.groupby(["pickup_hour", "PULocationID"])
        .size()
        .reset_index(name="demand")
    )
    del df_eval_raw
    df_eval_pivot = df_eval_demand.pivot_table(
        index="pickup_hour", columns="PULocationID", values="demand", fill_value=0
    )
    print("Applying Timezone (Converting Eval Index to UTC)...")
    try:
        if df_eval_pivot.index.tz is None:
            df_eval_pivot.index = df_eval_pivot.index.tz_localize("UTC")
        elif df_eval_pivot.index.tz != "UTC":
            df_eval_pivot.index = df_eval_pivot.index.tz_convert("UTC")
        else:
            print("Eval index already UTC.")
    except Exception as tz_err:
        print(f"ERROR: TZ handling failed for eval data: {tz_err}.")
        traceback.print_exc()
        exit()
    start_eval_utc = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    end_eval_utc = pd.Timestamp("2025-01-31 23:00:00", tz="UTC")
    all_hours_eval_utc = pd.date_range(start=start_eval_utc, end=end_eval_utc, freq="h")
    df_eval_pivot = df_eval_pivot.reindex(all_hours_eval_utc, fill_value=0)
    df_eval_pivot = df_eval_pivot.reindex(columns=ZONE_ORDER, fill_value=0)
    df_eval_pivot.fillna(0, inplace=True)
    df_eval_pivot = df_eval_pivot.astype(int)
    y_true = df_eval_pivot
    print(f"Processed ground truth data shape (y_true UTC): {y_true.shape}")
except FileNotFoundError:
    print(f"ERROR: Eval taxi data file not found: {EVAL_DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error processing eval taxi data: {e}")
    traceback.print_exc()
    exit()
print("\nGenerating time features for evaluation period (from UTC index)...")
try:
    eval_index = y_true.index
    df_eval_time_features = pd.DataFrame(index=eval_index)
    df_eval_time_features["year"] = (
        df_eval_time_features.index.year
    )  # Year will be 2025
    df_eval_time_features["hour"] = df_eval_time_features.index.hour
    df_eval_time_features["dayofweek"] = df_eval_time_features.index.dayofweek
    df_eval_time_features["dayofmonth"] = df_eval_time_features.index.day
    df_eval_time_features["month"] = df_eval_time_features.index.month
    df_eval_time_features["quarter"] = df_eval_time_features.index.quarter
    df_eval_time_features["dayofyear"] = df_eval_time_features.index.dayofyear
    df_eval_time_features["weekofyear"] = (
        df_eval_time_features.index.isocalendar().week.astype(int)
    )
    df_eval_time_features["is_weekend"] = (
        df_eval_time_features["dayofweek"].isin([5, 6]).astype(int)
    )
    us_holidays_eval = holidays.US(years=[2025])
    eval_dates = df_eval_time_features.index.date
    df_eval_time_features["is_holiday"] = np.isin(
        eval_dates, [d for d in us_holidays_eval]
    ).astype(int)
    seconds_in_day = 24 * 60 * 60
    seconds_in_year = 365.2425 * seconds_in_day
    timestamps_sec = df_eval_time_features.index.astype(np.int64) // 10**9
    df_eval_time_features["hour_sin"] = np.sin(
        timestamps_sec * (2 * np.pi / seconds_in_day)
    )
    df_eval_time_features["hour_cos"] = np.cos(
        timestamps_sec * (2 * np.pi / seconds_in_day)
    )
    df_eval_time_features["month_sin"] = np.sin(
        timestamps_sec * (2 * np.pi / seconds_in_year)
    )
    df_eval_time_features["month_cos"] = np.cos(
        timestamps_sec * (2 * np.pi / seconds_in_year)
    )
    print(f"Generated time features shape: {df_eval_time_features.shape}")
except Exception as e:
    print(f"Error generating time features: {e}")
    traceback.print_exc()
    exit()
print(
    "\nFetching & processing weather data for evaluation period (Jan 2025) using Meteostat..."
)
try:
    location = Point(NYC_LAT, NYC_LON, NYC_ALTITUDE)
    fetch_start_utc = y_true.index.min()
    fetch_end_utc = y_true.index.max()
    print(f"Fetching weather from {fetch_start_utc} to {fetch_end_utc}")
    fetch_start_dt_naive = fetch_start_utc.to_pydatetime().replace(tzinfo=None)
    fetch_end_dt_naive = fetch_end_utc.to_pydatetime().replace(tzinfo=None)
    weather_fetch_data = Hourly(location, fetch_start_dt_naive, fetch_end_dt_naive)
    df_eval_weather_hist = weather_fetch_data.fetch()
    if df_eval_weather_hist.empty:
        print("ERROR: Meteostat fetch returned no data.")
        exit()
    print(f"Fetched {len(df_eval_weather_hist)} hourly weather records.")
    df_eval_weather_hist.index = pd.to_datetime(df_eval_weather_hist.index, utc=True)
    if df_eval_weather_hist.index.tz is None:
        df_eval_weather_hist.index = df_eval_weather_hist.index.tz_localize("UTC")
    elif df_eval_weather_hist.index.tz != "UTC":
        df_eval_weather_hist.index = df_eval_weather_hist.index.tz_convert("UTC")
    df_eval_weather_hist = df_eval_weather_hist.rename(
        columns={
            "temp": "temperature",
            "prcp": "precipitation",
            "snow": "snow_depth",
            "rhum": "humidity",
            "wspd": "wind_speed",
        }
    )
    missing_weather_cols = [
        col for col in WEATHER_COLS_TO_USE if col not in df_eval_weather_hist.columns
    ]
    if missing_weather_cols:
        print(f"Warning: Weather cols missing: {missing_weather_cols}. Filling with 0.")
        for col in missing_weather_cols:
            df_eval_weather_hist[col] = 0.0
    df_eval_weather_hist = df_eval_weather_hist[WEATHER_COLS_TO_USE]
    if "snow_depth" in df_eval_weather_hist.columns:
        df_eval_weather_hist["snow_depth"].fillna(0, inplace=True)
    df_eval_weather_hist = (
        df_eval_weather_hist.interpolate(method="time").ffill().bfill()
    )
    if df_eval_weather_hist.isnull().values.any():
        print("ERROR: NaNs still in fetched eval weather data!")
        exit()
    print(f"Processed eval weather data shape: {df_eval_weather_hist.shape}")
except Exception as e:
    print(f"Error fetching/processing eval weather data: {e}")
    traceback.print_exc()
    exit()
print("\nCombining Time and Weather features for evaluation...")
try:
    df_eval_weather_hist = (
        df_eval_weather_hist.reindex(df_eval_time_features.index).ffill().bfill()
    )
    df_eval_time_features.index.name = "ts_t"
    df_eval_weather_hist.index.name = "ts_w"
    X_eval_combined = df_eval_time_features.join(df_eval_weather_hist, how="left")
    X_eval_combined.index.name = "timestamp"
    X_eval = X_eval_combined.reindex(columns=EXPECTED_FEATURES)
    if X_eval.isnull().values.any():
        print("Warning: NaNs found after combining features. Filling with 0.")
        X_eval.fillna(0.0, inplace=True)
    if X_eval.isnull().values.any():
        print("ERROR: NaNs remain in final evaluation features!")
        exit()
    print(f"Generated evaluation features shape (X_eval combined): {X_eval.shape}")
except Exception as e:
    print(f"Error combining features for evaluation: {e}")
    traceback.print_exc()
    exit()
print("\nScaling evaluation features...")
try:
    X_eval_scaled = feature_scaler.transform(X_eval)
    print(f"Scaled evaluation features shape: {X_eval_scaled.shape}")
except Exception as e:
    print(f"Error scaling evaluation features: {e}")
    traceback.print_exc()
    exit()
print("\nMaking predictions for evaluation period...")
try:
    y_pred_raw = model.predict(X_eval_scaled)
    y_pred = np.maximum(0, np.round(y_pred_raw)).astype(int)
    df_pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
    print(f"Generated predictions shape (df_pred): {df_pred.shape}")
except Exception as e:
    print(f"Error during prediction: {e}")
    traceback.print_exc()
    exit()
print("\nCalculating evaluation metrics...")
if not y_true.index.equals(df_pred.index) or not y_true.columns.equals(df_pred.columns):
    print("ERROR: Mismatch index/columns. Aligning...")
    df_pred = df_pred.reindex_like(y_true).fillna(0)
    if not y_true.index.equals(df_pred.index) or not y_true.columns.equals(
        df_pred.columns
    ):
        print("ERROR: Reindex failed.")
        exit()
try:
    mae_overall = mean_absolute_error(y_true, df_pred)
    rmse_overall = np.sqrt(mean_squared_error(y_true, df_pred))
    mask = y_true > 0
    mape_overall = np.mean(np.abs((y_true[mask] - df_pred[mask]) / y_true[mask])) * 100
    print("\n--- Overall Metrics (Jan 2025 - Using 2-Year Model) ---")
    print(f"Mean Absolute Error (MAE):  {mae_overall:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_overall:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape_overall:.2f}% (Actual > 0)")
    print("\n--- Per-Zone Metrics (Examples - Using 2-Year Model) ---")
    zone_total_demand = y_true.sum().sort_values(ascending=False)
    top_n = 5
    print(f"Metrics for Top {top_n} Zones by Actual Demand:")
    for zone_id in zone_total_demand.head(top_n).index:
        mae_zone = mean_absolute_error(y_true[zone_id], df_pred[zone_id])
        rmse_zone = np.sqrt(mean_squared_error(y_true[zone_id], df_pred[zone_id]))
        mask_zone = y_true[zone_id] > 0
        mape_zone = (
            np.mean(
                np.abs(
                    (y_true[zone_id][mask_zone] - df_pred[zone_id][mask_zone])
                    / y_true[zone_id][mask_zone]
                )
            )
            * 100
            if mask_zone.any()
            else np.nan
        )
        print(
            f"  Zone {zone_id}: MAE={mae_zone:.2f}, RMSE={rmse_zone:.2f}, MAPE={mape_zone:.1f}%"
            if not np.isnan(mape_zone)
            else f"  Zone {zone_id}: MAE={mae_zone:.2f}, RMSE={rmse_zone:.2f}, MAPE=N/A"
        )
except Exception as e:
    print(f"\nError calculating metrics: {e}")
    traceback.print_exc()

print("\n--- Evaluation Complete ---")
