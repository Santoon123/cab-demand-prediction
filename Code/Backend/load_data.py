import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import holidays
import os
import joblib
import traceback
from meteostat import Point, Hourly
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(
    CODE_DIR, "D:\\Projects\\Cab Demand Prediction - Copy\\Code\\Dataset"
)
ZONE_LOOKUP_PATH = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
HISTORICAL_WEATHER_PATH_2024 = os.path.join(
    CODE_DIR, "nyc_weather_2024_hourly_meteostat.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_ml/")
MODEL_DIR = os.path.join(BASE_DIR, "models_ml/")
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(
    "--- Data Loading and Processing Script (2023 & 2024 - Fetching 2023 Weather) ---"
)
print(f"Using Base Directory: {BASE_DIR}")
print(f"Using Data Directory: {DATA_DIR}")
print(f"Outputting Processed Data to: {OUTPUT_DIR}")
print(f"Outputting Model Assets to: {MODEL_DIR}")
print("\nLoading Taxi Zone Lookup CSV...")
try:
    zone_lookup_df = pd.read_csv(ZONE_LOOKUP_PATH)
    zone_lookup_df["LocationID"] = (
        zone_lookup_df["LocationID"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    zone_lookup_df = zone_lookup_df[["LocationID", "Borough", "Zone"]].drop_duplicates(
        subset=["LocationID"]
    )
    VALID_ZONE_IDS = sorted(list(zone_lookup_df["LocationID"].unique()))
    NUM_ZONES = len(VALID_ZONE_IDS)
    print(f"Loaded {NUM_ZONES} unique zones.")
except FileNotFoundError:
    print(f"ERROR: Zone lookup file not found: {ZONE_LOOKUP_PATH}.")
    exit()
except Exception as e:
    print(f"Error loading zone lookup: {e}")
    traceback.print_exc()
    exit()
print("\nLoading and Processing Taxi Trip Data (2023 & 2024)...")
all_trip_data = []
parquet_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".parquet") and "fhvhv" in f and ("2023" in f or "2024" in f)
]
if not parquet_files:
    print(f"ERROR: No 2023 or 2024 Parquet files found.")
    exit()
    print(f"Found {len(parquet_files)} files. Loading...")
parquet_cols_to_read = ["pickup_datetime", "PULocationID"]
for i, file_path in enumerate(parquet_files):
    year = "2023" if "2023" in file_path else "2024"
    print(f"Processing file ({year}): {os.path.basename(file_path)}...")
    try:
        df_month = pd.read_parquet(file_path, columns=parquet_cols_to_read)
        df_month.dropna(subset=parquet_cols_to_read, inplace=True)
        df_month["PULocationID"] = pd.to_numeric(
            df_month["PULocationID"], errors="coerce"
        )
        df_month.dropna(subset=["PULocationID"], inplace=True)
        df_month["PULocationID"] = df_month["PULocationID"].astype(int)
        df_month = df_month[df_month["PULocationID"].isin(VALID_ZONE_IDS)]
        df_month["pickup_datetime"] = pd.to_datetime(df_month["pickup_datetime"])
        df_month = df_month[df_month["pickup_datetime"].dt.year == int(year)]
        df_month.dropna(subset=["pickup_datetime"], inplace=True)
        if not df_month.empty:
            all_trip_data.append(df_month[["pickup_datetime", "PULocationID"]])
    except Exception as e:
        print(f"  Warning: Error processing file {file_path}: {e}")
if not all_trip_data:
    print("ERROR: No data loaded...")
    exit()
print("Concatenating all taxi data...")
df_trips = pd.concat(all_trip_data, ignore_index=True)
del all_trip_data
print(f"Total valid trips: {len(df_trips)}")
print("Aggregating demand...")
df_trips["pickup_hour"] = df_trips["pickup_datetime"].dt.floor("h")
df_demand = (
    df_trips.groupby(["pickup_hour", "PULocationID"]).size().reset_index(name="demand")
)
del df_trips
print("Pivoting demand data...")
df_pivot = df_demand.pivot_table(
    index="pickup_hour", columns="PULocationID", values="demand", fill_value=0
)
print("\nApplying Timezone (Converting to UTC) and Creating Full 2023-2024 Index...")
try:
    if df_pivot.index.tz is None:
        print("Taxi index naive, localizing directly to UTC...")
        df_pivot.index = df_pivot.index.tz_localize("UTC")
    elif df_pivot.index.tz != "UTC":
        print(f"Converting taxi index ({df_pivot.index.tz}) to UTC...")
        df_pivot.index = df_pivot.index.tz_convert("UTC")
    else:
        print("Taxi index already UTC.")
except Exception as tz_err:
    print(f"ERROR: TZ handling failed for taxi data: {tz_err}.")
    exit()
start_full_utc = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
end_full_utc = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
all_hours_utc = pd.date_range(start=start_full_utc, end=end_full_utc, freq="h")
df_pivot = df_pivot.reindex(all_hours_utc, fill_value=0)
df_pivot = df_pivot.reindex(columns=VALID_ZONE_IDS, fill_value=0)
df_pivot.fillna(0, inplace=True)
df_pivot = df_pivot.astype(int)
print(f"Pivoted demand DataFrame shape (UTC index, 2 years): {df_pivot.shape}")
print("\nGenerating time-based features (from 2-year UTC index)...")
df_features = pd.DataFrame(index=df_pivot.index)
df_features["year"] = df_features.index.year
df_features["hour"] = df_features.index.hour
df_features["dayofweek"] = df_features.index.dayofweek
df_features["dayofmonth"] = df_features.index.day
df_features["month"] = df_features.index.month
df_features["quarter"] = df_features.index.quarter
df_features["dayofyear"] = df_features.index.dayofyear
df_features["weekofyear"] = df_features.index.isocalendar().week.astype(int)
df_features["is_weekend"] = df_features["dayofweek"].isin([5, 6]).astype(int)
hist_years = df_features.index.year.unique()
us_holidays = holidays.US(years=hist_years)
df_features["is_holiday"] = df_features.index.date
df_features["is_holiday"] = (
    df_features["is_holiday"].map(lambda dt: 1 if dt in us_holidays else 0).astype(int)
)
seconds_in_day = 24 * 60 * 60
seconds_in_year = 365.2425 * seconds_in_day
timestamps_sec = df_features.index.astype(np.int64) // 10**9
df_features["hour_sin"] = np.sin(timestamps_sec * (2 * np.pi / seconds_in_day))
df_features["hour_cos"] = np.cos(timestamps_sec * (2 * np.pi / seconds_in_day))
df_features["month_sin"] = np.sin(timestamps_sec * (2 * np.pi / seconds_in_year))
df_features["month_cos"] = np.cos(timestamps_sec * (2 * np.pi / seconds_in_year))
print(f"Time+Year features generated. Shape: {df_features.shape}")
print("\nFetching historical weather data for 2023 using Meteostat...")
df_weather_2023 = None
try:
    location = Point(NYC_LAT, NYC_LON, NYC_ALTITUDE)
    start_2023 = datetime(2023, 1, 1)
    end_2023 = datetime(2023, 12, 31, 23, 59, 59)
    print(f"Fetching weather from {start_2023} to {end_2023}")

    weather_fetch_data_2023 = Hourly(location, start_2023, end_2023)
    df_weather_2023 = weather_fetch_data_2023.fetch()

    if df_weather_2023.empty:
        print("ERROR: Meteostat fetch returned no data for 2023.")
        exit()
    print(f"Fetched {len(df_weather_2023)} hourly weather records for 2023.")
    df_weather_2023.index = pd.to_datetime(df_weather_2023.index, utc=True)
    df_weather_2023 = df_weather_2023.rename(
        columns={
            "temp": "temperature",
            "prcp": "precipitation",
            "snow": "snow_depth",
            "rhum": "humidity",
            "wspd": "wind_speed",
        }
    )
    missing_cols_23 = [
        col for col in WEATHER_COLS_TO_USE if col not in df_weather_2023.columns
    ]
    if missing_cols_23:
        print(
            f"Warning: Cols missing in fetched 2023 weather: {missing_cols_23}. Filling with 0."
        )
        for col in missing_cols_23:
            df_weather_2023[col] = 0.0
    df_weather_2023 = df_weather_2023[WEATHER_COLS_TO_USE]
    if "snow_depth" in df_weather_2023.columns:
        df_weather_2023["snow_depth"].fillna(0, inplace=True)
    df_weather_2023 = df_weather_2023.interpolate(method="time").ffill().bfill()
    if df_weather_2023.isnull().values.any():
        print("ERROR: NaNs remain in fetched 2023 weather!")
        exit()
    print(f"Processed 2023 weather data shape: {df_weather_2023.shape}")

except Exception as e:
    print(f"Error fetching/processing 2023 weather data: {e}")
    traceback.print_exc()
    exit()
print(
    f"\nLoading historical weather data for 2024 from {HISTORICAL_WEATHER_PATH_2024}..."
)
df_weather_2024 = None
try:
    df_weather_2024 = pd.read_csv(
        HISTORICAL_WEATHER_PATH_2024, index_col=0, parse_dates=True
    )
    if df_weather_2024.index.tz is None:
        df_weather_2024.index = df_weather_2024.index.tz_localize("UTC")
    elif df_weather_2024.index.tz != "UTC":
        df_weather_2024.index = df_weather_2024.index.tz_convert("UTC")
    missing_cols_24 = [
        col for col in WEATHER_COLS_TO_USE if col not in df_weather_2024.columns
    ]
    if missing_cols_24:
        print(f"ERROR: Cols missing in 2024 weather CSV: {missing_cols_24}")
        exit()
    df_weather_2024 = df_weather_2024[WEATHER_COLS_TO_USE]
    if "snow_depth" in df_weather_2024.columns:
        df_weather_2024["snow_depth"].fillna(0, inplace=True)
    df_weather_2024 = df_weather_2024.interpolate(method="time").ffill().bfill()
    if df_weather_2024.isnull().values.any():
        print("ERROR: NaNs remain in loaded 2024 weather!")
        exit()
    print(f"Loaded and processed 2024 weather data shape: {df_weather_2024.shape}")

except FileNotFoundError:
    print(f"ERROR: 2024 Weather file not found: {HISTORICAL_WEATHER_PATH_2024}")
    exit()
except Exception as e:
    print(f"Error loading/processing 2024 weather data: {e}")
    traceback.print_exc()
    exit()
print("\nConcatenating 2023 and 2024 weather data...")
df_weather_hist_combined = pd.concat([df_weather_2023, df_weather_2024])
df_weather_hist_combined.sort_index(inplace=True)
print(f"Combined historical weather data shape: {df_weather_hist_combined.shape}")
print("\nMerging Time, Year, and Weather features...")
try:
    df_weather_hist_aligned = (
        df_weather_hist_combined.reindex(df_features.index).ffill().bfill()
    )
    df_features.index.name = "ts_feat"
    df_weather_hist_aligned.index.name = "ts_weather"
    df_features_combined = df_features.join(df_weather_hist_aligned, how="left")
    df_features_combined.index.name = "timestamp"
    if df_features_combined.isnull().values.any():
        print("Warning: NaNs found after final join. Filling with 0...")
        df_features_combined.fillna(0, inplace=True)
    if df_features_combined.isnull().values.any():
        print("ERROR: NaNs remain!")
        exit()
    else:
        print("No NaNs found in final combined features.")
    print(f"Combined features shape before scaling: {df_features_combined.shape}")
    FEATURE_NAMES = df_features_combined.columns.tolist()
except Exception as e:
    print(f"Error merging features: {e}")
    traceback.print_exc()
    exit()
print("\nScaling combined features (Time + Year + Weather)...")
feature_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(df_features_combined)
df_scaled_features = pd.DataFrame(
    scaled_features, index=df_features_combined.index, columns=FEATURE_NAMES
)
print("\nSaving 2-year processed data and assets...")
df_target = df_pivot
target_file = os.path.join(OUTPUT_DIR, "target_demand_ml_2yr.parquet")
df_target.to_parquet(target_file)
print(f"Target saved (Shape: {df_target.shape})")
features_file = os.path.join(OUTPUT_DIR, "scaled_features_ml_2yr.parquet")
df_scaled_features.to_parquet(features_file)
print(f"Features saved (Shape: {df_scaled_features.shape})")
scaler_filename_features = os.path.join(MODEL_DIR, "feature_scaler_ml_2yr.joblib")
joblib.dump(feature_scaler, scaler_filename_features)
print(f"Scaler saved.")
zone_order_file = os.path.join(MODEL_DIR, "zone_order_ml.joblib")
joblib.dump(VALID_ZONE_IDS, zone_order_file)
print(f"Zone order saved.")
feature_names_file = os.path.join(MODEL_DIR, "feature_names_ml_2yr.joblib")
joblib.dump(FEATURE_NAMES, feature_names_file)
print(f"Feature names saved.")
print("\n--- Data processing (2023(fetched) & 2024(loaded) with weather) complete. ---")
