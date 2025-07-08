import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib
import warnings
import traceback

warnings.filterwarnings("ignore", message="Found `n_estimators` in params")
warnings.filterwarnings(
    "ignore", message="[LightGBM] [Warning] feature_fraction is set"
)
warnings.filterwarnings(
    "ignore", message="[LightGBM] [Warning] bagging_fraction is set"
)
warnings.filterwarnings("ignore", message="[LightGBM] [Warning] bagging_freq is set")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_ml/")
MODEL_DIR = os.path.join(BASE_DIR, "models_ml/")
PROCESSED_TARGET_FILE = os.path.join(OUTPUT_DIR, "target_demand_ml_2yr.parquet")
PROCESSED_FEATURES_FILE = os.path.join(OUTPUT_DIR, "scaled_features_ml_2yr.parquet")
ZONE_ORDER_FILE = os.path.join(MODEL_DIR, "zone_order_ml.joblib")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "lgbm_demand_model_2yr.joblib")
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_leaves": 31,
    "max_depth": -1,
    "seed": 42,
    "n_jobs": -1,
    "verbose": -1,
    "boosting_type": "gbdt",
}
WRAPPER_N_JOBS = -1
print("--- Model Training Script (2-Year Data) ---")
print(f"Loading data from: {os.path.abspath(OUTPUT_DIR)}")
print(f"Saving model components to: {os.path.abspath(MODEL_DIR)}")
print("\nLoading 2-year processed data...")
try:
    df_target = pd.read_parquet(PROCESSED_TARGET_FILE)
    df_scaled_features = pd.read_parquet(PROCESSED_FEATURES_FILE)
    zone_order = joblib.load(ZONE_ORDER_FILE)
    NUM_ZONES = len(zone_order)
    df_target = df_target.reindex(df_scaled_features.index)
    df_target = df_target[zone_order]
    print("Data loaded successfully.")
    print(f"Features shape: {df_scaled_features.shape}")
    print(f"Target shape: {df_target.shape}")
except FileNotFoundError as fnf:
    print(f"ERROR: Required file not found: {fnf}. Run load_data.py first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
    exit()
if df_scaled_features.empty or df_target.empty:
    print("ERROR: Loaded data is empty.")
    exit()
if len(df_scaled_features) != len(df_target):
    print("ERROR: Row count mismatch.")
    exit()
if df_scaled_features.isnull().values.any():
    print("ERROR: NaNs in features.")
    exit()
if df_target.isnull().values.any():
    print("ERROR: NaNs in target.")
    exit()
X = df_scaled_features
y = df_target
print("\nDefining Base Estimator (LightGBM)...")
base_lgbm_estimator = lgb.LGBMRegressor(**LGBM_PARAMS)
print(f"\nSkipping Time Series Cross-Validation step.")
print("\nAttempting to train final Multi-Output LightGBM model (2-Year Data)...")
final_model_wrapped = MultiOutputRegressor(base_lgbm_estimator, n_jobs=WRAPPER_N_JOBS)
print(f"Using n_jobs={WRAPPER_N_JOBS} for parallel training.")
try:
    print(f"Fitting final model with X shape {X.shape} and y shape {y.shape}...")
    start_time = pd.Timestamp.now()
    final_model_wrapped.fit(X, y)
    end_time = pd.Timestamp.now()
    print(f"Final model training complete. Duration: {end_time - start_time}")
    print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
    joblib.dump(final_model_wrapped, MODEL_SAVE_PATH)
    print(f"Final wrapped LightGBM model (2-Year) saved successfully.")
except Exception as fit_err:
    print(f"---!!! FINAL FITTING FAILED !!!---")
    print(f"Error: {fit_err}")
    traceback.print_exc()
    exit()

print("\n--- Model Training Script End ---")
