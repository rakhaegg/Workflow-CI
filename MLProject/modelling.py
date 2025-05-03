"""
modelling.py  –  XGBoost GPU baseline + manual MLflow logging
----------------------------------------------------------------
•  Membaca train / test parquet keluaran preprocess
•  One‑Hot Encode kategori
•  Melatih XGBoost (gpu_hist)
•  Mencatat param & metrik ke MLflow aktif
•  Menyimpan artefak model ⇢ models/xgb_gpu.json
Cli:
    python modelling.py --num_boost_round 300
"""
import argparse, time, os, json, joblib, mlflow, xgboost as xgb
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ────────────────────────── CLI ARGUMENTS ───────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_boost_round", type=int, default=300)
args = parser.parse_args()

# ───────────────────────────── DATA ─────────────────────────────────
DATA_DIR = Path("namadataset_preprocessing")
train = pd.read_parquet(DATA_DIR / "train.parquet")
test  = pd.read_parquet(DATA_DIR / "test.parquet")
TARGET = "Log_Demand"

cat_cols = ["Product_Category", "Warehouse", "day_of_week"]
num_cols = [c for c in train.columns if c.startswith(("lag_", "roll_mean_"))]

ohe = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)
X_train = ohe.fit_transform(train[cat_cols + num_cols])
X_test  = ohe.transform(test [cat_cols + num_cols])
y_train, y_test = train[TARGET].values, test[TARGET].values

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test , label=y_test)

# ────────────────────────── MLFLOW RUN HANDLER ─────────────────────
# Jika script dipanggil via `mlflow run .`, environment sudah punya MLFLOW_RUN_ID.
# Bila dipanggil langsung, buat run baru agar UI tetap mencatat.
if mlflow.active_run() is None:
    mlflow.start_run(run_name="standalone_xgb_gpu")

# ─────────────────────────── TRAINING ──────────────────────────────
params = {
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "gpu_hist",
    "device" : "gpu",
    "tree_method": "hist"
}

tic = time.time()
booster = xgb.train(params, dtrain, num_boost_round=args.num_boost_round)
train_sec = time.time() - tic

preds = booster.predict(dtest)
mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / y_test))
p90_err = np.percentile(np.abs(y_test - preds), 90)

# ────────────────────────── LOGGING ────────────────────────────────
mlflow.log_params(params)
mlflow.log_param("num_boost_round", args.num_boost_round)
mlflow.log_metric("MAE",  mae)
mlflow.log_metric("RMSE", rmse)
mlflow.log_metric("MAPE", mape)               # extra 1
mlflow.log_metric("P90_absolute_error", p90_err)  # extra 2
mlflow.log_metric("train_time_sec", train_sec)

# ────────────────────────── SAVE MODEL ─────────────────────────────
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
model_path = OUT_DIR / "xgb_gpu.json"
booster.save_model(model_path)
mlflow.log_artifact(str(model_path), artifact_path="model")

# Opsional: simpan OHE encoder untuk serving
enc_path = OUT_DIR / "ohe_encoder.pkl"
joblib.dump(ohe, enc_path)
mlflow.log_artifact(str(enc_path), artifact_path="model")

print(f"✅  Training selesai in {train_sec:.1f}s | MAE={mae:.3f} | artefak → {model_path}")

# ────────────────────────── END RUN (optional) ─────────────────────
if mlflow.active_run():
    mlflow.end_run()
