"""
Credit Card Fraud Detection 2.0
Training script with MLflow experiment tracking.

Replicates the modeling workflow from notebooks/modeling.ipynb
and logs parameters, metrics, and the model artifact to MLflow.

Usage:
    python train.py
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT        = Path(__file__).resolve().parent
DATA_PATH   = ROOT / "data" / "processed" / "fraud_clean.csv"
MODEL_PATH  = ROOT / "models" / "fraud_detection_pipeline.joblib"


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE     = 42
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15
THRESHOLD        = 0.75

FEATURES = [
    "amt",
    "category",
    "merchant",
    "hour",
    "age",
    "gender",
    "job",
    "distance_km",
]

CATEGORICAL_FEATURES = ["category", "merchant", "gender", "job"]
NUMERICAL_FEATURES   = ["amt", "hour", "age", "distance_km"]

XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_STATE,
    "use_label_encoder": False,
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "age" not in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["dob"] = pd.to_datetime(df["dob"])
        df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

    if "hour" not in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hour"] = df["trans_date_trans_time"].dt.hour

    if "distance_km" not in df.columns:
        lat1 = np.radians(df["lat"].values)
        lon1 = np.radians(df["long"].values)
        lat2 = np.radians(df["merch_lat"].values)
        lon2 = np.radians(df["merch_long"].values)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df["distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a))

    return df


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ]
    )

    xgb_params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**xgb_params)),
        ]
    )


# ============================================================
# TRAINING
# ============================================================

def train():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION 2.0 — MLflow Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    print(f"      Dataset shape: {df.shape}")

    X = df[FEATURES]
    y = df["is_fraud"]

    fraud_rate = y.mean()
    scale_pos_weight = round((1 - fraud_rate) / fraud_rate, 2)
    print(f"      Fraud rate: {fraud_rate:.4%}")
    print(f"      scale_pos_weight: {scale_pos_weight}")

    # ── Split ────────────────────────────────────────────────
    print("\n[2/5] Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print(f"      Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── MLflow run ───────────────────────────────────────────
    print("\n[3/5] Starting MLflow run...")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-card-fraud-detection")

    with mlflow.start_run(run_name="xgboost-final"):

        # Log parameters
        mlflow.log_params({
            "features":          FEATURES,
            "n_features":        len(FEATURES),
            "categorical":       CATEGORICAL_FEATURES,
            "numerical":         NUMERICAL_FEATURES,
            "threshold":         THRESHOLD,
            "scale_pos_weight":  scale_pos_weight,
            "test_size":         TEST_SIZE,
            "val_size":          VAL_SIZE,
            "random_state":      RANDOM_STATE,
            **{f"xgb_{k}": v for k, v in XGBOOST_PARAMS.items()
               if k not in ("eval_metric", "use_label_encoder")},
        })

        # ── Train ────────────────────────────────────────────
        print("\n[4/5] Training model...")
        pipeline = build_pipeline(scale_pos_weight)
        pipeline.fit(X_train, y_train)

        # ── Evaluate on validation set ───────────────────────
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_pred  = (val_proba >= THRESHOLD).astype(int)

        val_metrics = {
            "val_roc_auc":   round(roc_auc_score(y_val, val_proba), 4),
            "val_pr_auc":    round(average_precision_score(y_val, val_proba), 4),
            "val_precision": round(precision_score(y_val, val_pred), 4),
            "val_recall":    round(recall_score(y_val, val_pred), 4),
            "val_f1":        round(f1_score(y_val, val_pred), 4),
        }
        mlflow.log_metrics(val_metrics)

        print("\n      Validation metrics:")
        for k, v in val_metrics.items():
            print(f"      {k}: {v}")

        # ── Evaluate on test set ─────────────────────────────
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred  = (test_proba >= THRESHOLD).astype(int)

        test_metrics = {
            "test_roc_auc":   round(roc_auc_score(y_test, test_proba), 4),
            "test_pr_auc":    round(average_precision_score(y_test, test_proba), 4),
            "test_precision": round(precision_score(y_test, test_pred), 4),
            "test_recall":    round(recall_score(y_test, test_pred), 4),
            "test_f1":        round(f1_score(y_test, test_pred), 4),
        }
        mlflow.log_metrics(test_metrics)

        print("\n      Test metrics:")
        for k, v in test_metrics.items():
            print(f"      {k}: {v}")

        # ── Save artifact ─────────────────────────────────────
        print("\n[5/5] Saving artifact...")

        preprocessor = pipeline.named_steps["preprocessor"]
        encoder      = preprocessor.named_transformers_["categorical"]

        category_values = {
            col: cats.tolist()
            for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)
        }

        artifact = {
            "model":            pipeline,
            "threshold":        THRESHOLD,
            "features":         FEATURES,
            "category_values":  category_values,
        }

        joblib.dump(artifact, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        print(f"      Artifact saved: {MODEL_PATH}")
        print(f"\n      MLflow run ID: {mlflow.active_run().info.run_id}")

    print("\n" + "=" * 60)
    print("Training complete. Run: mlflow ui")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()
