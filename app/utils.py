import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_detection_pipeline.joblib"


# ============================================================
# MODEL
# ============================================================

def load_model_artifact():
    """
    Load the trained fraud detection model artifact.
    """
    return joblib.load(MODEL_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_age(date_of_birth, transaction_datetime):
    return transaction_datetime.year - date_of_birth.year


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate the geographic distance between two coordinates
    using the Haversine formula.
    """

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


# ============================================================
# TRANSACTION PREPARATION
# ============================================================

def prepare_transaction(
    amount,
    category,
    merchant,
    transaction_datetime,
    date_of_birth,
    gender,
    job,
    customer_lat,
    customer_lon,
    merchant_lat,
    merchant_lon
):
    """
    Convert user inputs into the exact feature structure
    expected by the trained model.
    """

    age = calculate_age(
        date_of_birth,
        transaction_datetime
    )

    hour = transaction_datetime.hour

    distance_km = calculate_distance_km(
        customer_lat,
        customer_lon,
        merchant_lat,
        merchant_lon
    )

    transaction = pd.DataFrame(
        [
            {
                "amt": amount,
                "category": category,
                "merchant": merchant,
                "hour": hour,
                "age": age,
                "gender": gender,
                "job": job,
                "distance_km": distance_km
            }
        ]
    )

    return transaction


# ============================================================
# PREDICTION
# ============================================================

def predict_fraud(transaction, artifact):
    """
    Generate fraud probability and classification
    using the stored model and threshold.
    """

    model = artifact["model"]
    threshold = artifact["threshold"]
    features = artifact["features"]

    probability = model.predict_proba(
        transaction[features]
    )[0, 1]

    prediction = int(probability >= threshold)

    return probability, prediction