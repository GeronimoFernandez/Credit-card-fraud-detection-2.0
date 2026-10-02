import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# ── Allow imports from the project root ──────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.utils import load_model_artifact, prepare_transaction, predict_fraud
from api.schemas import TransactionRequest, PredictionResponse, HealthResponse


# ============================================================
# LIFESPAN — load model once at startup
# ============================================================

artifact = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact["model"] = load_model_artifact()
    print("Model artifact loaded.")
    yield
    artifact.clear()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Inference API for the Credit Card Fraud Detection model. "
        "Send transaction data and receive a fraud probability and classification."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check"
)
def health():
    """
    Returns the API status and whether the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded="model" in artifact
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict fraud probability for a transaction"
)
def predict(request: TransactionRequest):
    """
    Accepts transaction data and returns:
    - **fraud_probability**: model output (0.0 to 1.0)
    - **prediction**: 1 = fraudulent, 0 = legitimate
    - **classification**: human-readable label
    - **threshold**: decision boundary applied
    """
    if "model" not in artifact:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        transaction = prepare_transaction(
            amount=request.amount,
            category=request.category,
            merchant=request.merchant,
            transaction_datetime=request.transaction_datetime,
            date_of_birth=request.date_of_birth,
            gender=request.gender,
            job=request.job,
            customer_lat=request.customer_lat,
            customer_lon=request.customer_lon,
            merchant_lat=request.merchant_lat,
            merchant_lon=request.merchant_lon
        )

        probability, prediction = predict_fraud(
            transaction,
            artifact["model"]
        )

        classification = (
            "fraudulent" if prediction == 1 else "legitimate"
        )

        return PredictionResponse(
            fraud_probability=round(float(probability), 6),
            prediction=prediction,
            classification=classification,
            threshold=artifact["model"]["threshold"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
