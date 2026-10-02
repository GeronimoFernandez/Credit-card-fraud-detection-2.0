from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator


# ============================================================
# REQUEST
# ============================================================

class TransactionRequest(BaseModel):
    """
    Input schema for a fraud detection request.
    Mirrors the fields used by prepare_transaction() in utils.py.
    """

    amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount in USD. Must be greater than 0.",
        example=150.0
    )

    category: str = Field(
        ...,
        description="Merchant category (e.g. entertainment, food_dining).",
        example="entertainment"
    )

    merchant: str = Field(
        ...,
        description="Merchant name as it appears in the training data.",
        example="fraud_Abbott-Rogahn"
    )

    transaction_datetime: datetime = Field(
        ...,
        description="Full datetime of the transaction (ISO 8601 format).",
        example="2026-10-02T14:30:00"
    )

    date_of_birth: date = Field(
        ...,
        description="Customer date of birth (YYYY-MM-DD).",
        example="1990-01-01"
    )

    gender: str = Field(
        ...,
        description="Customer gender. Accepted values: F, M.",
        example="F"
    )

    job: str = Field(
        ...,
        description="Customer occupation as it appears in the training data.",
        example="Academic librarian"
    )

    customer_lat: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Customer latitude (-90 to 90).",
        example=40.7128
    )

    customer_lon: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Customer longitude (-180 to 180).",
        example=-74.0060
    )

    merchant_lat: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Merchant latitude (-90 to 90).",
        example=40.6892
    )

    merchant_lon: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Merchant longitude (-180 to 180).",
        example=-74.0445
    )

    @field_validator("date_of_birth")
    @classmethod
    def birth_before_transaction(cls, dob):
        if dob >= date.today():
            raise ValueError(
                "date_of_birth must be in the past."
            )
        return dob


# ============================================================
# RESPONSE
# ============================================================

class PredictionResponse(BaseModel):
    """
    Output schema for a fraud detection prediction.
    """

    fraud_probability: float = Field(
        ...,
        description="Model's estimated probability that the transaction is fraudulent (0.0 to 1.0)."
    )

    prediction: int = Field(
        ...,
        description="Binary classification result. 1 = fraudulent, 0 = legitimate."
    )

    classification: str = Field(
        ...,
        description="Human-readable classification label."
    )

    threshold: float = Field(
        ...,
        description="Decision threshold used to produce the binary prediction."
    )


# ============================================================
# HEALTH
# ============================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
