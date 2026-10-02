import streamlit as st
from datetime import datetime, date, time

from utils import (
    load_model_artifact,
    prepare_transaction,
    predict_fraud
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model_artifact():
    return load_model_artifact()


artifact = get_model_artifact()

category_values = artifact["category_values"]


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction information to estimate "
    "the probability of fraud."
)


# ============================================================
# TRANSACTION INFORMATION
# ============================================================

st.header("Transaction")

amount = st.number_input(
    "Transaction amount ($)",
    min_value=0.0,
    value=100.0,
    step=1.0
)

category = st.selectbox(
    "Category",
    options=category_values["category"]
)

merchant = st.selectbox(
    "Merchant",
    options=category_values["merchant"]
)

transaction_date = st.date_input(
    "Transaction date",
    value=date.today()
)

transaction_time = st.time_input(
    "Transaction time",
    value=time(12, 0)
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("Customer")

date_of_birth = st.date_input(
    "Date of birth",
    value=date(1990, 1, 1)
)

gender = st.radio(
    "Gender",
    options=category_values["gender"],
    horizontal=True
)

job = st.selectbox(
    "Job",
    options=category_values["job"]
)


# ============================================================
# LOCATION
# ============================================================

st.header("Location")

st.write(
    "Enter the customer's coordinates and the merchant's "
    "coordinates."
)

customer_lat = st.number_input(
    "Customer latitude",
    value=0.0,
    format="%.6f"
)

customer_lon = st.number_input(
    "Customer longitude",
    value=0.0,
    format="%.6f"
)

merchant_lat = st.number_input(
    "Merchant latitude",
    value=0.0,
    format="%.6f"
)

merchant_lon = st.number_input(
    "Merchant longitude",
    value=0.0,
    format="%.6f"
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Analyze transaction",
    type="primary",
    use_container_width=True
):
    errors = []

    # Amount validation
    if amount <= 0:
        errors.append(
            "Transaction amount must be greater than 0."
        )

    # Date validation
    if date_of_birth > transaction_date:
        errors.append(
            "Date of birth cannot be later than the transaction date."
        )

    # Latitude validation
    if not -90 <= customer_lat <= 90:
        errors.append(
            "Customer latitude must be between -90 and 90."
        )

    if not -90 <= merchant_lat <= 90:
        errors.append(
            "Merchant latitude must be between -90 and 90."
        )

    # Longitude validation
    if not -180 <= customer_lon <= 180:
        errors.append(
            "Customer longitude must be between -180 and 180."
        )

    if not -180 <= merchant_lon <= 180:
        errors.append(
            "Merchant longitude must be between -180 and 180."
        )

    # ========================================================
    # VALIDATION ERRORS
    # ========================================================

    if errors:
        st.error("Please correct the following inputs:")

        for error in errors:
            st.write(f"- {error}")

    # ========================================================
    # PREDICTION
    # ========================================================

    else:
        transaction_datetime = datetime.combine(
            transaction_date,
            transaction_time
        )

        transaction = prepare_transaction(
            amount=amount,
            category=category,
            merchant=merchant,
            transaction_datetime=transaction_datetime,
            date_of_birth=date_of_birth,
            gender=gender,
            job=job,
            customer_lat=customer_lat,
            customer_lon=customer_lon,
            merchant_lat=merchant_lat,
            merchant_lon=merchant_lon
        )

        probability, prediction = predict_fraud(
            transaction,
            artifact
        )

        st.divider()

        st.subheader("Prediction")

        st.metric(
            "Fraud probability",
            f"{probability:.2%}"
        )

        if prediction == 1:
            st.error(
                "⚠️ Potential fraudulent transaction"
            )
        else:
            st.success(
                "✅ Transaction classified as legitimate"
            )