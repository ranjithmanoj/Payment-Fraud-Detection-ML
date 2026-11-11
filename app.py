import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path
from datetime import datetime


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Payment Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. JAPANESE MIDNIGHT THEME
# =====================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #0B0F19;
    color: #F5F5F5;
}

/* Page spacing */

.block-container {
    max-width: 1750px;
    padding-top: 0.8rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Hide default Streamlit elements */

#MainMenu, footer, header {
    visibility: hidden;
}

/* Main heading */

.title {
    background: linear-gradient(
        110deg,
        #171D30,
        #202C48,
        #18243B
    );

    color: #F8FAFC;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 32px;
    font-weight: 900;
    letter-spacing: 2px;
    margin-bottom: 12px;
    border: 1px solid #34415C;
}

/* Section headings */

.section-heading {
    color: #A5B4FC;
    font-size: 18px;
    font-weight: 850;
    padding-bottom: 9px;
    border-bottom: 2px solid #5965A8;
    margin-bottom: 14px;
}

/* Labels */

label {
    color: #DCE3F0 !important;
    font-weight: 750 !important;
    font-size: 15px !important;
}

/* Input fields */

div[data-baseweb="input"] {
    background-color: #20283A !important;
    border: 1px solid #39465F !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    color: #FFFFFF !important;
    font-weight: 650 !important;
    background-color: #20283A !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-baseweb="select"] {
    background-color: #20283A !important;
    border: 1px solid #39465F !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] * {
    color: #FFFFFF !important;
    font-weight: 650 !important;
}

/* Input cards */

[data-testid="stForm"] {
    background-color: #151B2B;
    padding: 18px 20px 12px 20px;
    border: 1px solid #303B52;
    border-radius: 14px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.25);
}

/* Prediction button */

.stFormSubmitButton > button {
    background: linear-gradient(
        100deg,
        #3B5BDB,
        #5965A8
    ) !important;

    color: #FFFFFF !important;
    font-size: 17px !important;
    font-weight: 850 !important;
    border: none !important;
    border-radius: 9px !important;
    height: 48px !important;
}

.stFormSubmitButton > button:hover {
    background: #4C63C7 !important;
}

/* Clear button */

.stButton > button {
    background-color: #252D40 !important;
    color: #DCE3F0 !important;
    border: 1px solid #46536D !important;
    font-weight: 750 !important;
    border-radius: 8px !important;
}

.stButton > button:hover {
    background-color: #343F58 !important;
    color: #FFFFFF !important;
    border: 1px solid #8293C8 !important;
}

/* Fraud result */

.fraud-result {
    background-color: #29171D;
    border: 1px solid #D94841;
    border-left: 7px solid #D94841;
    border-radius: 11px;
    padding: 15px;
    text-align: center;
    margin-top: 10px;
}

.fraud-title {
    color: #FF7770;
    font-size: 25px;
    font-weight: 900;
}

.fraud-description {
    color: #E8B9B7;
    font-size: 14px;
    font-weight: 600;
}

/* Legitimate result */

.safe-result {
    background-color: #142820;
    border: 1px solid #3FA77A;
    border-left: 7px solid #3FA77A;
    border-radius: 11px;
    padding: 15px;
    text-align: center;
    margin-top: 10px;
}

.safe-title {
    color: #68D6A5;
    font-size: 25px;
    font-weight: 900;
}

.safe-description {
    color: #B3E4CF;
    font-size: 14px;
    font-weight: 600;
}

/* Dashboard cards */

[data-testid="stMetric"] {
    background-color: #151B2B;
    border: 1px solid #303B52;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #A6AEC1 !important;
    font-size: 14px !important;
}

[data-testid="stMetricValue"] {
    color: #A5B4FC !important;
    font-size: 27px !important;
    font-weight: 850 !important;
}

/* Information cards */

.info-card {
    background-color: #151B2B;
    border: 1px solid #303B52;
    border-radius: 9px;
    padding: 12px;
    margin-top: 8px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #303B52;
    border-radius: 10px;
}

/* Divider */

hr {
    border-color: #303B52 !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
}

/* Footer */

.footer {
    color: #8994AC;
    text-align: center;
    font-size: 12px;
    margin-top: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Payments_fraud_detection_model.pkl"

SCALER_PATH = BASE_DIR / "scaler.pkl"

DATASET_PATH = BASE_DIR / "ML_Synthetic_payments.csv"


# =====================================================
# 4. LOAD MODEL AND SCALER
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


@st.cache_data
def load_dataset():

    if DATASET_PATH.exists():

        return pd.read_csv(DATASET_PATH)

    return None


# =====================================================
# 5. IQR OUTLIER LIMITS
# =====================================================

def calculate_iqr_limits(dataset):

    limits = {}

    if dataset is None:
        return limits

    columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    for column in columns:

        if column in dataset.columns:

            q1 = dataset[column].quantile(0.25)

            q3 = dataset[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            limits[column] = (lower, upper)

    return limits


# =====================================================
# 6. ACCOUNT ID CONVERSION
# =====================================================

def convert_account_id(account_id):

    digits = re.sub(
        r"[^0-9]",
        "",
        str(account_id)
    )

    if not digits:

        raise ValueError(
            "Enter a valid account ID."
        )

    return int(digits)


# =====================================================
# 7. PREPROCESSING
# =====================================================

def preprocess_transaction(
    transaction_type,
    amount,
    name_orig,
    oldbalance_org,
    newbalance_orig,
    name_dest,
    oldbalance_dest,
    newbalance_dest,
    limits
):

    type_mapping = {

        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4

    }

    transaction = {

        "type": type_mapping[transaction_type],

        "amount": amount,

        "nameOrig": convert_account_id(name_orig),

        "oldbalanceOrg": oldbalance_org,

        "newbalanceOrig": newbalance_orig,

        "nameDest": convert_account_id(name_dest),

        "oldbalanceDest": oldbalance_dest,

        "newbalanceDest": newbalance_dest

    }

    # IQR outlier capping

    for column, (lower, upper) in limits.items():

        if column in transaction:

            transaction[column] = np.clip(
                transaction[column],
                lower,
                upper
            )

    feature_order = [

        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest"

    ]

    return pd.DataFrame(
        [transaction],
        columns=feature_order
    )


# =====================================================
# 8. LOAD REQUIRED FILES
# =====================================================

try:

    model, scaler = load_model()

    dataset = load_dataset()

    limits = calculate_iqr_limits(dataset)

except Exception as error:

    st.error(
        "Unable to load the model or required files."
    )

    st.code(str(error))

    st.stop()


# =====================================================
# 9. SESSION STATE
# =====================================================

if "history" not in st.session_state:

    st.session_state.history = []

if "prediction_result" not in st.session_state:

    st.session_state.prediction_result = None

if "form_version" not in st.session_state:

    st.session_state.form_version = 0


# =====================================================
# 10. APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="title">PAYMENT FRAUD DETECTION</div>',
    unsafe_allow_html=True
)


# =====================================================
# 11. CLEAR TRANSACTION
# =====================================================

if st.button(
    "↻ Clear Transaction",
    key="clear_transaction"
):

    st.session_state.prediction_result = None

    st.session_state.form_version += 1

    st.rerun()


# =====================================================
# 12. TRANSACTION INPUT FORM
# =====================================================

with st.form(
    f"transaction_form_{st.session_state.form_version}"
):

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    # -----------------------------------------------
    # COLUMN 1 - TRANSACTION DETAILS
    # -----------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-heading">'
            '01 | TRANSACTION DETAILS'
            '</div>',
            unsafe_allow_html=True
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ],
            key=f"transaction_type_{st.session_state.form_version}"
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key=f"amount_{st.session_state.form_version}"
        )

    # -----------------------------------------------
    # COLUMN 2 - SENDER DETAILS
    # -----------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-heading">'
            '02 | SENDER DETAILS'
            '</div>',
            unsafe_allow_html=True
        )

        name_orig = st.text_input(
            "Sender Account ID",
            placeholder="C123456",
            key=f"name_orig_{st.session_state.form_version}"
        )

        oldbalance_org = st.number_input(
            "Sender Old Balance",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            key=f"oldbalance_org_{st.session_state.form_version}"
        )

        newbalance_orig = st.number_input(
            "Sender New Balance",
            min_value=0.0,
            value=4000.0,
            step=100.0,
            key=f"newbalance_orig_{st.session_state.form_version}"
        )

    # -----------------------------------------------
    # COLUMN 3 - RECEIVER DETAILS
    # -----------------------------------------------

    with col3:

        st.markdown(
            '<div class="section-heading">'
            '03 | RECEIVER DETAILS'
            '</div>',
            unsafe_allow_html=True
        )

        name_dest = st.text_input(
            "Receiver Account ID",
            placeholder="M987654",
            key=f"name_dest_{st.session_state.form_version}"
        )

        oldbalance_dest = st.number_input(
            "Receiver Old Balance",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            key=f"oldbalance_dest_{st.session_state.form_version}"
        )

        newbalance_dest = st.number_input(
            "Receiver New Balance",
            min_value=0.0,
            value=3000.0,
            step=100.0,
            key=f"newbalance_dest_{st.session_state.form_version}"
        )

    st.divider()

    submit = st.form_submit_button(
        "🔍 ANALYSE TRANSACTION",
        use_container_width=True
    )


# =====================================================
# 13. PREDICTION
# =====================================================

if submit:

    if not name_orig.strip() or not name_dest.strip():

        st.error(
            "Please enter both sender and receiver account IDs."
        )

    else:

        try:

            input_data = preprocess_transaction(

                transaction_type,
                amount,
                name_orig,
                oldbalance_org,
                newbalance_orig,
                name_dest,
                oldbalance_dest,
                newbalance_dest,
                limits

            )

            scaled_data = scaler.transform(input_data)

            prediction = int(
                model.predict(scaled_data)[0]
            )

            record = {

                "Transaction ID": len(
                    st.session_state.history
                ) + 1,

                "Date & Time": datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                ),

                "Type": transaction_type,

                "Amount": amount,

                "Prediction": (
                    "FRAUD"
                    if prediction == 1
                    else "LEGITIMATE"
                )

            }

            st.session_state.prediction_result = {

                "prediction": prediction,

                "type": transaction_type,

                "amount": amount,

                "sender": name_orig,

                "receiver": name_dest,

                "old_sender": oldbalance_org,

                "new_sender": newbalance_orig,

                "old_receiver": oldbalance_dest,

                "new_receiver": newbalance_dest,

                "time": record["Date & Time"]

            }

            st.session_state.history.append(record)

        except Exception as error:

            st.error(
                "Prediction failed. Please check your input values."
            )

            st.code(str(error))


# =====================================================
# 14. PREDICTION RESULT
# =====================================================
if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    if result["prediction"] == 1:

        st.markdown(
            """
            <div class="fraud-result">
                <div class="fraud-title">
                    🚨 FRAUD DETECTED
                </div>
                <div class="fraud-description">
                    The machine learning model classified
                    this transaction as fraudulent.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="safe-result">
                <div class="safe-title">
                    ✓ NOT FRAUD
                </div>
                <div class="safe-description">
                    The machine learning model classified
                    this transaction as legitimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    expected_sender_balance = (
        result["old_sender"] - result["amount"]
    )

    if not np.isclose(
        expected_sender_balance,
        result["new_sender"],
        atol=0.01
    ):

        st.warning(
            "Sender balance change does not match "
            "the transaction amount. Please verify the values."
        )

    with st.expander("📋 Transaction Summary"):

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.markdown("**TRANSACTION**")

            st.write("Type:", result["type"])

            st.write(
                "Amount:",
                f"₹ {result['amount']:,.2f}"
            )

        with summary_col2:

            st.markdown("**SENDER**")

            st.write("Account:", result["sender"])

            st.write(
                "Old Balance:",
                f"₹ {result['old_sender']:,.2f}"
            )

            st.write(
                "New Balance:",
                f"₹ {result['new_sender']:,.2f}"
            )

        with summary_col3:

            st.markdown("**RECEIVER**")

            st.write("Account:", result["receiver"])

            st.write(
                "Old Balance:",
                f"₹ {result['old_receiver']:,.2f}"
            )

            st.write(
                "New Balance:",
                f"₹ {result['new_receiver']:,.2f}"
            )

        st.caption(
            "Prediction generated on: " + result["time"]
        )

# =====================================================
# 15. TRANSACTION DASHBOARD
# =====================================================

st.markdown(
    '<div class="section-heading">TRANSACTION DASHBOARD</div>',
    unsafe_allow_html=True
)

history_df = pd.DataFrame(
    st.session_state.history
)

total = len(history_df)

if total > 0:

    fraud_count = int(
        (history_df["Prediction"] == "FRAUD").sum()
    )

    legitimate_count = total - fraud_count

    fraud_percentage = (
        fraud_count / total
    ) * 100

else:

    fraud_count = 0

    legitimate_count = 0

    fraud_percentage = 0


# Dashboard cards

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Transactions",
    total
)

c2.metric(
    "Fraud Detected",
    fraud_count
)

c3.metric(
    "Legitimate",
    legitimate_count
)

c4.metric(
    "Fraud Rate",
    f"{fraud_percentage:.1f}%"
)


# Dashboard chart and history

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.markdown("**PREDICTION DISTRIBUTION**")

    if total > 0:

        chart_data = pd.DataFrame(
            {
                "Transactions": [
                    fraud_count,
                    legitimate_count
                ]
            },
            index=[
                "Fraud",
                "Legitimate"
            ]
        )

        st.bar_chart(chart_data)

    else:

        st.caption(
            "Prediction statistics will appear here."
        )


with chart_col2:

    st.markdown("**RECENT TRANSACTIONS**")

    if total > 0:

        st.dataframe(
            history_df.tail(5),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.caption(
            "No transactions analysed yet."
        )


# Download history

if total > 0:

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Transaction History",
        csv,
        "payment_fraud_history.csv",
        "text/csv"
    )

# =====================================================
# 16. FOOTER
# =====================================================

st.markdown("""
<div class="footer">

    PAYMENT FRAUD DETECTION SYSTEM
    | POWERED BY MACHINE LEARNING

</div>
""", unsafe_allow_html=True)