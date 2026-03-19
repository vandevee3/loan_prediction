import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="wide")

HOME_OWNERSHIP_OPTIONS  = ["RENT", "MORTGAGE", "OWN", "OTHER"]
LOAN_INTENT_OPTIONS     = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
LOAN_GRADE_OPTIONS      = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_ON_FILE_OPTIONS = ["Y", "N"]

def call_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend API. Is it running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.text}")
    return None

st.title("💳 Loan Default Predictor")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("👤 Personal Information")
    person_age            = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    person_income         = st.number_input("Annual Income ($)", min_value=0, max_value=10_000_000, value=50000, step=1000)
    person_home_ownership = st.selectbox("Home Ownership", HOME_OWNERSHIP_OPTIONS)
    person_emp_length     = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    cb_person_default_on_file  = st.selectbox("Historical Default on File", DEFAULT_ON_FILE_OPTIONS)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5, step=1)

with col2:
    st.subheader("🏦 Loan Information")
    loan_intent         = st.selectbox("Loan Intent", LOAN_INTENT_OPTIONS)
    loan_grade          = st.selectbox("Loan Grade", LOAN_GRADE_OPTIONS)
    loan_amnt           = st.number_input("Loan Amount ($)", min_value=500, max_value=500_000, value=10000, step=500)
    loan_int_rate       = st.number_input("Interest Rate (%)", min_value=1.0, max_value=50.0, value=11.0, step=0.1)
    loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")

st.divider()

payload = {
    "person_age":                 person_age,
    "person_income":              person_income,
    "person_home_ownership":      person_home_ownership,
    "person_emp_length":          person_emp_length,
    "loan_intent":                loan_intent,
    "loan_grade":                 loan_grade,
    "loan_amnt":                  loan_amnt,
    "loan_int_rate":              loan_int_rate,
    "loan_percent_income":        loan_percent_income,
    "cb_person_default_on_file":  cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
}

if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):
    with st.spinner("Predicting…"):
        result = call_predict(payload)

    if result:
        pred = result["prediction"]
        if pred == 1:
            st.error("❌ Default — This applicant is likely to default.")
        else:
            st.success("✅ No Default — This applicant is unlikely to default.")