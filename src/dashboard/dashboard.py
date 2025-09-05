import streamlit as st
import requests
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Fraud Detection Dashboard")
st.markdown("Enter transaction details to predict fraud risk using the latest Production model from DagsHub.")

# --- Sidebar for API health ---
st.sidebar.header("API Status")
try:
    health = requests.get(f"{API_URL}/health").json()
    if health.get("status") == "ok":
        st.sidebar.success(f"✅ API Online\nModel: {health.get('model_name')} ({health.get('model_stage')})")
    else:
        st.sidebar.error("❌ API Offline")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to API: {e}")

# --- Tabs for raw vs preprocessed input ---
tab1, tab2 = st.tabs(["Raw Transaction Input", "Preprocessed Features"])

with tab1:
    st.subheader("Raw Transaction Data")
    purchase_value = st.number_input("Purchase Value", min_value=0.0, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    source = st.selectbox("Source", ["SEO", "Ads", "Direct"])
    browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])
    sex = st.selectbox("Sex", ["M", "F"])
    hour_of_day = st.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.slider("Day of Week", 0, 6, 0)
    country = st.text_input("Country", "US")

    if st.button("Predict (Raw Data)", key="raw"):
        payload = {
            "purchase_value": purchase_value,
            "age": age,
            "source": source,
            "browser": browser,
            "sex": sex,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "country": country
        }
        try:
            response = requests.post(f"{API_URL}/predict-raw", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}** | Probability: **{result['probability']:.2f}**")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

with tab2:
    st.subheader("Preprocessed Features")
    features_str = st.text_area("Enter features as comma-separated values", "")
    if st.button("Predict (Preprocessed)", key="preprocessed"):
        try:
            features = [float(x.strip()) for x in features_str.split(",") if x.strip()]
            payload = {"features": features}
            response = requests.post(f"{API_URL}/predict-preprocessed", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}** | Probability: **{result['probability']:.2f}**")
            else:
                st.error(f"Error: {response.text}")
        except ValueError:
            st.error("Invalid input: please enter numeric values separated by commas.")
        except Exception as e:
            st.error(f"Request failed: {e}")
