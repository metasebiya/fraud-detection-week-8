import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🔍 Fraud Detection Monitoring Dashboard")

# Load metrics
metrics = pd.read_json("results/metrics.json")
st.metric("Latest F1 Score", f"{metrics['f1_score']:.2f}")
st.metric("Latest AUC-PR", f"{metrics['auc_pr']:.2f}")

# SHAP Summary Plot
st.subheader("📈 SHAP Feature Importance")
summary_plot_path = "results/lgbm_fraud_shap_summary.png"
st.image(summary_plot_path, caption="SHAP Summary Plot")

# Drift Detection (placeholder)
st.subheader("📉 Drift Detection")
drift_detected = metrics.get("drift_detected", False)
if drift_detected:
    st.error("⚠️ Drift detected! Consider retraining.")
else:
    st.success("✅ No drift detected.")

# Prediction Explorer
st.subheader("🔎 Individual Prediction Explorer")
model = joblib.load("models/lgbm_fraud.pkl")
input_data = st.text_input("Enter JSON input for prediction:")

if input_data:
    try:
        df = pd.DataFrame([eval(input_data)])
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        st.write(f"Prediction: {'Fraud' if prediction else 'Legit'}")
        st.write(f"Probability of Fraud: {prob:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
