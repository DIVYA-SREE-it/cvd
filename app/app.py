import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================
# LOAD MODEL SAFELY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

loaded_obj = pickle.load(open(model_path, "rb"))

# Handle both cases safely
if isinstance(loaded_obj, tuple):
    model, scaler, feature_names = loaded_obj
else:
    model = loaded_obj
    scaler = None
    feature_names = [
        "age","sex","chol","trestbps",
        "steps","sleep_hours","avg_hr",
        "lifestyle_score","stress_index"
    ]

# =========================
# UI
# =========================
st.set_page_config(page_title="CVD Risk", page_icon="❤️")

st.title("❤️ Cardiovascular Risk Prediction")
st.write("AI-based heart risk estimation using lifestyle + clinical data")

# =========================
# INPUTS
# =========================
age = st.slider("Age", 20, 80, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chol = st.number_input("Cholesterol", 100, 400, 200)
bp = st.number_input("Blood Pressure", 80, 200, 120)

steps = st.slider("Daily Steps", 0, 20000, 5000)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 6.5)
hr = st.slider("Heart Rate", 40, 120, 70)

sex = 1 if sex == "Male" else 0

# =========================
# FEATURE ENGINEERING
# =========================
lifestyle_score = (steps * 0.3) + (sleep * 0.3) + (1/(hr+1) * 0.4)
stress_index = hr / (sleep + 1)

# =========================
# BUILD INPUT
# =========================
input_dict = {
    "age": age,
    "sex": sex,
    "chol": chol,
    "trestbps": bp,
    "steps": steps,
    "sleep_hours": sleep,
    "avg_hr": hr,
    "lifestyle_score": lifestyle_score,
    "stress_index": stress_index
}

# Ensure feature alignment
for col in feature_names:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[feature_names]

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Risk"):
    try:
        if scaler:
            X = scaler.transform(input_df)
        else:
            X = input_df.values

        risk = model.predict_proba(X)[0][1]

        # Risk category
        if risk < 0.3:
            level = "Low Risk 🟢"
        elif risk < 0.6:
            level = "Moderate Risk 🟡"
        else:
            level = "High Risk 🔴"

        st.subheader("Result")
        st.metric("Risk Score", f"{risk:.2f}")
        st.success(level)

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# DEBUG (remove later)
# =========================
with st.expander("Debug Info"):
    st.write("Model Loaded:", type(model))
    st.write("Scaler Present:", scaler is not None)
    st.write("Features:", feature_names)
