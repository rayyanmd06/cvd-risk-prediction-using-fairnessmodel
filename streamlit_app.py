
import os
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# Page config 
st.set_page_config(
    page_title="💓 CVD Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Banner & CSS
if os.path.exists("assets/heart_logo.png"):
    banner = Image.open("assets/heart_logo.png")
    st.image(banner, use_column_width=True)

st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; }
      .stButton>button { background-color: #e63946; color: white; }
      .stSelectbox>div>div { background-color: #ffa69e; }
      .stNumberInput>div>div>input { background-color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar: Model Selection
st.sidebar.header("🔧 Configuration")
model_choice = st.sidebar.radio("Select model:", ["Standard (XGBoost)", "Fair (DP‑constrained)"])
DATA_DIR    = "data"
RESULTS_DIR = "results"

# Map to file names
if model_choice == "Standard (XGBoost)":
    model_path    = os.path.join(RESULTS_DIR, "xgboost_model.pkl")
    supports_proba = True
    supports_shap  = True
else:
    model_path    = os.path.join(RESULTS_DIR, "final_model.pkl")
    supports_proba = False
    supports_shap  = False

scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")

# Load artifacts
@st.cache_resource
def load_model_and_scaler():
    m = joblib.load(model_path)
    s = joblib.load(scaler_path)
    return m, s

model, scaler = load_model_and_scaler()

# Title & Intro
st.title("💓 Cardiovascular Disease Risk Prediction")
st.write(f"**Using Best Model:** {model_choice}")

# User Input Form 
with st.expander("👤 Patient Basic Information", expanded=True):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 20, 100, 50)
    sex = c2.selectbox("Sex", ("Male", "Female"))

with st.expander("❤️ Clinical Measurements"):
    c1, c2, c3 = st.columns(3)
    cp       = c1.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = c2.number_input("Rest BP (mm Hg)", 80, 200, 120)
    chol     = c3.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    c4, c5, c6 = st.columns(3)
    fbs    = c4.selectbox("FBS > 120 mg/dl", [0, 1])
    restecg= c5.selectbox("Rest ECG", [0, 1, 2])
    thalach= c6.number_input("Max HR Achieved", 60, 250, 150)

    c7, c8, c9 = st.columns(3)
    exang   = c7.selectbox("Exercise Angina", [0, 1])
    oldpeak = c8.number_input("ST Depressoin (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope   = c9.selectbox("Slope ST Segment", [0, 1, 2])

    c10, c11, c12 = st.columns(3)
    ca      = c10.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
    thal    = c11.selectbox("Thalassemia", [0, 1, 2, 3])
    st.write("")  # spacer

# collect into DataFrame
user_input = pd.DataFrame([{
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])
def plot_risk_gauge(prob):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.axis("off")
    # background bar
    ax.barh(0, 1, color="#44475a", height=0.3)
    # risk level bar
    ax.barh(0, prob, color="#ff5555", height=0.3)
    # floating label
    x = prob if prob > 0.08 else 0.08
    ax.text(x, 0.45, f"{prob*100:.1f}%", ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#ff5555')
    ax.set_xlim(0, 1); ax.set_ylim(-0.2, 0.8)
    st.pyplot(fig)
# ── Prediction ───────────────────────────────────────────────────────
if st.button("🔍 Predict Risk"):
    # scale & predict
    X_scaled = scaler.transform(user_input)
    pred     = model.predict(X_scaled)[0]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("🏷️ Prediction", "High Risk" if pred == 1 else "Low Risk")
    if supports_proba and hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][1]
        col2.metric("📊 Confidence", f"{prob:.1%}")
        # gauge
        st.subheader("🔴Risk Gauge")
        st.progress(int(prob * 100))
    else:
        col2.metric("📊 Confidence", "N/A")
    col3.metric("⚙️ Model Used", model_choice)

    # celebrate
    st.toast("🎉 Prediction Complete!", icon="✅")
    celebrations = ['balloons', 'snow']
    choice = random.choice(celebrations)
    if choice == 'balloons':
        st.balloons()
    elif choice == 'snow':
        st.snow()
        
    # SHAP explanation
    # with st.expander("🔎 Model Explanation (SHAP)"):
    #     if supports_shap:
    #         explainer   = shap.TreeExplainer(model)
    #         shap_values = explainer.shap_values(X_scaled)
    #         fig, ax = plt.subplots()
    #         shap.summary_plot(
    #             shap_values,
    #             X_scaled,
    #             feature_names=user_input.columns,
    #             show=False
    #         )
    #         st.pyplot(fig)
    #     else:
    #         st.warning("⚠️ SHAP explanation not available for this model.")

