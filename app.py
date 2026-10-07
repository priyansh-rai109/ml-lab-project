import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

model = joblib.load("crop_model.pkl")

EMOJI = {
    "rice": "🌾", "maize": "🌽", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "coconut": "🥥", "coffee": "☕", "cotton": "☁️",
    "jute": "🌿", "papaya": "🧡", "pomegranate": "🔴",
}

with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** Random Forest")
    st.write("**Test accuracy:** 99.5%")
    st.write("**Crops:** 22 | **Features:** 7")
    st.caption("ML Lab Project")

st.title("🌾 Crop Recommendation System")
st.caption("Mitti aur mausam ki details daalo, model sahi fasal batayega.")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("🧪 Mitti ke nutrients")
    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 5, 145, 50)
    K = st.slider("Potassium (K)", 5, 205, 50)
    ph = st.slider("pH", 3.5, 9.9, 6.5, step=0.1)

with right:
    st.subheader("🌦️ Mausam")
    temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, step=0.5)
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0, step=1.0)
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=1.0)

if st.button("🌱 Fasal batao", type="primary"):
    X_new = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    proba = model.predict_proba(X_new)[0]
    top3 = proba.argsort()[::-1][:3]

    best = model.classes_[top3[0]]
    st.success(f"{EMOJI.get(best, '🌱')} Best choice: **{best.title()}**")

    st.subheader("Top 3 recommendations")
    cols = st.columns(3)
    for col, i in zip(cols, top3):
        name = model.classes_[i]
        col.metric(f"{EMOJI.get(name, '🌱')} {name.title()}", f"{proba[i] * 100:.0f}%")
        col.progress(float(proba[i]))