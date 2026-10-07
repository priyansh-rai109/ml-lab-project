import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

model = joblib.load("crop_model.pkl")

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
    pred = model.predict(X_new)[0]
    st.success(f"Recommended crop: **{pred.title()}**")