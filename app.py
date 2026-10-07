import streamlit as st
import joblib
import pandas as pd

model = joblib.load("crop_model.pkl")

st.title("Crop Recommendation System")
st.write("Mitti aur mausam ki details daalo, model sahi fasal batayega.")

N = st.number_input("Nitrogen (N)", 0, 140, 50)
P = st.number_input("Phosphorus (P)", 5, 145, 50)
K = st.number_input("Potassium (K)", 5, 205, 50)
temperature = st.number_input("Temperature (°C)", 8.0, 44.0, 25.0)
humidity = st.number_input("Humidity (%)", 14.0, 100.0, 70.0)
ph = st.number_input("pH", 3.5, 9.9, 6.5)
rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("Fasal batao"):
    X_new = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    pred = model.predict(X_new)[0]
    st.success(f"Recommended crop: {pred}")