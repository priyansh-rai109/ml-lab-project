import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="KisanMitra", page_icon="🌾", layout="wide")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🔴", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🧡", "coconut": "🥥", "cotton": "☁️",
    "jute": "🌿", "coffee": "☕",
}

# Har crop ka typical profile (dataset ke averages ke aas-paas)
PRESETS = {
    "🌾 Rice": dict(N=80, P=48, K=40, ph=6.4, temperature=23.5, humidity=82.0, rainfall=236.0),
    "🌽 Maize": dict(N=78, P=48, K=20, ph=6.2, temperature=22.5, humidity=65.0, rainfall=85.0),
    "🫘 Chickpea": dict(N=40, P=68, K=80, ph=7.3, temperature=19.0, humidity=17.0, rainfall=80.0),
    "☁️ Cotton": dict(N=118, P=46, K=20, ph=6.9, temperature=24.0, humidity=80.0, rainfall=80.0),
    "☕ Coffee": dict(N=101, P=28, K=30, ph=6.8, temperature=25.5, humidity=58.0, rainfall=158.0),
    "🍎 Apple": dict(N=21, P=134, K=200, ph=5.9, temperature=22.5, humidity=92.0, rainfall=113.0),
}

DEFAULTS = dict(N=50, P=50, K=50, ph=6.5, temperature=25.0, humidity=70.0, rainfall=100.0)
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def apply_preset(name):
    for key, val in PRESETS[name].items():
        st.session_state[key] = val


@st.cache_resource
def load_model():
    return joblib.load("crop_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")


model = load_model()
df = load_data()

# ---------- Custom CSS ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {
  background: linear-gradient(120deg, #0f5132 0%, #1b7a4a 55%, #c9a227 130%);
  border-radius: 22px; padding: 34px 40px; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.hero h1 {margin: 0; font-size: 2.6rem; color: #fff; letter-spacing: 0.5px;}
.hero p {margin: 6px 0 0 0; color: #e6f4ea; font-size: 1.05rem;}
.pill {
  display: inline-block; background: rgba(255,255,255,0.16); color: #fff;
  border-radius: 999px; padding: 4px 14px; margin: 14px 8px 0 0; font-size: 0.85rem;
}
.result {
  display: flex; align-items: center; gap: 24px;
  background: linear-gradient(135deg, #12301f, #1b4d33);
  border: 1px solid #c9a227; border-radius: 20px; padding: 24px 30px; margin: 18px 0;
  box-shadow: 0 0 25px rgba(201,162,39,0.25);
}
.result .big {font-size: 4.5rem; line-height: 1;}
.result .small {color: #c9a227; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase;}
.result .name {font-size: 2.4rem; font-weight: 800; color: #fff;}
.result .conf {color: #8be0b3; font-size: 1.05rem;}
.bar-row {margin: 10px 0;}
.bar-label {display: flex; justify-content: space-between; color: #e8f5ec; margin-bottom: 4px;}
.bar-track {background: #12301f; border-radius: 10px; height: 12px; overflow: hidden;}
.bar-fill {height: 12px; border-radius: 10px; background: linear-gradient(90deg, #34d399, #c9a227);}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown("""
<div class="hero">
<h1>🌾 KisanMitra</h1>
<p>Mitti aur mausam batao, sahi fasal ka faisla hum karenge.</p>
<span class="pill">Random Forest</span>
<span class="pill">22 fasalein</span>
<span class="pill">99.5% test accuracy</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Fasal Chuno", "📊 Model Lab", "🔍 Data Explorer"])

# ================= TAB 1 =================
with tab1:
    st.markdown("**⚡ Quick presets** (ek click mein sliders set)")
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("🧪 Mitti ke nutrients")
        st.slider("Nitrogen (N)", 0, 140, key="N")
        st.slider("Phosphorus (P)", 5, 145, key="P")
        st.slider("Potassium (K)", 5, 205, key="K")
        st.slider("pH", 3.5, 9.9, step=0.1, key="ph")
    with right:
        st.subheader("🌦️ Mausam")
        st.slider("Temperature (°C)", 8.0, 44.0, step=0.5, key="temperature")
        st.slider("Humidity (%)", 14.0, 100.0, step=1.0, key="humidity")
        st.slider("Rainfall (mm)", 20.0, 300.0, step=1.0, key="rainfall")

    if st.button("🌱 Fasal batao", type="primary", use_container_width=True):
        values = [st.session_state[f] for f in FEATURES]
        X_new = pd.DataFrame([values], columns=FEATURES)
        proba = model.predict_proba(X_new)[0]
        top3 = proba.argsort()[::-1][:3]
        best = model.classes_[top3[0]]

        result_html = (
            f'<div class="result"><div class="big">{EMOJI.get(best, "🌱")}</div>'
            f'<div><div class="small">Sabse behtar fasal</div>'
            f'<div class="name">{best.title()}</div>'
            f'<div class="conf">{proba[top3[0]] * 100:.0f}% confidence</div></div></div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

        st.markdown("**Top 3 options**")
        for i in top3:
            name = model.classes_[i]
            pct = proba[i] * 100
            row = (
                f'<div class="bar-row"><div class="bar-label">'
                f'<span>{EMOJI.get(name, "🌱")} {name.title()}</span><span>{pct:.0f}%</span></div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{max(pct, 1):.0f}%"></div></div></div>'
            )
            st.markdown(row, unsafe_allow_html=True)

        with st.expander(f"🔬 Tumhara input vs {best.title()} ka average"):
            avg = df[df["label"] == best][FEATURES].mean()
            compare = pd.DataFrame(
                {"Tumhara input": values, f"{best.title()} ka average": avg.values},
                index=FEATURES,
            ).round(1)
            st.dataframe(compare, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("Teen models ka muqabla")
    acc = pd.DataFrame(
        {"Accuracy (%)": [99.55, 97.95, 97.27]},
        index=["Random Forest", "Decision Tree", "Logistic Regression"],
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("🥇 Random Forest", "99.55%")
    c2.metric("🥈 Decision Tree", "97.95%")
    c3.metric("🥉 Logistic Regression", "97.27%")
    st.bar_chart(acc)
    st.caption("80/20 stratified split, random_state=42, test set = 440 rows (har fasal ke 20).")

    st.subheader("Kaunsa factor sabse zyada matter karta hai?")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    st.bar_chart(imp)
    st.info(f"**{imp.index[0]}** aur **{imp.index[1]}** top pe hain. Alag-alag fasalon ko "
            f"baarish aur nami ki bahut alag zaroorat hoti hai, isliye model inpe sabse zyada bharosa karta hai.")

# ================= TAB 3 =================
with tab3:
    st.subheader("Dataset ek nazar mein")
    d1, d2, d3 = st.columns(3)
    d1.metric("Rows", f"{df.shape[0]:,}")
    d2.metric("Features", len(FEATURES))
    d3.metric("Fasalein", df["label"].nunique())

    crop = st.selectbox("Fasal chuno", sorted(df["label"].unique()))
    st.markdown(f"### {EMOJI.get(crop, '🌱')} {crop.title()} ko kya chahiye?")
    summary = df[df["label"] == crop][FEATURES].describe().T[["min", "mean", "max"]].round(1)
    st.dataframe(summary, use_container_width=True)
    st.caption("min/mean/max us fasal ke 100 rows se nikle hain.")