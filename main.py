import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# --- Auth Logic ---
import pandas as pd
import streamlit as st
import os

st.set_page_config(page_title="Parkinson's Disease Detection", layout="centered")


# Load users from CSV
def load_users():
    if os.path.exists("users.csv"):
        return pd.read_csv("users.csv")
    return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    users = load_users()
    users = users.append({"username": username, "password": password}, ignore_index=True)
    users.to_csv("users.csv", index=False)

users_df = load_users()

# Authentication block
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

auth_mode = st.sidebar.radio("🔐 Choose Action", ["Login", "Sign Up"])

if not st.session_state.authenticated:
    st.title("🔒 Authentication")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_mode == "Login":
        if st.button("Login"):
            if ((users_df["username"] == username) & (users_df["password"] == password)).any():
                st.session_state.authenticated = True
                st.success("✅ Logged in successfully!")
            else:
                st.error("❌ Invalid credentials.")

    elif auth_mode == "Sign Up":
        if st.button("Create Account"):
            if username in users_df["username"].values:
                st.warning("⚠️ Username already exists.")
            else:
                save_user(username, password)
                st.success("🎉 Account created! You can now log in.")

    st.stop()

# --- Load model (Make sure model.pkl exists) ---
model = joblib.load("model.pkl")

# --- Load animation ---
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- UI Styling ---

st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            color: #2c3e50;
        }
        .subtitle {
            color: #34495e;
            font-size: 18px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# --- Header with animation ---
lottie_animation = load_lottie("https://assets10.lottiefiles.com/packages/lf20_tutvdkg0.json")

st.markdown("<h1 class='title'>🧠 Parkinson's Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload patient clinical data to predict Parkinson's likelihood.</p>", unsafe_allow_html=True)

# --- Upload + Prediction UI ---
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("🧪 Predict PD Diagnosis"):
        prediction = model.predict(data)
        data['Prediction'] = prediction

        st.success("✅ Prediction complete!")
        st.subheader("📋 Results")
        st.dataframe(data)

        # Summary chart
        chart_data = data['Prediction'].value_counts().reset_index()
        chart_data.columns = ['Diagnosis', 'Count']
        chart_data['Diagnosis'] = chart_data['Diagnosis'].replace({1: "Parkinson's", 0: "Healthy"})

        fig = px.bar(chart_data, x='Diagnosis', y='Count', color='Diagnosis',
                     color_discrete_map={"Parkinson's": '#e74c3c', 'Healthy': '#2ecc71'},
                     title="🧾 Diagnosis Summary")
        st.plotly_chart(fig)

        # Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

else:
    st.info("📤 Upload a CSV file containing clinical features.")

st.markdown("</div>", unsafe_allow_html=True)
