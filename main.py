import streamlit as st
import pandas as pd
import joblib
import urllib.request
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import os
from io import BytesIO
from fpdf import FPDF

from utils.preprocessing import preprocess_data
from utils.feature_engineering import extract_features
from utils.predict import make_prediction
from utils.training import train_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Parkinson's Disease Detection", layout="centered")

# --- Load users from CSV ---
def load_users():
    if os.path.exists("users.csv"):
        return pd.read_csv("users.csv")
    return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    users = load_users()
    users = users.append({"username": username, "password": password}, ignore_index=True)
    users.to_csv("users.csv", index=False)

users_df = load_users()

# --- Authentication ---
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
model_option = st.selectbox("Choose Model for Prediction", ["Default Model", "User Trained Model"])
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("🧪 Predict PD Diagnosis"):
        preprocessed = preprocess_data(data)
        features = extract_features(preprocessed)

        model_path = "model/user_trained_model.pkl"
        MODEL_URL = "https://raw.githubusercontent.com/ABUBAKAR1884/PD-Detection/main/model/user_trained_model.pkl"
        model = joblib.load(model_path)
        features = extract_features(preprocessed)

        # Align feature columns
        features = features[model.feature_names_in_]

        prediction = model.predict(features)



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

        # Download CSV option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

        # --- Model Evaluation Section ---
        st.subheader("📊 Model Evaluation Metrics")
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(features)[:, 1] > 0.5
        else:
            y_proba = prediction

        report = classification_report(prediction, y_proba, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        cm = confusion_matrix(prediction, y_proba)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # --- PDF Report Download ---
        st.subheader("📄 Generate PDF Report")
        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Parkinson's Disease Prediction Report", ln=True, align='C')
            pdf.ln(10)
            pdf.multi_cell(0, 10, txt=f"Model Used: {model_option}\nTotal Records: {len(data)}\nPredicted Parkinson's: {sum(data['Prediction'] == 1)}\nPredicted Healthy: {sum(data['Prediction'] == 0)}")
            
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_output.getvalue(),
                file_name="parkinsons_report.pdf",
                mime="application/pdf"
            )

else:
    st.info("📤 Upload a CSV file containing clinical features.")

# --- Train Your Own Model Section ---
st.markdown("---")
st.header("🧠 Train Your Own Model")

train_file = st.file_uploader("📁 Upload labeled data (CSV with target column)", key="train")
model_type = st.selectbox("Select Model Type", ["Random Forest", "Logistic Regression", "Neural Network"])
target_column = st.text_input("Enter the name of the target column (e.g., 'target')")

if train_file and target_column:
    train_data = pd.read_csv(train_file)
    st.write("📄 Training Data Preview:", train_data.head())

    if st.button("🚀 Train Model"):
        try:
            model, acc, f1 = train_model(train_data, target_column, model_type)
            st.success(f"✅ Model trained successfully!\nAccuracy: {acc:.2f}\nF1 Score: {f1:.2f}")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")

st.markdown("</div>", unsafe_allow_html=True)
