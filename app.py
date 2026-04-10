import streamlit as st
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Predictor AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ANIMATION LOADER ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a medical-themed animation (you can change the URL to any Lottie file you like)
lottie_health = load_lottieurl("https://lottie.host/8b725c1b-2615-4ba3-ab45-b467cb4adcd5/n9jN6o9A1u.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Make sure 'model (1).pkl' is in the same folder as this script
    with open('model (1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- MAIN APP UI ---
st.title("🏥 Diabetes Risk Assessment Tool")
st.markdown("""
Welcome to the interactive health predictor! Adjust the medical parameters below, 
and our K-Nearest Neighbors machine learning model will assess the risk profile.
""")

# Display Animation
if lottie_health:
    st_lottie(lottie_health, height=200, key="health_animation")

st.divider()

# --- INPUT SECTION ---
st.header("Patient Vitals & Information")

# Using columns for a neat, attractive layout
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

with col2:
    glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=120, step=1)
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900, value=80, step=1)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, step=1)

with col3:
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)

# --- PREDICTION LOGIC ---
st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
center_button = st.columns([1, 1, 1])[1] # Center the button

with center_button:
    if st.button("🔮 Generate Prediction", use_container_width=True):
        
        # Structure the input exactly as the model expects
        input_data = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': pedigree,
            'Age': age
        }])
        
        with st.spinner("Analyzing data..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
        st.divider
