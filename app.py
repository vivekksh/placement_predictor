import streamlit as st
import numpy as np
from joblib import load
import base64

# Debug prints to trace execution
print("Starting app...")

# Load the trained model pipeline with error handling
try:
    print("Loading model pipeline...")
    pipeline = load('placement_model_pipeline.joblib')
    print("Model pipeline loaded successfully.")
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# Custom function to add background image (optional)
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Optional: Uncomment and add a local image (e.g., NIT Silchar logo) for background
# set_background("nit_silchar_logo.png")  # Place the image in the same directory

# Page configuration
st.set_page_config(page_title="NIT Silchar Placement Predictor", layout="centered", initial_sidebar_state="auto")

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4f8; /* Light background */
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #8B0000; /* NIT Silchar Maroon */
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px #FFD700; /* Gold shadow */
    }
    .section {
        background-color: rgba(255, 215, 0, 0.1); /* Light gold tint */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #FFD700; /* NIT Silchar Gold */
        color: #8B0000; /* Maroon text */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFC107; /* Lighter gold on hover */
        transform: scale(1.05);
    }
    .stSlider, .stSelectbox {
        margin: 10px 0;
    }
    .result {
        font-size: 1.2em;
        font-weight: bold;
        padding: 10px;
        border-left: 5px solid #8B0000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">NIT Silchar Placement Success Predictor</div>', unsafe_allow_html=True)

# Main section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.write("Welcome! Enter your details below to predict your placement probability during NIT Silchar's placement drive.")

# Input fields with improved layout
with st.expander("📋 Personal & Academic Details"):
    cgpa = st.slider("CGPA (6.0 - 10.0)", 6.0, 10.0, 8.0, help="Your cumulative grade point average.")
    internship = st.selectbox("Have you completed an internship? (0 = No, 1 = Yes)", [0, 1])
    coding_score = st.slider("Coding Test Score (0 - 100)", 0, 100, 50, help="Score from coding assessments.")
    referral = st.selectbox("Do you have an alumni referral? (0 = No, 1 = Yes)", [0, 1])

with st.expander("🎤 Interview & Skills"):
    interview_score = st.slider("Interview Score (0 - 100)", 0, 100, 50, help="Score from mock interviews.")
    communication_skills = st.slider("Communication Skills (0 - 5)", 0, 5, 3, help="Rating of your soft skills.")
    project_experience = st.slider("Project Experience (0 - 3)", 0, 3, 1, help="Number of significant projects.")

with st.expander("🏫 Department & Timing"):
    department = st.selectbox("Department", ['CSE', 'ECE', 'ME', 'CE', 'EE'])
    placement_season = st.selectbox("Placement Season", ['7th Sem', '8th Sem', 'Pre-Final'])

# Prediction button with loading state
if st.button("Predict Placement Probability"):
    with st.spinner("Calculating your placement probability..."):
        input_data = np.array([[cgpa, internship, coding_score, referral, interview_score,
                                communication_skills, project_experience, department, placement_season]])
        try:
            prediction = pipeline.predict_proba(input_data)
            placement_probability = prediction[0][1]
            st.markdown(f'<div class="result">**Placement Probability: {placement_probability:.2f}**</div>', unsafe_allow_html=True)
            if placement_probability > 0.5:
                st.success("**Recommendation:** 🎉 Likely to be placed! Focus on interview preparation.")
            else:
                st.warning("**Recommendation:** 🚀 Work on improving CGPA, internships, coding skills, or seek an alumni referral.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer
st.markdown("</div>", unsafe_allow_html=True)  # Close section div
st.markdown(
    """
    <div style='text-align: center; color: #8B0000; font-size: 0.9em; margin-top: 20px;'>
        Developed for NIT Silchar by Vivek Sharma | © 2025
    </div>
    """,
    unsafe_allow_html=True
)

# Optional: Add a logo or image (uncomment and adjust path)
# st.image("nit_silchar_logo.png", width=200, caption="NIT Silchar")