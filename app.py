import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

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

# Page configuration
st.set_page_config(page_title="NIT Silchar Placement Predictor", layout="centered", initial_sidebar_state="auto")

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f4f8;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #8B0000;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px #FFD700;
    }
    .section {
        background-color: rgba(255, 215, 0, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #FFD700;
        color: #8B0000;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFC107;
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
    .badge {
        font-size: 1.1em;
        padding: 5px 10px;
        border-radius: 5px;
        color: white;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">NIT Silchar Placement Success Predictor</div>', unsafe_allow_html=True)

# Main section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.write("Welcome! Explore your placement potential with this interactive tool designed for NIT Silchar students.")

# Input fields with real-time tips
with st.expander("📋 Personal & Academic Details"):
    cgpa = st.slider("CGPA (6.0 - 10.0)", 6.0, 10.0, 8.0, help="Your cumulative grade point average.")
    if cgpa > 8.0:
        st.info("Great CGPA! Consider adding an internship.")
    internship = st.selectbox("Have you completed an internship? (0 = No, 1 = Yes)", [0, 1])
    coding_score = st.slider("Coding Test Score (0 - 100)", 0, 100, 50, help="Score from coding assessments.")
    if coding_score < 70:
        st.warning("Boost your coding skills for better chances!")
    referral = st.selectbox("Do you have an alumni referral? (0 = No, 1 = Yes)", [0, 1])

with st.expander("🎤 Interview & Skills"):
    interview_score = st.slider("Interview Score (0 - 100)", 0, 100, 50, help="Score from mock interviews.")
    communication_skills = st.slider("Communication Skills (0 - 5)", 0, 5, 3, help="Rating of your soft skills.")
    project_experience = st.slider("Project Experience (0 - 3)", 0, 3, 1, help="Number of significant projects.")
    if project_experience < 2:
        st.info("Add more projects to strengthen your profile.")

with st.expander("🏫 Department & Timing"):
    department = st.selectbox("Department", ['CSE', 'ECE', 'ME', 'CE', 'EE'])
    placement_season = st.selectbox("Placement Season", ['7th Sem', '8th Sem', 'Pre-Final'])

# Calculate Placement Score for gamification
def calculate_placement_score(cgpa, internship, coding_score, referral, interview_score,
                             communication_skills, project_experience):
    return (cgpa / 10 * 25 + internship * 15 + coding_score / 100 * 15 + referral * 10 +
            interview_score / 100 * 15 + communication_skills / 5 * 10 + project_experience * 10)

placement_score = calculate_placement_score(cgpa, internship, coding_score, referral, interview_score,
                                          communication_skills, project_experience)
badge = "Placement Pro" if placement_score > 75 else "Rising Star" if placement_score > 50 else "Rookie"

# Prediction button with loading state and "What If" tool
if st.button("Predict Placement Probability"):
    with st.spinner("Calculating your placement potential..."):
        # Create a DataFrame with column names matching the training data
        input_data = pd.DataFrame({
            'CGPA': [cgpa],
            'Internships': [internship],
            'Coding_Score': [coding_score],
            'Alumni_Referral': [referral],
            'Interview_Score': [interview_score],
            'Communication_Skills': [communication_skills],
            'Project_Experience': [project_experience],
            'Department': [department],
            'Placement_Season': [placement_season]
        })
        try:
            prediction = pipeline.predict_proba(input_data)
            placement_probability = prediction[0][1]
            st.markdown(f'<div class="result">**Placement Probability: {placement_probability:.2f}**</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="badge" style="background-color: {"#4CAF50" if placement_probability > 0.5 else "#FF9800"};">{badge} (Score: {placement_score:.1f}/100)</div>', unsafe_allow_html=True)
            if placement_probability > 0.5:
                st.success("🎉 **Recommendation:** Likely to be placed! Focus on interview preparation.")
            else:
                st.warning("🚀 **Recommendation:** Work on improving CGPA, internships, coding skills, or seek an alumni referral.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # What If Tool
    st.subheader("🔍 What If? Explore Scenarios")
    with st.expander("Adjust one factor to see the impact"):
        what_if_param = st.selectbox("Change Parameter", ['CGPA', 'Coding_Score', 'Interview_Score'])
        what_if_value = st.slider(f"{what_if_param} (Adjusted)", 0.0 if what_if_param == 'CGPA' else 0,
                                  10.0 if what_if_param == 'CGPA' else 100,
                                  cgpa if what_if_param == 'CGPA' else coding_score if what_if_param == 'Coding_Score' else interview_score)
        new_input = input_data.copy()
        if what_if_param == 'CGPA':
            new_input['CGPA'] = what_if_value
        elif what_if_param == 'Coding_Score':
            new_input['Coding_Score'] = what_if_value
        elif what_if_param == 'Interview_Score':
            new_input['Interview_Score'] = what_if_value
        new_prediction = pipeline.predict_proba(new_input)
        new_probability = new_prediction[0][1]
        st.write(f"New Placement Probability with {what_if_param} = {what_if_value}: **{new_probability:.2f}**")

# Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #8B0000; font-size: 0.9em; margin-top: 20px;'>
        Developed for NIT Silchar by [Your Name] | © 2025 | <a href="https://www.nits.ac.in/" target="_blank">NIT Silchar</a>
    </div>
    """,
    unsafe_allow_html=True
)