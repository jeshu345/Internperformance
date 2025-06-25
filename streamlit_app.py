import streamlit as st
from model import InternPerformancePredictor

st.set_page_config(page_title="Intern Performance Predictor", layout="centered")

st.title("🎓 Intern Performance Predictor")
st.write("Enter the intern's details to predict their performance level.")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        internships = st.number_input("Internships Completed", min_value=0, max_value=10, value=1)
        soft_skills = st.slider("Soft Skills Score (1-10)", 1, 10, 7)
        satisfaction = st.slider("Career Satisfaction (1-10)", 1, 10, 8)
        high_school_gpa = st.number_input("High School CGPA (1-10)", min_value=1.0, max_value=10.0, value=7.5)
    with col2:
        projects = st.number_input("Projects Completed", min_value=0, max_value=20, value=3)
        networking = st.slider("Networking Score (1-10)", 1, 10, 6)
        work_life_balance = st.slider("Work-Life Balance (1-10)", 1, 10, 7)
        university_gpa = st.number_input("University CGPA (1-10)", min_value=1.0, max_value=10.0, value=7.2)
    
    salary = st.number_input("Expected Starting Salary (₹)", min_value=100000.0, max_value=5000000.0, value=300000.0)

    submitted = st.form_submit_button("🔮 Predict Performance")

# Load model
predictor = InternPerformancePredictor()
if not predictor.load_model():
    predictor.train_model()
    predictor.save_model()

# Prediction
if submitted:
    input_data = {
        'internships': internships,
        'projects': projects,
        'soft_skills': soft_skills,
        'networking': networking,
        'salary': salary,
        'satisfaction': satisfaction,
        'work_life_balance': work_life_balance,
        'high_school_gpa': high_school_gpa,
        'university_gpa': university_gpa
    }

    with st.spinner("Analyzing and predicting..."):
        result = predictor.predict(input_data)

    st.success(f"🎯 Predicted Performance: **{result['prediction']}**")
    st.caption(f"Confidence: {result['confidence']:.1f}%")

    st.subheader("📊 Probability Breakdown")
    for label, prob in result["probabilities"].items():
        st.progress(prob / 100, text=f"{label}: {prob:.1f}%")
