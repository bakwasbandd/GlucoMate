import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

st.set_page_config(layout="wide")
st.title("🩺 GlucoMate: Diabetes Prediction App")
st.markdown("An AI-powered assistant to help you assess your risk of diabetes based on key medical inputs.")

left, right = st.columns(2)

with left:
    st.header("🔢 Input Medical Information")

    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.slider("Age", 10, 100, 30)

with right:
    st.header("📊 Prediction Result")

    if st.button("Predict Diabetes Risk"):
        # Prepare input
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction]

        if prediction == 1:
            st.error("⚠️ The model predicts you **may have diabetes** !!")
        else:
            st.success("✅ The model predicts you **do not have diabetes** !!")

        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.markdown("---")
        st.caption("🔍 Always consult a medical professional for accurate diagnosis.")

