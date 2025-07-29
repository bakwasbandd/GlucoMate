import streamlit as st
import pickle
import numpy as np
from ai_healthtips import generate_health_tips

with open('diabetes_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

st.set_page_config(layout="wide")
st.title("🩺 GlucoMate: Diabetes Prediction App")
st.markdown(
    "An AI-powered assistant to help you assess your risk of diabetes based on key medical inputs.")

st.subheader("")
left, right = st.columns(2)

with left:
    st.markdown(
        """
        <div style="
            background-color: #722F37;
            padding: 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            font-size: 16px;
        ">
        <h3>🔢 Input Medical Information</h3>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function",
                          min_value=0.0, max_value=3.0, value=0.5)
    age = st.slider("Age", 10, 100, 30)

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown(
        """
        <div style="
            background-color: #722F37;
            padding: 20px;
            border-radius: 12px;
            color: white;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            font-size: 16px;
        ">
        <h3>📊 Prediction Result</h3>
        """,
        unsafe_allow_html=True
    )
    st.subheader("")

    if st.button("Predict Diabetes Risk"):
        input_data = np.array(
            [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction]

        # Store in session state
        st.session_state['input_data'] = input_data
        st.session_state['prediction'] = prediction
        st.session_state['confidence'] = confidence

        # Show prediction if available
    if 'prediction' in st.session_state and 'input_data' in st.session_state:
        prediction = st.session_state['prediction']
        input_data = st.session_state['input_data']
        confidence = st.session_state['confidence']

        if prediction == 1:
            st.error("⚠️ The model predicts you **may have diabetes** !!")
        else:
            st.success("✅ The model predicts you **do not have diabetes** !!")

        st.metric("🎯 Confidence", f"{confidence * 100:.2f}%")
        st.markdown("---")

        st.info("💡 Want personalized tips on maintaining healthy glucose levels from our AI assistant using your medical info?")
        if st.button("📌 Show Health Tips"):
            with st.spinner("Generating personalized tips..."):
                tips = generate_health_tips(input_data, prediction)
                st.markdown("### 🧠 AI-Generated Health Tips:")
                st.warning(tips)

        st.markdown("</div>", unsafe_allow_html=True)
