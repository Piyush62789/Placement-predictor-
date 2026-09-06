import os
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Placement Predictor")
st.write("Predict whether a student will be placed based on **CGPA** and **IQ** using Logistic Regression.")

model_file = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_file = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("Model files not found! Please run `python train.py` first.")
else:
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.number_input("CGPA", min_value=1.0, max_value=10.0, value=7.5, step=0.1)
    with col2:
        iq = st.number_input("IQ Score", min_value=40.0, max_value=250.0, value=110.0, step=1.0)

    if st.button("Predict Placement"):
        input_df = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])
        input_scaled = scaler.transform(input_df)
        prediction = clf.predict(input_scaled)[0]
        prob = clf.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.success(f"🎉 **Placed!** (Probability: {prob * 100:.1f}%)")
        else:
            st.error(f"❌ **Not Placed** (Placement Probability: {prob * 100:.1f}%)")
