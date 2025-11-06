"""
Streamlit App for Patient Readmission Risk Prediction

This app loads the trained model and provides a web interface for predicting
patient readmission risk based on input features.

Author: PETER KAMAU MWAURA
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = Path('src/readmission_model.joblib')
    scaler_path = Path('src/scaler.joblib')

    if not model_path.exists() or not scaler_path.exists():
        st.error("Model or scaler not found. Please run the pipeline first to train and save the model.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def main():
    st.set_page_config(
        page_title="Patient Readmission Predictor",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Patient Readmission Risk Predictor")
    st.markdown("""
    This application predicts the risk of patient readmission within 30 days of discharge
    using a machine learning model trained on synthetic patient data.
    """)

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Information")

        # Input fields
        age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1)

        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])

        length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=100, value=5, step=1)

        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=20, step=1)

        num_medications = st.number_input("Number of Medications", min_value=1, max_value=50, value=8, step=1)

        diagnosis_code = st.selectbox("Diagnosis Code", ["I10", "E11", "J44", "F32", "M79", "N18", "K21", "I50"])

    with col2:
        st.subheader("Prediction Results")

        # Prediction button
        if st.button("Predict Readmission Risk", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'admission_type': [admission_type],
                'length_of_stay': [length_of_stay],
                'num_lab_procedures': [num_lab_procedures],
                'num_medications': [num_medications],
                'diagnosis_code': [diagnosis_code]
            })

            # Calculate comorbidity score
            score = 0

            # Age factor
            if age > 75:
                score += 2
            elif age > 65:
                score += 1
            elif age > 50:
                score += 0.5

            # Length of stay factor
            if length_of_stay > 14:
                score += 2
            elif length_of_stay > 7:
                score += 1

            # Medication count factor
            if num_medications > 15:
                score += 1.5
            elif num_medications > 10:
                score += 0.5

            # Diagnosis code factor
            high_complexity_codes = ['I10', 'E11', 'J44', 'N18']
            if diagnosis_code in high_complexity_codes:
                score += 1

            # Add some randomness (using fixed seed for consistency)
            np.random.seed(42)
            score += np.random.uniform(-0.5, 0.5)

            # Ensure score is between 0 and 5
            comorbidity_score = max(0, min(5, score))

            input_data['comorbidity_score'] = comorbidity_score

            # One-hot encode categorical variables
            categorical_cols = ['gender', 'admission_type', 'diagnosis_code']
            input_encoded = pd.get_dummies(input_data, columns=categorical_cols, prefix=categorical_cols, drop_first=False)

            # Ensure all expected columns are present (add missing columns with 0)
            expected_columns = [
                'age', 'length_of_stay', 'num_lab_procedures', 'num_medications', 'comorbidity_score',
                'gender_Female', 'gender_Male', 'gender_Other',
                'admission_type_Elective', 'admission_type_Emergency', 'admission_type_Urgent',
                'diagnosis_code_E11', 'diagnosis_code_F32', 'diagnosis_code_I10', 'diagnosis_code_I50',
                'diagnosis_code_J44', 'diagnosis_code_K21', 'diagnosis_code_M79', 'diagnosis_code_N18'
            ]

            for col in expected_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            # Reorder columns to match training data
            input_encoded = input_encoded[expected_columns]

            # Scale the features
            input_scaled = scaler.transform(input_encoded)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            # Display results
            if prediction == 1:
                st.error(f"⚠️ **High Risk of Readmission**")
                st.markdown(f"**Probability:** {probability:.1%}")
                st.markdown("""
                **Recommendations:**
                - Schedule follow-up appointment within 7 days
                - Consider home care services
                - Review medication regimen
                - Monitor for complications
                """)
            else:
                st.success(f"✅ **Low Risk of Readmission**")
                st.markdown(f"**Probability:** {probability:.1%}")
                st.markdown("""
                **Recommendations:**
                - Standard discharge follow-up
                - Routine post-discharge care
                """)

            # Show comorbidity score
            st.info(f"**Comorbidity Score:** {comorbidity_score:.2f} (0-5 scale, higher = more complex)")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Model Information:**
    - Algorithm: Logistic Regression with L2 regularization
    - Training Data: 1000 synthetic patient records
    - Features: Demographics, admission details, medical history
    - Target: Readmission within 30 days
    """)

if __name__ == "__main__":
    main()
