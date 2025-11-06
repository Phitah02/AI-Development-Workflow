# Patient Readmission Risk Predictor (Streamlit)

## Overview

This Streamlit application provides a user-friendly web interface for predicting patient readmission risk. It loads the trained machine learning model from the AI Readmission Predictor project and offers:

- Interactive form for entering patient information
- Real-time risk assessment with probability scores
- Personalized recommendations based on prediction results
- Comorbidity score calculation and display

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser automatically.

## Features

- **Patient Input Form**: Easy-to-use interface for entering demographic and medical information
- **Risk Prediction**: Binary classification with probability estimates
- **Clinical Recommendations**: Tailored suggestions based on risk level
- **Comorbidity Scoring**: Automatic calculation of patient complexity score
- **Model Transparency**: Display of key model information and training details

## Technologies Used

- **Streamlit**: Framework for building data science web apps
- **scikit-learn**: Machine learning models and preprocessing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **joblib**: Model and scaler loading

## Related Project

This application uses the trained model from the [AI Readmission Predictor](../Part%202/) project. The model must be trained and saved in the main project before using this app.
