# Patient Readmission Prediction Dashboard (Dash)

## Overview

This Dash application provides an interactive web interface for the AI Readmission Predictor project. It allows users to:

- Run the complete training pipeline (data generation, preprocessing, model training)
- Make real-time predictions on new patient data
- View model performance metrics and visualizations (confusion matrix, ROC curves, feature importance)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The app will be available at `http://127.0.0.1:8050/`

## Features

- **Model Training**: One-click execution of the full ML pipeline
- **Interactive Predictions**: Input patient information to get readmission risk assessment
- **Performance Visualization**: View confusion matrices, ROC curves, and feature importance plots
- **Real-time Results**: Instant predictions with probability scores and comorbidity analysis

## Technologies Used

- **Dash**: Web framework for building interactive dashboards
- **Plotly**: Data visualization library
- **scikit-learn**: Machine learning models and evaluation
- **pandas**: Data manipulation
- **joblib**: Model serialization

## Related Project

This dashboard is part of the [AI Readmission Predictor](../Part%202/) project. Ensure the main project is set up first for full functionality.
