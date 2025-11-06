# AI Readmission Predictor

## Project Overview

This project implements a machine learning system to predict patient readmission risk. It serves as a proof-of-concept for an end-to-end data science pipeline, from data preprocessing to model training and evaluation.

## Goals

- Demonstrate a complete machine learning workflow for healthcare prediction
- Build a predictive model using Logistic Regression for interpretability
- Evaluate model performance using appropriate metrics (precision, recall, confusion matrix)
- Create a reproducible pipeline for data preprocessing and model training

## Project Structure

```
/ai-readmission-predictor
├── README.md                 # Project documentation
├── data/
│   └── synthetic_patient_data.csv  # Synthetic patient dataset
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # Data loading, cleaning, and feature engineering
│   └── 02_model_training.ipynb      # Model training, evaluation, and optimization
├── src/
│   ├── readmission_model.joblib     # Saved trained model (generated after training)
│   └── scaler.joblib                # Saved scaler for preprocessing (generated after training)
├── generate_data.py                  # Script to generate synthetic patient data
└── requirements.txt                  # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

First, run the data generation script to create the synthetic patient dataset:

```bash
python generate_data.py
```

This will create `data/synthetic_patient_data.csv` with 1000+ patient records.

### 3. Run the Notebooks

Open and run the notebooks in order:

1. **01_data_preprocessing.ipynb**: 
   - Loads and explores synthetic patient data
   - Handles missing values (introduces and imputes)
   - Performs feature engineering (creates comorbidity score)
   - Encodes categorical variables using one-hot encoding
   - Splits data into training/test sets (80/20)
   - **Visualizations**: Target variable distribution, age distributions, length of stay box plots, categorical analysis, correlation heatmap, missing values before/after, comorbidity score distributions

2. **02_model_training.ipynb**: 
   - Trains initial Logistic Regression model with default parameters
   - Evaluates model performance using multiple metrics
   - Optimizes model with L2 regularization (C=0.1)
   - Compares initial vs optimized models
   - Saves trained model and scaler for deployment
   - **Visualizations**: Confusion matrix heatmaps, ROC curves, Precision-Recall curves, model comparison bar charts, feature importance plots, training vs test accuracy comparison

### 4. Model Usage

After training, the model and scaler are saved to `src/` and can be loaded for predictions:

```python
import joblib
import pandas as pd

# Load the saved model and scaler
model = joblib.load('src/readmission_model.joblib')
scaler = joblib.load('src/scaler.joblib')

# Preprocess new patient data (apply same preprocessing as training)
# ... perform feature engineering and encoding ...

# Scale the features
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)[:, 1]  # Probability of readmission
```

## Key Features

- **Synthetic Data Generation**: Creates realistic patient data with correlated features (1000+ records)
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and creates engineered features (comorbidity score)
- **Comprehensive Visualizations**: Includes data exploration plots, correlation heatmaps, missing value analysis, and model performance visualizations
- **Model Training**: Uses Logistic Regression with L2 regularization for interpretability
- **Model Evaluation**: Comprehensive metrics including confusion matrix, precision, recall, ROC curves, and Precision-Recall curves
- **Model Comparison**: Visual comparison between initial and optimized models
- **Feature Importance**: Identifies top features contributing to readmission risk
- **Model Persistence**: Saves trained model and scaler for deployment using joblib

## Technologies Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models and evaluation metrics
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization for deployment
- **jupyter**: Interactive notebook environment

## Visualization Features

### Data Preprocessing Notebook (`01_data_preprocessing.ipynb`)
- **Target Variable Distribution**: Pie chart showing readmission vs non-readmission rates
- **Age Distribution**: Histogram comparing age distributions by readmission status
- **Length of Stay Analysis**: Box plots showing stay duration by readmission status
- **Categorical Variable Analysis**: Bar charts showing readmission rates by gender
- **Correlation Heatmap**: Visual correlation matrix of numerical features
- **Missing Values Visualization**: Before/after imputation comparison charts
- **Comorbidity Score Analysis**: Histograms and box plots showing score distributions

### Model Training Notebook (`02_model_training.ipynb`)
- **Confusion Matrix Heatmaps**: Visual representation of model predictions for both initial and optimized models
- **ROC Curves**: Receiver Operating Characteristic curves with AUC scores for model comparison
- **Precision-Recall Curves**: Precision-recall plots showing model performance at different thresholds
- **Model Comparison Bar Charts**: Side-by-side comparison of Accuracy, Precision, Recall, and F1-Score
- **Feature Importance Plot**: Top 10 features ranked by their logistic regression coefficients
- **Training vs Test Accuracy**: Bar chart showing overfitting analysis with gap annotations

## Notes

This is a proof-of-concept project for educational purposes. In a real-world scenario, you would:
- Use real patient data with proper privacy considerations (HIPAA compliance)
- Perform more extensive feature engineering and domain-specific feature creation
- Try multiple algorithms (Random Forest, XGBoost, Neural Networks) and ensemble methods
- Conduct more thorough hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Implement proper data validation, monitoring, and model versioning
- Add cross-validation for more robust performance evaluation
- Implement feature selection techniques to reduce dimensionality

