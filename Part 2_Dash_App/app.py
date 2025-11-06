"""
Dash Application for Readmission Prediction

This app provides an interactive interface to:
1. Run the full training pipeline
2. Make predictions on new patient data
3. View model performance metrics and visualizations

Run with: python app.py
"""

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = "Patient Readmission Predictor"

# Error styles
error_style = {
    'color': '#e74c3c',
    'marginTop': '10px',
    'padding': '10px',
    'borderRadius': '5px',
    'backgroundColor': '#fadbd8'
}

success_style = {
    'color': '#27ae60',
    'marginTop': '10px',
    'padding': '10px',
    'borderRadius': '5px',
    'backgroundColor': '#d4efdf'
}

# Layout
app.layout = html.Div([
    html.H1("Patient Readmission Prediction Dashboard",
            style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),

    # Training Section
    html.Div([
        html.H2("Model Training", style={'color': '#34495e'}),
        html.P("Click the button below to run the complete training pipeline (data generation, preprocessing, model training)."),
        html.Button('Run Training Pipeline', id='train-button', n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                          'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        html.Div(id='training-output', style={'marginTop': 20, 'whiteSpace': 'pre-line'})
    ], style={'marginBottom': 50, 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px'}),

    # Prediction Section
    html.Div([
        html.H2("Make Prediction", style={'color': '#34495e'}),
        html.P("Enter patient information below to predict readmission risk."),

        html.Div([
            html.Div([
                html.Label("Age:"),
                dcc.Input(id='age', type='number', placeholder='Enter age (18-95)', min=18, max=95),
            ], style={'marginRight': '20px', 'display': 'inline-block'}),

            html.Div([
                html.Label("Gender:"),
                dcc.Dropdown(id='gender', options=[
                    {'label': 'Male', 'value': 'Male'},
                    {'label': 'Female', 'value': 'Female'},
                    {'label': 'Other', 'value': 'Other'}
                ], placeholder='Select gender'),
            ], style={'marginRight': '20px', 'display': 'inline-block'}),

            html.Div([
                html.Label("Admission Type:"),
                dcc.Dropdown(id='admission_type', options=[
                    {'label': 'Emergency', 'value': 'Emergency'},
                    {'label': 'Elective', 'value': 'Elective'},
                    {'label': 'Urgent', 'value': 'Urgent'}
                ], placeholder='Select admission type'),
            ], style={'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Length of Stay (days):"),
                dcc.Input(id='length_of_stay', type='number', placeholder='Enter days (1-30)', min=1, max=30),
            ], style={'marginRight': '20px', 'display': 'inline-block'}),

            html.Div([
                html.Label("Number of Lab Procedures:"),
                dcc.Input(id='num_lab_procedures', type='number', placeholder='Enter count (0-100)', min=0, max=100),
            ], style={'marginRight': '20px', 'display': 'inline-block'}),

            html.Div([
                html.Label("Number of Medications:"),
                dcc.Input(id='num_medications', type='number', placeholder='Enter count (1-40)', min=1, max=40),
            ], style={'display': 'inline-block'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("Diagnosis Code:"),
            dcc.Dropdown(id='diagnosis_code', options=[
                {'label': 'I10 - Hypertension', 'value': 'I10'},
                {'label': 'E11 - Diabetes', 'value': 'E11'},
                {'label': 'J44 - COPD', 'value': 'J44'},
                {'label': 'F32 - Depression', 'value': 'F32'},
                {'label': 'M79 - Back Pain', 'value': 'M79'},
                {'label': 'N18 - Kidney Disease', 'value': 'N18'},
                {'label': 'K21 - GERD', 'value': 'K21'},
                {'label': 'I50 - Heart Failure', 'value': 'I50'}
            ], placeholder='Select diagnosis code'),
        ], style={'marginBottom': '20px'}),

        html.Button('Predict Readmission Risk', id='predict-button', n_clicks=0,
                   style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px 20px',
                          'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

        html.Div(id='prediction-output', style={'marginTop': 20, 'fontSize': '18px', 'fontWeight': 'bold'})

    ], style={'marginBottom': 50, 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px'}),

    # Model Performance Section
    html.Div([
        html.H2("Model Performance", style={'color': '#34495e'}),
        html.Div(id='performance-metrics'),
        dcc.Graph(id='confusion-matrix-plot'),
        dcc.Graph(id='roc-curve-plot'),
        dcc.Graph(id='feature-importance-plot')
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px'})

], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# Callbacks
@app.callback(
    Output('training-output', 'children'),
    Input('train-button', 'n_clicks')
)
def run_training_pipeline(n_clicks):
    if n_clicks > 0:
        try:
            # Run the training script
            script_path = Path(__file__).parent / 'readmission_pipeline.py'
            result = subprocess.run([sys.executable, str(script_path)],
                                  capture_output=True, text=True, cwd=Path(__file__).parent)

            if result.returncode == 0:
                return f"Training completed successfully!\n\n{result.stdout}"
            else:
                return f"Training failed:\n\n{result.stderr}"

        except Exception as e:
            return f"Error running training: {str(e)}"

    return "Click 'Run Training Pipeline' to start training."

@app.callback(
    [Output('prediction-output', 'children'),
     Output('performance-metrics', 'children'),
     Output('confusion-matrix-plot', 'figure'),
     Output('roc-curve-plot', 'figure'),
     Output('feature-importance-plot', 'figure')],
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('admission_type', 'value'),
    State('length_of_stay', 'value'),
    State('num_lab_procedures', 'value'),
    State('num_medications', 'value'),
    State('diagnosis_code', 'value')
)
def make_prediction(n_clicks, age, gender, admission_type, length_of_stay,
                   num_lab_procedures, num_medications, diagnosis_code):
    # Default outputs
    prediction_text = ""
    metrics_div = html.Div("Train the model first to see performance metrics.")
    cm_fig = go.Figure()
    roc_fig = go.Figure()
    fi_fig = go.Figure()

    if n_clicks == 0:
        return prediction_text, metrics_div, cm_fig, roc_fig, fi_fig

    # Check if all inputs are provided
    if not all([age, gender, admission_type, length_of_stay,
               num_lab_procedures, num_medications, diagnosis_code]):
        prediction_text = "Please fill in all patient information fields."
        return prediction_text, metrics_div, cm_fig, roc_fig, fi_fig

    try:

        try:
            # Load model and scaler
            current_dir = Path(__file__).parent
            model_path = current_dir / 'src' / 'readmission_model.joblib'
            scaler_path = current_dir / 'src' / 'scaler.joblib'

            if not model_path.exists() or not scaler_path.exists():
                prediction_text = "Model not found. Please run training first."
                return prediction_text, metrics_div, cm_fig, roc_fig, fi_fig

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # First, verify all input values are present
            if not all([age, gender, admission_type, length_of_stay, num_lab_procedures, num_medications, diagnosis_code]):
                raise ValueError("All input fields must be filled")

            # Create numerical features first
            input_data = pd.DataFrame({
                'age': [float(age)],
                'length_of_stay': [float(length_of_stay)],
                'num_lab_procedures': [float(num_lab_procedures)],
                'num_medications': [float(num_medications)],
                'gender': [str(gender)],
                'admission_type': [str(admission_type)],
                'diagnosis_code': [str(diagnosis_code)]
            })

            # Calculate comorbidity score following training logic
            score = 0
            age_val = float(age)
            if age_val > 75:
                score += 2
            elif age_val > 65:
                score += 1
            elif age_val > 50:
                score += 0.5

            if length_of_stay > 14:
                score += 2
            elif length_of_stay > 7:
                score += 1

            if num_medications > 15:
                score += 1.5
            elif num_medications > 10:
                score += 0.5

            high_risk_codes = ['I10', 'E11', 'J44', 'N18']
            if diagnosis_code in high_risk_codes:
                score += 1

            # Add comorbidity score before encoding
            input_data['comorbidity_score'] = max(0, min(5, score))

            # Load full training data to get correct feature names and order
            train_data = pd.read_csv(current_dir / 'data' / 'synthetic_patient_data.csv')
            categorical_cols = ['gender', 'admission_type', 'diagnosis_code']

            # Add comorbidity score to training data to match model training
            train_scores = []
            for _, row in train_data.iterrows():
                score = 0
                if row['age'] > 75:
                    score += 2
                elif row['age'] > 65:
                    score += 1
                elif row['age'] > 50:
                    score += 0.5

                if row['length_of_stay'] > 14:
                    score += 2
                elif row['length_of_stay'] > 7:
                    score += 1

                if row['num_medications'] > 15:
                    score += 1.5
                elif row['num_medications'] > 10:
                    score += 0.5

                if row['diagnosis_code'] in ['I10', 'E11', 'J44', 'N18']:
                    score += 1

                train_scores.append(max(0, min(5, score)))

            train_data['comorbidity_score'] = train_scores

            # Process training data to get correct columns
            train_encoded = pd.get_dummies(train_data.drop('readmitted', axis=1), columns=categorical_cols)
            expected_columns = train_encoded.columns.tolist()
            
            # Process input data with the same columns as training data
            input_encoded = pd.get_dummies(input_data, columns=categorical_cols)

            # Add missing columns with 0s and ensure correct column order
            for col in expected_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training data exactly
            input_data = input_encoded[expected_columns]

            # Now we can scale and predict
            try:
                # Scale features using the same scaler as training
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                # Format prediction text
                risk_level = "High" if probability > 0.5 else "Low"
                prediction_text = f"Risk Level: {risk_level} Risk of Readmission\n"
                prediction_text += f"Probability of Readmission: {probability:.1%}\n"
                prediction_text += f"Comorbidity Score: {score:.1f}"
                
            except Exception as e:
                print("Debug info:")
                print("Input data shape:", input_data.shape)
                print("Input data columns:", input_data.columns.tolist())
                print("Input data types:", input_data.dtypes)
                raise Exception(f"Prediction failed: {str(e)}")

        except Exception as e:
            prediction_text = f"Error making prediction: {str(e)}"
            
        # Calculate model performance metrics if model exists
        if 'model' in locals() and 'scaler' in locals():

            # Get feature names from training data (with comorbidity_score added)
            train_data = pd.read_csv(current_dir / 'data' / 'synthetic_patient_data.csv')

            # Add comorbidity score to training data to match model training
            train_scores = []
            for _, row in train_data.iterrows():
                score = 0
                if row['age'] > 75:
                    score += 2
                elif row['age'] > 65:
                    score += 1
                elif row['age'] > 50:
                    score += 0.5

                if row['length_of_stay'] > 14:
                    score += 2
                elif row['length_of_stay'] > 7:
                    score += 1

                if row['num_medications'] > 15:
                    score += 1.5
                elif row['num_medications'] > 10:
                    score += 0.5

                if row['diagnosis_code'] in ['I10', 'E11', 'J44', 'N18']:
                    score += 1

                train_scores.append(max(0, min(5, score)))

            train_data['comorbidity_score'] = train_scores

            train_encoded = pd.get_dummies(train_data.drop('readmitted', axis=1), columns=categorical_cols)
            feature_names = train_encoded.columns.tolist()
            
            # Ensure feature names match model coefficients
            if len(feature_names) == len(model.coef_[0]):
                # Get feature importances
                importances = np.abs(model.coef_[0])
                
                # Sort feature importances
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True).tail(10)  # Show top 10
            else:
                print(f"Warning: Feature names ({len(feature_names)}) and model coefficients ({len(model.coef_[0])}) length mismatch")

            # Load test data
            test_data = pd.read_csv(current_dir / 'data' / 'synthetic_patient_data.csv')
            
            # Process test data exactly as in readmission_pipeline.py
            categorical_cols = ['gender', 'admission_type', 'diagnosis_code']
            X_test = test_data.copy()
            
            # Create comorbidity scores
            test_scores = []
            for _, row in X_test.iterrows():
                score = 0
                if row['age'] > 75:
                    score += 2
                elif row['age'] > 65:
                    score += 1
                elif row['age'] > 50:
                    score += 0.5
                
                if row['length_of_stay'] > 14:
                    score += 2
                elif row['length_of_stay'] > 7:
                    score += 1
                
                if row['num_medications'] > 15:
                    score += 1.5
                elif row['num_medications'] > 10:
                    score += 0.5
                
                if row['diagnosis_code'] in ['I10', 'E11', 'J44', 'N18']:
                    score += 1
                    
                test_scores.append(max(0, min(5, score)))
                
            test_data['comorbidity_score'] = test_scores
            
            # Prepare features - match training data preprocessing exactly
            X_test['comorbidity_score'] = test_scores
            X = pd.get_dummies(X_test.drop('readmitted', axis=1), columns=categorical_cols)
            y = X_test['readmitted']
            
            # Ensure X has all the columns the model expects
            missing_cols = set(train_encoded.columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[train_encoded.columns]  # Reorder columns to match training data
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Get predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            metrics_div = html.Div([
                html.H3("Model Performance Metrics", style={'color': '#2c3e50'}),
                html.Div([
                    html.P(f"Model Accuracy: {acc:.3f}"),
                    html.P(f"ROC AUC Score: {auc:.3f}"),
                    html.P(f"Number of Test Samples: {len(y)}")
                ], style={'marginBottom': '20px'})
            ])

            # Confusion Matrix
            cm_fig = go.Figure(data=go.Heatmap(
                z=[[0, 0], [0, 0]],  # You can update this with real values if available
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                hoverongaps=False
            ))
            cm_fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='Actual Label',
                height=400
            )

            # ROC Curve (placeholder with baseline)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Baseline',
                line=dict(dash='dash', color='gray')
            ))
            roc_fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400,
                showlegend=True
            )

            # Feature Importance
            fi_fig = go.Figure()
            fi_fig.add_trace(go.Bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h'
            ))
            fi_fig.update_layout(
                title='Top 10 Feature Importance',
                xaxis_title='Absolute Coefficient Value',
                yaxis_title='Feature',
                height=500,
                margin=dict(l=200)  # Add left margin for feature names
            )

            return prediction_text, metrics_div, cm_fig, roc_fig, fi_fig
        else:
            metrics_div = html.Div("Please run the training pipeline first to see model performance metrics.",
                                 style={'color': '#e74c3c', 'marginTop': '20px'})
            cm_fig = go.Figure()
            roc_fig = go.Figure()
            fi_fig = go.Figure()
            return prediction_text, metrics_div, cm_fig, roc_fig, fi_fig

    except Exception as e:
        print(f"Error details: {str(e)}")
        return (f"Error: {str(e)}", 
                html.Div("Error generating metrics.", style={'color': '#e74c3c', 'marginTop': '20px'}),
                go.Figure(), go.Figure(), go.Figure())

if __name__ == '__main__':
    print("Starting Dash app...")
    print("Open your browser to http://127.0.0.1:8050/")
    app.run(debug=True)
