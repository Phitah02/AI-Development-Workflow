"""
Combined Readmission Pipeline Script No 6

This script combines all processes from Part 2 into a single executable:
1. Data generation (from generate_data.py)
2. Data preprocessing (from 01_data_preprocessing.ipynb)
3. Model training and evaluation (from 02_model_training.ipynb)

Run this script to generate data, preprocess it, train the model, and save artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create directories
current_dir = Path(__file__).parent
data_dir = current_dir / 'data'
src_dir = current_dir / 'src'
data_dir.mkdir(exist_ok=True)
src_dir.mkdir(exist_ok=True)

print("Starting combined readmission pipeline...")

# Step 1: Data Generation
print("\n" + "="*50)
print("STEP 1: DATA GENERATION")
print("="*50)

n_samples = 1000
print(f"Generating {n_samples} synthetic patient records...")

# Generate patient demographics
age = np.random.randint(18, 96, size=n_samples)
gender = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.50, 0.02])
admission_type = np.random.choice(['Emergency', 'Elective', 'Urgent'], size=n_samples, p=[0.50, 0.25, 0.25])
length_of_stay = np.random.exponential(scale=5, size=n_samples)
length_of_stay = np.clip(np.round(length_of_stay).astype(int), 1, 30)
num_lab_procedures = np.random.poisson(lam=20, size=n_samples)
num_lab_procedures = np.clip(num_lab_procedures + (length_of_stay // 3), 0, 100)
num_medications = np.random.poisson(lam=8, size=n_samples)
num_medications = np.clip(num_medications + (age // 15) + (length_of_stay // 5), 1, 40)
diagnosis_codes = ['I10', 'E11', 'J44', 'F32', 'M79', 'N18', 'K21', 'I50']
diagnosis_code = np.random.choice(diagnosis_codes, size=n_samples, p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.08, 0.07, 0.07])

# Generate target variable
readmission_prob = np.zeros(n_samples)
for i in range(n_samples):
    prob = 0.2
    if age[i] > 65:
        prob += 0.15
    elif age[i] > 50:
        prob += 0.08
    if length_of_stay[i] > 14:
        prob += 0.20
    elif length_of_stay[i] > 7:
        prob += 0.10
    high_risk_diagnoses = ['I10', 'E11', 'J44', 'N18']
    if diagnosis_code[i] in high_risk_diagnoses:
        prob += 0.15
    if num_medications[i] > 15:
        prob += 0.10
    if admission_type[i] == 'Emergency':
        prob += 0.05
    readmission_prob[i] = min(prob, 0.85)

readmitted = np.random.binomial(1, readmission_prob, size=n_samples)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'admission_type': admission_type,
    'length_of_stay': length_of_stay,
    'num_lab_procedures': num_lab_procedures,
    'num_medications': num_medications,
    'diagnosis_code': diagnosis_code,
    'readmitted': readmitted
})

# Save to CSV
output_path = data_dir / 'synthetic_patient_data.csv'
df.to_csv(output_path, index=False)
print(f"[OK] Dataset generated and saved to: {output_path}")
print(f"Readmission rate: {df['readmitted'].mean():.2%}")

# Step 2: Data Preprocessing
print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING")
print("="*50)

# Introduce missing values
numerical_cols = ['age', 'length_of_stay', 'num_lab_procedures', 'num_medications']
categorical_cols = ['gender', 'admission_type', 'diagnosis_code']
for col in numerical_cols:
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, col] = np.nan
for col in categorical_cols:
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[missing_indices, col] = np.nan

# Impute missing values
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("[OK] Missing values handled")

# Feature engineering: comorbidity score
comorbidity_scores = []
for idx, row in df.iterrows():
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
    high_complexity_codes = ['I10', 'E11', 'J44', 'N18']
    if row['diagnosis_code'] in high_complexity_codes:
        score += 1
    score += np.random.uniform(-0.5, 0.5)
    score = max(0, min(5, score))
    comorbidity_scores.append(round(score, 2))

df['comorbidity_score'] = comorbidity_scores
print("[OK] Comorbidity score feature added")

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
df = df_encoded.copy()
print("[OK] Categorical variables encoded")

# Data splitting
X = df.drop('readmitted', axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("[OK] Data split into train/test sets")

# Step 3: Model Training
print("\n" + "="*50)
print("STEP 3: MODEL TRAINING")
print("="*50)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[OK] Features scaled")

# Train optimized model
model = LogisticRegression(C=0.1, penalty='l2', random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
print("[OK] Model trained")

# Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("[OK] Model evaluated")
print(".4f")
print(".4f")
print(".4f")
print(".4f")
print(".4f")

# Save model and scaler
model_path = src_dir / 'readmission_model.joblib'
scaler_path = src_dir / 'scaler.joblib'
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"[OK] Model saved to: {model_path}")
print(f"[OK] Scaler saved to: {scaler_path}")

print("\n" + "="*50)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)
print("All artifacts saved and ready for deployment.")
