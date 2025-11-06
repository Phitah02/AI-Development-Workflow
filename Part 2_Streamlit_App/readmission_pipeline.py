"""
Complete Readmission Prediction Pipeline

This script combines all processes from Part 2:
1. Data generation
2. Data preprocessing (missing values, feature engineering, encoding)
3. Model training and evaluation
4. Model saving for deployment

Author: PETER KAMAU MWAURA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("READMISSION PREDICTION PIPELINE")
print("="*60)

# Create directories
data_dir = Path('data')
src_dir = Path('src')
data_dir.mkdir(exist_ok=True)
src_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: DATA GENERATION
# ============================================================================
print("\n" + "="*60)
print("STEP 1: DATA GENERATION")
print("="*60)

# Set random seed for reproducibility
np.random.seed(42)

# Number of patient records to generate
n_samples = 1000

print(f"Generating {n_samples} synthetic patient records...")

# Generate patient demographics
age = np.random.randint(18, 96, size=n_samples)
gender = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.50, 0.02])

# Generate admission type
admission_type = np.random.choice(['Emergency', 'Elective', 'Urgent'],
                                  size=n_samples, p=[0.50, 0.25, 0.25])

# Generate length of stay (skewed distribution - most patients stay shorter)
length_of_stay = np.random.exponential(scale=5, size=n_samples)
length_of_stay = np.clip(np.round(length_of_stay).astype(int), 1, 30)

# Generate lab procedures (correlated with length of stay and age)
num_lab_procedures = np.random.poisson(lam=20, size=n_samples)
num_lab_procedures = np.clip(num_lab_procedures + (length_of_stay // 3), 0, 100)

# Generate medications (correlated with age and length of stay)
num_medications = np.random.poisson(lam=8, size=n_samples)
num_medications = np.clip(num_medications + (age // 15) + (length_of_stay // 5), 1, 40)

# Generate diagnosis codes (common medical conditions)
diagnosis_codes = ['I10', 'E11', 'J44', 'F32', 'M79', 'N18', 'K21', 'I50']
diagnosis_code = np.random.choice(diagnosis_codes, size=n_samples,
                                  p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.08, 0.07, 0.07])

# Generate target variable (readmitted) with correlations
readmission_prob = np.zeros(n_samples)

for i in range(n_samples):
    prob = 0.2  # Base readmission probability

    # Age factor
    if age[i] > 65:
        prob += 0.15
    elif age[i] > 50:
        prob += 0.08

    # Length of stay factor
    if length_of_stay[i] > 14:
        prob += 0.20
    elif length_of_stay[i] > 7:
        prob += 0.10

    # Diagnosis code factor
    high_risk_diagnoses = ['I10', 'E11', 'J44', 'N18']
    if diagnosis_code[i] in high_risk_diagnoses:
        prob += 0.15

    # Medication count factor
    if num_medications[i] > 15:
        prob += 0.10

    # Admission type factor
    if admission_type[i] == 'Emergency':
        prob += 0.05

    readmission_prob[i] = min(prob, 0.85)  # Cap at 85%

# Generate binary readmission outcome based on probabilities
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

print(f"✓ Dataset generated successfully!")
print(f"✓ Saved to: {output_path}")
print(f"  - Total records: {len(df)}")
print(f"  - Readmission rate: {df['readmitted'].mean():.2%}")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*60)
print("STEP 2: DATA PREPROCESSING")
print("="*60)

# Introduce random missing values to simulate real-world data
np.random.seed(42)

# Introduce 5% missing values in numerical columns
numerical_cols = ['age', 'length_of_stay', 'num_lab_procedures', 'num_medications']
for col in numerical_cols:
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, col] = np.nan

# Introduce 3% missing values in categorical columns
categorical_cols = ['gender', 'admission_type', 'diagnosis_code']
for col in categorical_cols:
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[missing_indices, col] = np.nan

print("Missing values introduced for simulation.")

# Impute numerical columns with mean
for col in numerical_cols:
    mean_value = df[col].mean()
    df[col].fillna(mean_value, inplace=True)
    print(f"Imputed {col} with mean: {mean_value:.2f}")

# Impute categorical columns with mode
for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)
    print(f"Imputed {col} with mode: {mode_value}")

print("✓ Missing values handled!")

# Feature engineering: comorbidity score
np.random.seed(42)
comorbidity_scores = []

for idx, row in df.iterrows():
    score = 0

    # Age factor (older patients have higher scores)
    if row['age'] > 75:
        score += 2
    elif row['age'] > 65:
        score += 1
    elif row['age'] > 50:
        score += 0.5

    # Length of stay factor (longer stays indicate complexity)
    if row['length_of_stay'] > 14:
        score += 2
    elif row['length_of_stay'] > 7:
        score += 1

    # Medication count factor (more medications = more conditions)
    if row['num_medications'] > 15:
        score += 1.5
    elif row['num_medications'] > 10:
        score += 0.5

    # Diagnosis code factor (certain conditions are more complex)
    high_complexity_codes = ['I10', 'E11', 'J44', 'N18']
    if row['diagnosis_code'] in high_complexity_codes:
        score += 1

    # Add some randomness to make it more realistic
    score += np.random.uniform(-0.5, 0.5)

    # Ensure score is between 0 and 5
    score = max(0, min(5, score))
    comorbidity_scores.append(round(score, 2))

# Add the new feature to the dataframe
df['comorbidity_score'] = comorbidity_scores

print("✓ Comorbidity score feature created!")

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
df = df_encoded.copy()

print("✓ Categorical variables encoded!")

# Split data
X = df.drop('readmitted', axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("✓ Data split into training and testing sets!")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 3: MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "="*60)
print("STEP 3: MODEL TRAINING AND EVALUATION")
print("="*60)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled!")

# Train initial model
model_initial = LogisticRegression(random_state=42, max_iter=1000)
model_initial.fit(X_train_scaled, y_train)

print("✓ Initial model trained!")

# Train optimized model
model_optimized = LogisticRegression(C=0.1, penalty='l2', random_state=42, max_iter=1000)
model_optimized.fit(X_train_scaled, y_train)

print("✓ Optimized model trained!")

# Evaluate optimized model
y_pred_optimized = model_optimized.predict(X_test_scaled)
y_pred_proba_optimized = model_optimized.predict_proba(X_test_scaled)[:, 1]

accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
precision_optimized = precision_score(y_test, y_pred_optimized)
recall_optimized = recall_score(y_test, y_pred_optimized)
f1_optimized = f1_score(y_test, y_pred_optimized)

cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"Accuracy:  {accuracy_optimized:.4f}")
print(f"Precision: {precision_optimized:.4f}")
print(f"Recall:    {recall_optimized:.4f}")
print(f"F1-Score:  {f1_optimized:.4f}")

print(f"\nConfusion Matrix:")
print(cm_optimized)

# ============================================================================
# STEP 4: SAVE MODEL AND SCALER
# ============================================================================
print("\n" + "="*60)
print("STEP 4: SAVE MODEL AND SCALER")
print("="*60)

# Save the optimized model
model_path = src_dir / 'readmission_model.joblib'
joblib.dump(model_optimized, model_path)
print(f"✓ Model saved to: {model_path}")

# Save the scaler
scaler_path = src_dir / 'scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to: {scaler_path}")

# Verify loading
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
test_predictions = loaded_model.predict(loaded_scaler.transform(X_test.iloc[:5]))
print("✓ Model and scaler loading verified!")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("The trained model and scaler are ready for deployment.")
print("Run 'streamlit run app.py' to launch the prediction app.")
