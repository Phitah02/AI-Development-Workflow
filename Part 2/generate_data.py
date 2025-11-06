"""
Synthetic Patient Data Generation Script

This script generates a realistic synthetic dataset for patient readmission prediction.
The data includes patient demographics, admission details, and medical information.
The target variable (readmitted) is correlated with various features to simulate
real-world patterns where certain patient characteristics increase readmission risk.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

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
# Using exponential distribution for realistic skew
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
# Higher readmission risk for:
# - Older patients (age > 65)
# - Longer stays (length_of_stay > 7)
# - Certain diagnosis codes (I10, E11, J44, N18)
# - More medications (num_medications > 15)

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
data = pd.DataFrame({
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
data.to_csv(output_path, index=False)

print(f"✓ Dataset generated successfully!")
print(f"✓ Saved to: {output_path}")
print(f"\nDataset Summary:")
print(f"  - Total records: {len(data)}")
print(f"  - Readmission rate: {data['readmitted'].mean():.2%}")
print(f"\nFirst 5 rows:")
print(data.head())
print(f"\nDataset info:")
print(data.info())



