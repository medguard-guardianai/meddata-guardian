"""
MedData Guardian - Synthetic Dataset Generator
Generates 4 datasets for hackathon demo with intentional problems
"""

import pandas as pd
import numpy as np
from faker import Faker
import os

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)

print("=" * 60)
print("MedData Guardian - Synthetic Dataset Generator")
print("=" * 60)

# Create output directory
os.makedirs('../data/synthetic', exist_ok=True)

# ============================================================================
# DATASET 1: Heart Disease - Data Quality Issues
# ============================================================================

print("\n📊 Generating Dataset 1: Heart Disease (Data Quality Focus)...")

n_patients = 1000

data1 = {
    'patient_id': [f'SYNTH_{i:04d}' for i in range(1, n_patients + 1)],
    'age': np.random.randint(25, 85, n_patients),
    'sex': np.random.choice(['Male', 'Female', 'M', 'F', '1', '0'], n_patients, 
                           p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]),
    'chest_pain_type': np.random.choice(['typical', 'Typical', 'atypical', 'non-anginal', 'asymptomatic'], n_patients),
    'resting_blood_pressure': np.random.randint(90, 200, n_patients),
    'cholesterol': np.random.randint(120, 400, n_patients),
    'fasting_blood_sugar': np.random.choice([0, 1], n_patients),
    'rest_ecg': np.random.choice([0, 1, 2], n_patients),
    'max_heart_rate': np.random.randint(60, 202, n_patients),
    'exercise_angina': np.random.choice([0, 1], n_patients),
    'oldpeak': np.random.uniform(0, 6, n_patients).round(1),
    'slope': np.random.choice([0, 1, 2], n_patients),
    'num_major_vessels': np.random.choice([0, 1, 2, 3], n_patients),
    'thalassemia': np.random.choice([0, 1, 2, 3], n_patients),
    'heart_disease': np.random.choice([0, 1], n_patients, p=[0.45, 0.55])
}

df1 = pd.DataFrame(data1)

# INJECT PROBLEMS
print("  ⚠️  Planting data quality issues...")

# 1. Missing cholesterol (17.2%)
missing_chol = np.random.choice(df1.index, size=172, replace=False)
df1.loc[missing_chol, 'cholesterol'] = np.nan
print(f"    - {len(missing_chol)} missing cholesterol values (17.2%)")

# 2. Missing blood pressure (0.8%)
missing_bp = np.random.choice(df1.index, size=8, replace=False)
df1.loc[missing_bp, 'resting_blood_pressure'] = np.nan
print(f"    - {len(missing_bp)} missing blood pressure values (0.8%)")

# 3. Duplicates (23 rows)
duplicate_rows = df1.sample(23)
df1 = pd.concat([df1, duplicate_rows], ignore_index=True)
print(f"    - 23 duplicate records")

# 4. Outliers in cholesterol (>450)
outlier_indices = np.random.choice(df1.index, size=14, replace=False)
df1.loc[outlier_indices, 'cholesterol'] = np.random.randint(450, 600, 14)
print(f"    - 14 cholesterol outliers (>450 mg/dL)")

# 5. Low heart rate outliers
low_hr = np.random.choice(df1.index, size=3, replace=False)
df1.loc[low_hr, 'max_heart_rate'] = np.random.randint(40, 58, 3)
print(f"    - 3 impossible heart rate values (<60 bpm)")

# 6. Age = 0 (impossible)
df1.loc[567, 'age'] = 0
df1.loc[678, 'age'] = 0
print(f"    - 2 impossible age values (age=0)")

# Save
output_path = '../data/synthetic/dataset1_heart_disease_quality.csv'
df1.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")
print(f"   Total records: {len(df1)} (including duplicates)")

# ============================================================================
# DATASET 2: Diabetes - Gender Bias
# ============================================================================

print("\n📊 Generating Dataset 2: Diabetes (Gender Bias Focus)...")

n_patients = 1200

# Generate with 73% male bias
n_male = 876
n_female = 324

data2 = {
    'patient_id': [f'SYNTH_{i:04d}' for i in range(1, n_patients + 1)],
    'age': np.random.randint(30, 80, n_patients),
    'sex': ['Male'] * n_male + ['Female'] * n_female,
    'bmi': np.random.uniform(18, 45, n_patients).round(1),
    'blood_pressure': np.random.randint(80, 180, n_patients),
    'glucose': np.random.randint(70, 200, n_patients),
    'insulin': np.random.randint(0, 300, n_patients),
    'skin_thickness': np.random.randint(10, 60, n_patients),
    'pregnancies': np.concatenate([
        np.zeros(n_male),  # Males = 0 pregnancies
        np.random.randint(0, 10, n_female)  # Females 0-10
    ]),
    'family_history': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'physical_activity': np.random.choice(['low', 'medium', 'high'], n_patients),
    'smoking': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
    'alcohol': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'diabetes': np.random.choice([0, 1], n_patients, p=[0.65, 0.35])
}

df2 = pd.DataFrame(data2)

# Shuffle rows
df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)

print("  ⚠️  Bias planted:")
print(f"    - Male: {n_male} patients (73.0%)")
print(f"    - Female: {n_female} patients (27.0%)")
print(f"    - Expected: 50/50 split")

# Save
output_path = '../data/synthetic/dataset2_diabetes_gender_bias.csv'
df2.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")
print(f"   Total records: {len(df2)}")

# ============================================================================
# DATASET 3: Heart Disease - Indigenous/Race Bias
# ============================================================================

print("\n📊 Generating Dataset 3: Heart Disease (Indigenous Bias Focus)...")

n_patients = 800

# Race distribution with 0% Indigenous
race_dist = {
    'White': 680,
    'Black': 80,
    'Hispanic': 40,
    'Indigenous': 0,  # CRITICAL GAP
    'Asian': 0
}

races = []
for race, count in race_dist.items():
    races.extend([race] * count)

data3 = {
    'patient_id': [f'SYNTH_{i:04d}' for i in range(1, n_patients + 1)],
    'age': np.random.randint(35, 85, n_patients),
    'sex': np.random.choice(['Male', 'Female'], n_patients, p=[0.55, 0.45]),
    'race_ethnicity': races,
    'cholesterol': np.random.randint(150, 350, n_patients),
    'blood_pressure': np.random.randint(90, 190, n_patients),
    'bmi': np.random.uniform(20, 40, n_patients).round(1),
    'smoking': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
    'diabetes': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
    'family_history': np.random.choice([0, 1], n_patients, p=[0.65, 0.35]),
    'exercise': np.random.choice(['none', 'light', 'moderate', 'heavy'], n_patients),
    'diet_quality': np.random.choice(['poor', 'fair', 'good', 'excellent'], n_patients),
    'stress_level': np.random.randint(1, 11, n_patients),
    'heart_disease': np.random.choice([0, 1], n_patients, p=[0.55, 0.45])
}

df3 = pd.DataFrame(data3)

# Shuffle
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)

print("  ⚠️  CRITICAL BIAS - Race distribution:")
for race, count in race_dist.items():
    pct = (count / n_patients) * 100
    print(f"    - {race}: {count} patients ({pct:.1f}%)")
print("  🚨 Indigenous Americans: 0% (HIGHEST RISK GROUP MISSING)")

# Save
output_path = '../data/synthetic/dataset3_heart_disease_indigenous.csv'
df3.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")
print(f"   Total records: {len(df3)}")

# ============================================================================
# DATASET 4: Diabetes - Combined Problems
# ============================================================================

print("\n📊 Generating Dataset 4: Diabetes (Combined Quality + Bias)...")

n_patients = 1000

# Gender bias (68% male)
n_male = 680
n_female = 320

# Race with gaps
races_combined = (
    ['White'] * 700 +
    ['Black'] * 150 +
    ['Hispanic'] * 30 +  # Underrepresented (should be 18%)
    ['Indigenous'] * 0 +  # Missing entirely
    ['Asian'] * 50 +
    ['Other'] * 70
)

data4 = {
    'patient_id': [f'SYNTH_{i:04d}' for i in range(1, n_patients + 1)],
    'age': np.random.randint(25, 75, n_patients),  # No elderly (>75)
    'sex': ['Male'] * n_male + ['Female'] * n_female,
    'race_ethnicity': races_combined,
    'bmi': np.random.uniform(18, 50, n_patients).round(1),
    'glucose': np.random.randint(60, 220, n_patients),
    'hba1c': np.random.uniform(4.5, 14, n_patients).round(1),
    'blood_pressure_systolic': np.random.randint(90, 200, n_patients),
    'blood_pressure_diastolic': np.random.randint(60, 120, n_patients),
    'cholesterol': np.random.randint(120, 350, n_patients),
    'triglycerides': np.random.randint(50, 400, n_patients),
    'family_history': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
    'smoking': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
    'alcohol_use': np.random.choice([0, 1, 2], n_patients),  # 0=none, 1=moderate, 2=heavy
    'physical_activity': np.random.choice([0, 1, 2], n_patients),  # 0=low, 1=med, 2=high
    'medications': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    'diabetes': np.random.choice([0, 1], n_patients, p=[0.63, 0.37])
}

df4 = pd.DataFrame(data4)

# Shuffle
df4 = df4.sample(frac=1, random_state=42).reset_index(drop=True)

# INJECT DATA QUALITY PROBLEMS
print("  ⚠️  Planting data quality issues...")

# Missing HbA1c (24.5%)
missing_hba1c = np.random.choice(df4.index, size=245, replace=False)
df4.loc[missing_hba1c, 'hba1c'] = np.nan
print(f"    - {len(missing_hba1c)} missing HbA1c values (24.5%)")

# Missing BMI (8.9%)
missing_bmi = np.random.choice(df4.index, size=89, replace=False)
df4.loc[missing_bmi, 'bmi'] = np.nan
print(f"    - {len(missing_bmi)} missing BMI values (8.9%)")

# Duplicates (31 rows)
duplicate_rows = df4.sample(31)
df4 = pd.concat([df4, duplicate_rows], ignore_index=True)
print(f"    - 31 duplicate records")

# Glucose outliers
glucose_outliers = np.random.choice(df4.index, size=12, replace=False)
df4.loc[glucose_outliers, 'glucose'] = np.random.randint(300, 500, 12)
print(f"    - 12 glucose outliers (>300 mg/dL)")

print("  ⚠️  BIAS ISSUES:")
print(f"    - Gender: {n_male} male (68%), {n_female} female (32%)")
print(f"    - Age: No patients >75 (elderly excluded)")
print(f"    - Hispanic: 30 patients (3%) vs expected 18%")
print(f"    - Indigenous: 0 patients (0%) - MISSING ENTIRELY")

# Save
output_path = '../data/synthetic/dataset4_diabetes_combined.csv'
df4.to_csv(output_path, index=False)
print(f"✅ Saved: {output_path}")
print(f"   Total records: {len(df4)} (including duplicates)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\nDatasets created:")
print("  1. dataset1_heart_disease_quality.csv - Data quality issues")
print("  2. dataset2_diabetes_gender_bias.csv - Gender bias (73% male)")
print("  3. dataset3_heart_disease_indigenous.csv - Race bias (0% Indigenous)")
print("  4. dataset4_diabetes_combined.csv - Both quality + bias issues")
print("\nLocation: data/synthetic/")
print("=" * 60)