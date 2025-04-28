import pandas as pd
from tabulate import tabulate
# Load the dataset
df = pd.read_excel("Data_vancomycin.xlsx", skiprows=1)

df.columns = df.columns.str.lower()
df.columns = df.columns.str.lstrip()

# Initialize summary dict
summary = {}

# Age
if 'age' in df.columns:
    summary['Age (years)'] = f"{df['age'].mean():.1f} ± {df['age'].std():.1f}"

# Gender (0 = Male, 1 = Female)
if 'gender' in df.columns:
    male_count = (df['gender'] == 0).sum()
    female_count = (df['gender'] == 1).sum()
    total = male_count + female_count
    summary['Male'] = f"{male_count} ({(male_count / total * 100):.1f}%)"
    summary['Female'] = f"{female_count} ({(female_count / total * 100):.1f}%)"

# BMI
if 'bmi' in df.columns:
    summary['BMI (kg/m2)'] = f"{df['bmi'].mean():.1f} ± {df['bmi'].std():.1f}"

# Comorbidities (0 = No, 1 = Yes)
comorbidities = {
    'diabetes': 'Diabetes',
    'hypertension': 'Hypertension',
    'heart failure': 'Heart Failure',
    'kidney failure': 'Kidney Failure'
}

for column, label in comorbidities.items():
    if column in df.columns:
        yes_count = (df[column] == 1).sum()
        summary[label] = f"{yes_count} ({(yes_count / len(df) * 100):.1f}%)"

# Other diseases (if not empty, then it's present)
if 'other diseases' in df.columns:
    other_count = df['other diseases'].astype(str).str.strip().replace('nan', '').replace('NaN', '').ne('').sum()
    summary['Other Diseases'] = f"{other_count} ({(other_count / len(df) * 100):.1f}%)"

# Sample vs. patient count
if 'hsba code' in df.columns:
    summary['Samples'] = f"{len(df)}"
    summary['Unique Patients'] = f"{df['hsba code'].nunique()}"

# Lab values (replace column names if needed)
lab_columns = {
    'vancomycin loading dose': 'Vancomycin Serum Concentration (mg/L)',
    'serum creatinine (scr)': 'Creatinine (SCr)(mg/dL) ',
    'creatinine clearance (clcr)': 'Creatinine Clearance (mL/min)',
    'blood urea nitrogen': 'BUN (mg/dL)',
    'osmolality': 'Osmolality (mOsm/kg)'
}

for col, label in lab_columns.items():
    if col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if not series.empty:
            summary[label] = f"{series.mean():.1f} ± {series.std():.1f}"

# Convert summary to DataFrame
summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])

# Pretty print
# print(tabulate(summary_df, headers=['Characteristic', 'Value'], tablefmt='grid'))
print(summary_df)