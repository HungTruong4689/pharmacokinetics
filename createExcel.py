import pandas as pd
from tabulate import tabulate
# Load the dataset
kp = pd.read_excel("Data_vancomycin.xlsx", skiprows=1)

kp.columns = kp.columns.str.lower()
kp.columns = kp.columns.str.lstrip()
# Rename columns for Pumas compatibility
kp = kp.rename(columns={
    "patient's name": "patient",
    "use from date": "time",
    "vancomycin maintenance dose before dose adjustment": "amt",
    "cmax": "conc",
    "weight": "weight",
    "age": "age",
    "serum creatinine (scr)": "scr",
    "gender": "gender",
    "maintenance dose frequency": "evid" # 1 = dose given, 0 = observation


})

# Drop rows with missing time, amt, or conc (critical for PK modeling)
kp = kp.dropna(subset=["patient", "time", "amt", "conc", "weight", "age", "scr", "gender", "evid"])

# Convert 'time' to elapsed time (in days) since first dose per patient
kp['time'] = pd.to_datetime(kp['time'])
kp['time'] = kp.groupby("patient")['time'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)

# Keep only the relevant columns
kp = kp[["patient", "time", "amt", "conc", "weight", "age", "scr", "gender", "evid"]]

# Save to CSV for use in Julia
kp.to_csv("output.csv", index=False)