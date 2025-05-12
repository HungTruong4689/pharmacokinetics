from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import json

# Import main functions 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import solve_two_compartment, find_dose_for_target_cmin, calculate_auc


app = FastAPI()

# ---------------------------
# Input Schema
# ---------------------------
class PatientRecord(BaseModel):
    patient: int
    time: float
    amt: float
    evid: int
    conc: Optional[float] = None
    weight: float
    scr: float
    age: int
    gender: int

class SimulationInput(BaseModel):
    target_cmin: float
    new_patient: List[PatientRecord]

# ---------------------------
# Load static population parameters (or load from file/model if needed)
# ---------------------------
try:
    with open("pop_params.json", "r") as f:
        pop_params = json.load(f)
except FileNotFoundError:
    raise RuntimeError("Population parameters not found. Please run main.py first.")


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/simulate")
def simulate(input_data: SimulationInput):
    try:
        # Convert to DataFrame
        patient_df = pd.DataFrame([p.dict() for p in input_data.new_patient])

        # Estimate dose
        estimated_dose = find_dose_for_target_cmin(input_data.target_cmin, patient_df.copy(), pop_params)
        patient_df.loc[patient_df['evid'] == 1, 'amt'] = estimated_dose

        # Simulate concentration
        times, concs = solve_two_compartment(patient_df, pop_params)

        # Calculate AUC
        auc = calculate_auc(times, concs)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(times, concs, label='Predicted Concentration', marker='o')
        ax.axhline(input_data.target_cmin, color='red', linestyle='--', label='Target Cmin')
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title(f"Simulated Concentration\nTarget Cmin: {input_data.target_cmin:.2f} mg/L | AUC: {auc:.2f}")
        ax.legend()
        ax.grid(True)

        # Convert plot to base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Return result
        return {
            "estimated_dose_mg": round(estimated_dose, 2),
            "predicted_auc": round(auc, 2),
            "plot_base64": plot_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
