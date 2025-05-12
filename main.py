import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import json

# --- Dose Finder and AUC Calculation Extension ---
from scipy.optimize import minimize
from scipy.integrate import simps

# Two-compartment ODE system
# Define the ODE system for a two-compartment model
# ODE stands for "Ordinary Differential Equation"
# This model describes how a drug moves between two compartments in the body
# A1: Amount in central compartment
# A2: Amount in peripheral compartment
# V1: Volume of central compartment
# V2: Volume of peripheral compartment
# CL1: Clearance from central compartment
# CL2: Clearance from peripheral compartment
# Q: Inter-compartmental clearance
# dA1/dt = - (CL1 / V1) * A1 - Q * (A1 / V1 - A2 / V2)
# dA2/dt = Q * (A1 / V1 - A2 / V2)
# This function defines the ODE system for the two-compartment model
# The function takes time t, state variables y (A1, A2), and model parameters as inputs
# It returns the derivatives of A1 and A2 with respect to time
# The function uses the two-compartment model equations to calculate the rates of change
# of the drug amounts in the two compartments
# The function returns a list of the derivatives [dA1/dt, dA2/dt]
# The function uses the two-compartment model equations to calculate the rates of change
# of the drug amounts in the two compartments
def two_compartment_ode(t, y, V1, V2, CL1, CL2, Q):
    A1, A2 = y
    dA1dt = - (CL1 / V1) * A1 - Q * (A1 / V1 - A2 / V2)
    dA2dt = Q * (A1 / V1 - A2 / V2)
    return [dA1dt, dA2dt]


# Solve the system for one patient
## Simulates one patient's concentration-time curve:
## Starts with initial conditions.
## Adjusts V1, V2, CL1, CL2, Q based on patient weight, age, creatinine (Scr), and gender.
## Handles when doses happen (evid=1), observations (evid=0).
def solve_two_compartment(patient_data, base_params):
    times = patient_data['time'].values
    amts = patient_data['amt'].values # Amount of drug administered (mg)
    evids = patient_data['evid'].values # 1 = dose given, 0 = observation

    weight = patient_data['weight'].iloc[0]
    age = patient_data['age'].iloc[0]
    scr = patient_data['scr'].iloc[0]
    gender = patient_data['gender'].iloc[0]  # 0 for male, 1 for female
    sex_factor = 0.85 if gender == 1 else 1.0 #is set to 0.85 for females because it accounts for physiological differences in muscle mass and creatinine production between males and females. This adjustment is part of the Cockcroft-Gault equation, which is commonly used to estimate creatinine clearance (CLcr) as a measure of kidney function.

    # Covariate-adjusted parameters
    V1, V2, CL1, CL2, Q = base_params
    V1 *= weight / 70
    V2 *= weight / 70
    CL2 *= weight / 70
    Q *= weight / 70

    # Estimate CLcr using Cockcroft-Gault
    clcr = ((140 - age) * weight) / (72 * scr) * sex_factor  # mL/min
    clcr_Lph = clcr * 60 / 1000  # Convert to L/h

    CL1 = clcr_Lph  # Assume CL1 is renal clearance

    A1, A2 = 0, 0
    result_times = []
    result_concs = []

    for i in range(1, len(times)):
        t_start = times[i - 1]
        t_end = times[i]

        if evids[i - 1] == 1:
            A1 += amts[i - 1]

        sol = solve_ivp(
            fun=two_compartment_ode,
            t_span=[t_start, t_end],
            y0=[A1, A2],
            args=(V1, V2, CL1, CL2, Q),
            t_eval=[t_end]
        )

        A1, A2 = sol.y[:, -1]
        result_times.append(t_end)
        result_concs.append(A1 / V1)

    return np.array(result_times), np.array(result_concs)


# Population objective function
## Calculates total error between observed and predicted concentrations for the whole group.
## Goal: find population parameters that minimize total squared errors.
## The function iterates through each patient, simulates their concentration-time curve,
## and computes the squared error between observed and predicted concentrations.
## The function returns the total error across all patients.
def population_objective(params, data):
    total_error = 0
    for patient_id, patient_data in data.groupby('patient'):
        pred_times, pred_concs = solve_two_compartment(patient_data, params)
        obs_data = patient_data[patient_data['evid'] == 0]
        obs_concs = obs_data['conc'].values
        error = np.sum((obs_concs - pred_concs) ** 2)
        total_error += error
    return total_error


# Estimate population parameters
## Minimizes population objective using scipy.optimize.minimize.
## Estimates best-fit population-level V1, V2, CL1, CL2, Q.

def estimate_population_parameters(data, initial_params):
    result = minimize(population_objective, initial_params, args=(data,), method='L-BFGS-B')
    return result.x


# Estimate individual parameters
## Focuses on one patient.
## Tweaks parameters just for that patient to fit their data better.
def estimate_individual_parameters(patient_data, initial_params):
    def individual_objective(params):
        _, pred_concs = solve_two_compartment(patient_data, params)
        obs_concs = patient_data[patient_data['evid'] == 0]['conc'].values
        return np.sum((obs_concs - pred_concs) ** 2)

    result = minimize(individual_objective, initial_params, method='L-BFGS-B')
    return result.x


# Shrinkage
## Tells you how much variation there is between individual patients and population model.
## Low shrinkage = your model captures individual differences well!
def calculate_shrinkage(individual_params, population_params):
    return 100 * np.mean(np.abs(np.array(individual_params) - np.array(population_params)) / population_params)


# Predict for new patients
## Uses the model to predict concentrations for new, unseen patients — very useful for clinical decisions!
def predict_new_patient(new_patient_data, pop_params):
    times, preds = solve_two_compartment(new_patient_data, pop_params)
    pred_df = pd.DataFrame({'time': times, 'pred_conc': preds})
    return pred_df

# Define multiple new patients
def predict_batch_new_patients(new_data, pop_params):
    all_preds = []
    for pid, pdata in new_data.groupby('patient'):
        times, preds = solve_two_compartment(pdata, pop_params)
        pred_df = pd.DataFrame({
            'patient': pid,
            'time': times,
            'pred_conc': preds
        })
        all_preds.append(pred_df)
    return pd.concat(all_preds, ignore_index=True)


def plot_observed_vs_individual(data, individual_params):
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': 'white',
        'axes.titlecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.facecolor': '#222222',
    })
    plt.figure(figsize=(10, 6))

    all_preds = []
    all_obs = []

    for pid, pdata in data.groupby('patient'):
        # Get individual predictions
        times, preds = solve_two_compartment(pdata, individual_params[pid - 1])

        # Observed concentrations (evid == 0)
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values

        # Store for setting axis limits
        all_preds.extend(preds)
        all_obs.extend(obs_concs)

        # Plot observed vs individual predicted
        plt.scatter(preds, obs_concs, label=f'Patient {pid}', alpha=0.6)

    # Diagonal identity line
    min_val = min(min(all_preds), min(all_obs))
    max_val = max(max(all_preds), max(all_obs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Identity Line')

    plt.xlabel("Individual Predicted Concentration (ng/mL)")
    plt.ylabel("Observed Concentration (ng/mL)")
    plt.title("Observed vs Individual Predicted Concentrations")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.figtext(
    0.5, -0.1,
    "Figure 1. Observed vs. individual predicted vancomycin concentrations.\n"
    "Each point represents a measurement for one patient at a given time.\n"
    "Perfect predictions would fall on the identity line (dashed).\n"
    "Deviations from the line indicate model inaccuracies or variability.\n"
    "Good model fit is indicated by clustering of points near the line.",
    wrap=True, ha='center', fontsize=10,color='white'
)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to fit the text
    plt.show()



# Plot observed vs population prediction
def plot_observed_vs_population(data, pop_params):
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': 'white',
        'axes.titlecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.facecolor': '#222222',
    })
    plt.figure(figsize=(10, 6))

    all_preds = []
    all_obs = []

    for pid, pdata in data.groupby('patient'):
        # Get population model predictions
        times, preds = solve_two_compartment(pdata, pop_params)

        # Observed concentrations (evid == 0)
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values

        # Store all points for setting axis limits
        all_preds.extend(preds)
        all_obs.extend(obs_concs)

        # Plot observed vs population predicted
        plt.scatter(preds, obs_concs, label=f'Patient {pid}', alpha=0.6)

    # Diagonal identity line
    min_val = min(min(all_preds), min(all_obs))
    max_val = max(max(all_preds), max(all_obs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Identity Line')

    plt.xlabel("Population Predicted Concentration (ng/mL)")
    plt.ylabel("Observed Concentration (ng/mL)")
    plt.title("Observed vs Population Predicted Concentrations")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.figtext(
    0.5, -0.1,
    "Figure 2. Observed vs. population predicted concentrations.\n"
    "This plot evaluates how well the population-average parameters predict individual outcomes.\n"
    "Greater scattering around the identity line suggests the need for individualized fitting.",
    wrap=True, ha='center', fontsize=10,color='white'
)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to fit the text
    plt.show()

# Plot conditional weighted residuals
def plot_cwres(data, individual_params, population_params):
    # Plot Individual CWRES
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': 'white',
        'axes.titlecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.facecolor': '#222222',
    })
    plt.figure(figsize=(10, 6))
    for pid, pdata in data.groupby('patient'):
        times, individual_preds = solve_two_compartment(pdata, individual_params[pid - 1])
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values
        individual_residuals = obs_concs - individual_preds
        individual_cwres = individual_residuals  # assuming sigma = 1

        plt.plot(times, individual_cwres, 'o', label=f'Patient {pid}', alpha=0.6)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Time Post Dose (h)")
    plt.ylabel("Conditional Weighted Residuals (Individual)")
    plt.title("CWRES for Individual Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

    # Plot Population CWRES
    plt.figure(figsize=(10, 6))
    for pid, pdata in data.groupby('patient'):
        _, population_preds = solve_two_compartment(pdata, population_params)
        obs_data = pdata[pdata['evid'] == 0]
        obs_concs = obs_data['conc'].values
        population_residuals = obs_concs - population_preds
        population_cwres = population_residuals  # assuming sigma = 1

        times = pdata[pdata['evid'] == 0]['time'].values
        plt.plot(times, population_cwres, 'o', label=f'Patient {pid}', alpha=0.6)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Time Post Dose (h)")
    plt.ylabel("Conditional Weighted Residuals (Population)")
    plt.title("CWRES for Population Predictions")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.figtext(
    0.5, -0.1,
    "Figure 3. Conditional Weighted Residuals (CWRES) over time and prediction.\n"
    "CWRES should randomly scatter around zero.\n"
    "Trends or systematic patterns may indicate model misspecification or missing covariates.",
    wrap=True, ha='center', fontsize=10,color='white'
)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to fit the text
    plt.show()


# Visual Predictive Check (VPC)
## Simulates the model 1000× to check if real data sits within prediction intervals.
## If your model is good, real data should mostly fall inside 5%-95% simulation bands.
def vpc(data, pop_params, n_simulations=1000, percentile_range=(5, 50, 95)):
    all_simulated_concs = []

    # Simulate data for multiple virtual subjects
    for _ in range(n_simulations):
        simulated_concs = []
        for pid, pdata in data.groupby('patient'):
            times, preds = solve_two_compartment(pdata, pop_params)
            simulated_concs.append(preds)

        all_simulated_concs.append(np.concatenate(simulated_concs))

    # Calculate percentiles of the simulated concentrations
    all_simulated_concs = np.concatenate(all_simulated_concs)
    lower_percentile = np.percentile(all_simulated_concs, percentile_range[0])
    median_percentile = np.percentile(all_simulated_concs, percentile_range[1])
    upper_percentile = np.percentile(all_simulated_concs, percentile_range[2])

    # Plot observed vs simulated concentrations (VPC)
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': 'white',
        'axes.titlecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.facecolor': '#222222',
    })
    plt.figure(figsize=(8, 6))

    # Observed data (for reference)
    obs_data = data[data['evid'] == 0]
    plt.scatter(obs_data['time'], obs_data['conc'], color='black', alpha=0.6, label='Observed')

    # Simulated data percentiles
    plt.plot(obs_data['time'], np.repeat(median_percentile, len(obs_data)), 'b--', label='50th Percentile (Median)')
    plt.fill_between(obs_data['time'], lower_percentile, upper_percentile, color='gray', alpha=0.5,
                     label='5th-95th Percentile')

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (ng/mL)")
    plt.title("Visual Predictive Check (VPC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.figtext(
    0.5, -0.1,
    "Figure 4. Visual Predictive Check (VPC).\n"
    "The shaded region (5th–95th percentile) shows model-predicted variability.\n"
    "Most observed data points should lie within this region to indicate good predictive performance.",
    wrap=True, ha='center', fontsize=10,color='white'
)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to fit the text
    plt.show()



###BOX PLOT
# Define the therapeutic range (e.g., 5–10 mg/L)
THERAPEUTIC_RANGE_MIN = 5.0
THERAPEUTIC_RANGE_MAX = 10.0


# Function to calculate the actual concentrations
# Function to calculate the actual concentrations
def simulate_actual_concentrations(data, pop_params):
    all_concentrations = []  # To store concentrations from all patients

    for pid, pdata in data.groupby('patient'):
        # Simulate concentration-time profile based on population parameters
        times, concs = solve_two_compartment(pdata, pop_params)

        # Collect actual concentrations (e.g., concentrations at certain time points)
        all_concentrations.extend(concs)  # Append the concentrations for this patient

    return all_concentrations


# Generate the boxplot of actual concentrations and highlight the therapeutic range
def plot_boxplot_of_concentrations(concentrations):
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.edgecolor': 'white',
        'axes.titlecolor': 'white',
        'legend.edgecolor': 'white',
        'legend.facecolor': '#222222',
    })
    plt.figure(figsize=(8, 6))

    # Create the boxplot
    plt.boxplot(concentrations, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'))

    # Highlight the therapeutic range by adding horizontal lines within the plot
    plt.axhline(y=THERAPEUTIC_RANGE_MIN, color='red', linestyle='--',
                label=f'Therapeutic Range: {THERAPEUTIC_RANGE_MIN} mg/L')
    plt.axhline(y=THERAPEUTIC_RANGE_MAX, color='red', linestyle='--')

    # Title and labels
    plt.title('Distribution of Drug Concentrations')
    plt.xlabel('Patients')
    plt.ylabel('Concentration (ng/mL)')

    # Add grid and legend
    plt.grid(True)
    plt.legend(loc='best')
    plt.figtext(
    0.5, -0.1,
    "Figure 5. Boxplot of simulated vancomycin concentrations.\n"
    "Red dashed lines indicate the therapeutic window (5–10 mg/L).\n"
    "The box shows median, interquartile range (IQR), and spread of predicted concentrations.",
    wrap=True, ha='center', fontsize=10,color='white'
)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to fit the text
    # Show the plot
    plt.show()


# Function to calculate concentrations and plot the boxplot
## Simulates and visualizes how concentrations vary across all patients.
## Highlights therapeutic range (like 5-10 mg/L for vancomycin).
def plot_concentrations_and_boxplot(data, pop_params):
    # Get the actual concentrations from the data and model
    concentrations = simulate_actual_concentrations(data, pop_params)

    # Generate and show the boxplot with therapeutic range
    plot_boxplot_of_concentrations(concentrations)

# Example data
data = pd.read_excel("sample.xlsx")

# Initial guess: V1, V2, CL1, CL2, Q
initial_params = [10, 20, 5, 3, 2] ## Modify if necessary

# Run population parameter estimation
pop_params = estimate_population_parameters(data, initial_params)
print("Estimated Population Parameters:", pop_params)
# Convert to DataFrame for plotting
# Collect observed vs predicted data

# Save to file
with open("pop_params.json", "w") as f:
    json.dump(pop_params.tolist(), f)

#######R^2 error and RMSE
observed_vs_predicted = []

for pid in data['patient'].unique():
    pdata = data[data['patient'] == pid]
    ind_params = estimate_individual_parameters(pdata, pop_params)

    # Get observed time points
    obs_data = pdata[pdata['evid'] == 0]
    obs_times = obs_data['time'].values
    obs_concs = obs_data['conc'].values


    # Simulate at those times
    def simulate_at_obs_times(params):
        sol = solve_ivp(
            fun=two_compartment_ode,
            t_span=[0, max(obs_times)],
            y0=[pdata[pdata['evid'] == 1]['amt'].sum(), 0],
            args=tuple(params),
            t_eval=obs_times
        )
        return sol.y[0] / (params[0] * pdata['weight'].iloc[0] / 70)  # A1 / V1


    pred_concs = simulate_at_obs_times(ind_params)

    for t, obs, pred in zip(obs_times, obs_concs, pred_concs):
        observed_vs_predicted.append({
            'patient': pid,
            'time': t,
            'observed': obs,
            'predicted': pred
        })

ovp_df = pd.DataFrame(observed_vs_predicted)
# R² and RMSE for Individual Fits
## R² (Coefficient of Determination): how well your model explains the observed data.
## RMSE (Root Mean Square Error): how big your average prediction error is.
r2 = r2_score(ovp_df['observed'], ovp_df['predicted'])
rmse = mean_squared_error(ovp_df['observed'], ovp_df['predicted'])

print(f"R²: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")


#######R^2 error and RMSE



# Estimate individual parameters
individuals = []
for pid in data['patient'].unique():
    pdata = data[data['patient'] == pid]
    ind_params = estimate_individual_parameters(pdata, pop_params)
    individuals.append(ind_params)

# Calculate shrinkage
shrinkage = calculate_shrinkage(individuals, pop_params)
print(f"Shrinkage: {shrinkage:.2f}%")

# Plot the observed vs individual predicted concentrations
#plot_observed_vs_individual(data, individuals)
# Plot the observed vs population predicted concentrations
#plot_observed_vs_population(data, pop_params)
#plot_cwres(data, individuals, pop_params)
# Perform Visual Predictive Check
#vpc(data, pop_params, n_simulations=1000, percentile_range=(5, 50, 95))

#box plot
# plot_concentrations_and_boxplot(data, pop_params)





# Function to estimate dose to reach target Cmin
def find_dose_for_target_cmin(target_cmin, patient_data, base_params):
    def objective(dose):
        pdata = patient_data.copy()
        pdata.loc[pdata['evid'] == 1, 'amt'] = dose
        times, concs = solve_two_compartment(pdata, base_params)
        pred_cmin = np.min(concs)
        return (pred_cmin - target_cmin) ** 2

    result = minimize(objective, x0=[1000], bounds=[(100, 3000)])
    return result.x[0] if result.success else None

# Function to calculate AUC from concentration-time profile
def calculate_auc(time_points, concentrations):
    return simps(concentrations, time_points)



###SIMULATE FOR NEW PATIENT
# Assuming you have new patient data
new_patient = pd.DataFrame({
    'patient': [999, 999, 999],
    'time':    [0, 1, 2], #Time (hours) after starting dose
    'amt':     [1000, 0, 0],#Amount of drug administered (mg)
    'evid':    [1, 0, 0],# # 1 = dose given, 0 = observation
    'conc':    [np.nan, np.nan, np.nan],  # No observations, just predictions Observed drug concentration (ng/mL)
    'weight':  [75, 75, 75],
    'scr':     [1.1, 1.1, 1.1],
    'age':     [55, 55, 55],
    'gender':  [0, 0, 0]  # 0 = male
})

# Predict using population parameters
predicted = predict_new_patient(new_patient, pop_params)
print(predicted)

# Optional: plot prediction
# plt.plot(predicted['time'], predicted['pred_conc'], marker='o', label='Predicted Concentration')
# plt.xlabel("Time (h)")
# plt.ylabel("Concentration")
# plt.title("Prediction for New Patient")
# plt.grid(True)
# plt.legend()
# plt.show()
###SIMULATE FOR NEW PATIENT

# Choose target Cmin
target_cmin = 10.0  # mg/L

def simulate_dose_for_target_cmin(target_cmin, base_patient_data, pop_params):
    # Clone the input patient data to avoid modifying original
    patient = base_patient_data.copy()

    # Step 1: Estimate the dose
    estimated_dose = find_dose_for_target_cmin(target_cmin, patient, pop_params)
    print(f"Estimated dose to reach Cmin {target_cmin} mg/L: {estimated_dose:.2f} mg")

    # Step 2: Apply the dose to the patient's dosing row
    patient.loc[patient['evid'] == 1, 'amt'] = estimated_dose

    # Step 3: Simulate the model with the estimated dose
    times, concs = solve_two_compartment(patient, pop_params)

    # Step 4: Calculate AUC
    auc = calculate_auc(times, concs)
    print(f"Predicted AUC: {auc:.2f} mg·h/L")

    # Step 5: Plot result
    plt.plot(times, concs, label='Predicted Concentration', marker='o')
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (mg/L)")
    plt.title(f"New Patient Simulation\nCmin Target: {target_cmin} mg/L | AUC: {auc:.2f} mg·h/L")
    plt.axhline(target_cmin, color='red', linestyle='--', label='Target Cmin')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

##Run the simulation for the new patient
# simulate_dose_for_target_cmin(target_cmin=10.0, base_patient_data=new_patient, pop_params=pop_params)
