import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# EV and Environment Specifications (Matching Training Code)
EV_CAPACITY = 5.0  # in kWh (Same as E_BB in training code)
SOC_MIN = 0.2 * EV_CAPACITY  # Minimum State of Charge (1.0 kWh)
SOC_MAX = 0.8 * EV_CAPACITY  # Maximum State of Charge (4.0 kWh)
CHARGER_POWER = 5.0  # Maximum charging power in kW
EV_MAX_DISCHARGE_POWER = 5.0  # Maximum discharging power in kW

# Charging efficiencies (Matches training code)
ETA_CH = 0.95  # Charging efficiency
ETA_DIS = 0.95  # Discharging efficiency

# Define Maximum Charging and Discharging Powers
MAX_GRID_CHARGE_POWER = 5.0  # Maximum power from grid to charge battery in kW
MAX_BATTERY_CHARGE_POWER = 5.0  # Maximum battery charging power in kW
MAX_BATTERY_DISCHARGE_POWER = 5.0  # Maximum battery discharging power in kW

# Initialize charging state
charging_in_progress = False  # Global flag to indicate if EV is charging
charged_today = False  # Flag to indicate if EV has charged today
current_day = None  # Variable to track the current day
charging_duration = 0  # Duration of continuous charging (in time steps)

# Function to determine if EV is available
def is_ev_available(day, hour):
    return True  # EV (Battery) is always available

# Define the process_action function with the specified constraints
def process_action(action, load, pv, tou_tariff, fit, soc, ev_available, day, hour):
    global charging_in_progress, charging_duration, charged_today, current_day  # Access global variables

    # Initialize allocations
    allocations = {
        'PV to Load': 0.0,
        'PV to EV': 0.0,
        'PV to Grid': 0.0,
        'EV to Load': 0.0,
        'EV to Grid': 0.0,  # This will remain zero
        'Grid to Load': 0.0,
        'Grid to EV': 0.0
    }

    # Priority 1: Use PV to meet Load
    allocations['PV to Load'] = min(pv, load)
    load_remaining = load - allocations['PV to Load']

    # Calculate PV remaining after supplying to load
    pv_remaining = pv - allocations['PV to Load']

    # Action mapping
    # 0: Do nothing
    # 1: Charge battery
    # 2: Discharge battery

    P_BB_ch = 0.0
    P_BB_dis = 0.0

    # Conditions for Grid Charging
    can_charge_from_grid = (
        soc <= SOC_MIN and
        not charged_today and
        pv == 0
    )

    if action == 1 and soc < SOC_MAX:
        # Calculate maximum possible charging power
        P_BB_ch_max = min((SOC_MAX - soc) * EV_CAPACITY / ETA_CH, MAX_BATTERY_CHARGE_POWER)

        # Charge battery with available PV
        P_BB_ch_from_PV = min(pv_remaining, P_BB_ch_max)
        P_BB_ch += P_BB_ch_from_PV
        allocations['PV to EV'] = P_BB_ch_from_PV
        pv_remaining -= P_BB_ch_from_PV

        # Charge battery from Grid if conditions are met
        if P_BB_ch < P_BB_ch_max and can_charge_from_grid:
            P_BB_ch_from_Grid = min(P_BB_ch_max - P_BB_ch, MAX_GRID_CHARGE_POWER)
            P_BB_ch += P_BB_ch_from_Grid
            allocations['Grid to EV'] = P_BB_ch_from_Grid
            charged_today = True
            charging_duration += 1
    elif action == 2 and soc > SOC_MIN:
        # Determine if it's peak hours for peak shaving
        is_peak_hour = 17 <= hour <= 20  # Peak hours from 5 PM to 8 PM

        # Allow discharging during peak hours or if load exceeds average load
        if is_peak_hour or load > 1.470597768:
            # Calculate maximum possible discharging power
            P_BB_dis_max = min((soc - SOC_MIN) * EV_CAPACITY * ETA_DIS, MAX_BATTERY_DISCHARGE_POWER)
            P_BB_dis = min(P_BB_dis_max, load_remaining)
            allocations['EV to Load'] = P_BB_dis
            load_remaining -= P_BB_dis
    else:
        # Do nothing
        pass

    # Update SoC
    delta_SoC = (ETA_CH * P_BB_ch - P_BB_dis / ETA_DIS) / EV_CAPACITY
    soc += delta_SoC
    soc = max(SOC_MIN, min(soc, SOC_MAX))  # Enforce SoC limits

    # Any remaining PV goes to Grid
    if pv_remaining > 0:
        allocations['PV to Grid'] = pv_remaining

    # Remaining load met by Grid
    if load_remaining > 0:
        allocations['Grid to Load'] = load_remaining

    # Calculate power flows for cost calculation
    P_grid = allocations['Grid to Load'] + allocations['Grid to EV']
    P_export = allocations['PV to Grid'] + allocations['EV to Grid']  # allocations['EV to Grid'] is always zero

    # Calculate Purchase, Sell, and Bill
    purchase = P_grid * tou_tariff
    sell = P_export * fit  # Use FiT for selling price
    bill = purchase - sell

    return soc, allocations, purchase, sell, bill

# Paths (Update these paths based on your file locations)
model_path = 'dqn_energy_model.h5'  # Path to your trained model
dataset_path = 'dataset.csv'  # Path to your dataset

# Load the trained DQN model without compiling
def weighted_mse(y_true, y_pred):
    w = y_true[:, -1]
    y_true = y_true[:, :-1]
    w = tf.expand_dims(w, axis=-1)
    return tf.reduce_mean(w * tf.square(y_true - y_pred))

custom_objects = {'weighted_mse': weighted_mse}
model = load_model(model_path, custom_objects=custom_objects, compile=False)
print("Trained DQN model loaded successfully.")

# Load the dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")

# Display the first few rows to verify
print("\nInitial Dataset Preview:")
print(df.head())

# Define the columns to fill
columns_to_fill = [
    'PV to Load',
    'PV to EV',
    'PV to Grid',
    'EV to Load',
    'EV to Grid',
    'Grid to Load',
    'Grid to EV',
    'Purchase',
    'Sell',
    'Bill'
]

# Initialize columns with NaN if they don't exist
for col in columns_to_fill:
    if col not in df.columns:
        df[col] = np.nan

print("\nColumns initialized for filling:")
print(df[columns_to_fill].head())

# Precompute maximum values for normalization
load_max = df['Load'].max()
pv_max = df['PV'].max()
tou_tariff_max = df['Tou Tariff'].max()
day_max = df['Day'].max()
hour_max = 23  # Assuming Hour ranges from 0 to 23

# Initialize SoC
soc = SOC_MAX  # Starting at maximum SoC

# Iterate through each row to fill in the columns
for index, row in df.iterrows():
    load = row['Load']        # in kW
    pv = row['PV']            # in kW
    tou_tariff = row['Tou Tariff']  # $/kWh
    fit = row['FiT']          # $/kWh
    day = int(row['Day'])     # Ensure integer type
    hour = int(row['Hour'])   # Ensure integer type

    # Determine if EV is available
    ev_available = is_ev_available(day, hour)

    # Normalize features for the state vector
    load_norm = load / load_max if load_max != 0 else 0
    pv_norm = pv / pv_max if pv_max != 0 else 0
    tou_tariff_norm = tou_tariff / tou_tariff_max if tou_tariff_max != 0 else 0
    day_norm = day / day_max if day_max != 0 else 0
    hour_norm = hour / hour_max if hour_max != 0 else 0

    # Construct the state vector (Matching Training Code)
    state = np.array([
        soc / EV_CAPACITY,  # Normalize SoC
        load_norm,          # Normalized Load
        pv_norm,            # Normalized PV
        tou_tariff_norm,    # Normalized Tou Tariff
        hour_norm,          # Normalized Hour
        day_norm            # Normalized Day
    ])

    # Determine feasible actions based on SoC constraints
    feasible_actions = [0]  # Do nothing is always feasible

    if soc < SOC_MAX:
        feasible_actions.append(1)  # Charge battery
    if soc > SOC_MIN:
        feasible_actions.append(2)  # Discharge battery

    # Predict action using the DQN model
    action_probs = model.predict(state.reshape(1, -1), verbose=0)
    action = np.argmax([action_probs[0][a] if a in feasible_actions else -np.inf for a in range(3)])

    # Update charged_today flag at the start of a new day
    if current_day != day:
        current_day = day
        charged_today = False
        charging_duration = 0  # Reset charging duration at the start of the day

    # Process the action to get updated SoC, allocations, Purchase, Sell, and Bill
    soc, allocations, purchase, sell, bill = process_action(
        action, load, pv, tou_tariff, fit, soc, ev_available, day, hour
    )

    # Fill in the calculated values into the DataFrame
    df.at[index, 'PV to Load'] = allocations['PV to Load']
    df.at[index, 'PV to EV'] = allocations['PV to EV']
    df.at[index, 'PV to Grid'] = allocations['PV to Grid']
    df.at[index, 'EV to Load'] = allocations['EV to Load']
    df.at[index, 'EV to Grid'] = allocations['EV to Grid']
    df.at[index, 'Grid to Load'] = allocations['Grid to Load']
    df.at[index, 'Grid to EV'] = allocations['Grid to EV']
    df.at[index, 'Purchase'] = purchase
    df.at[index, 'Sell'] = sell
    df.at[index, 'Bill'] = bill

    # Optional: Print progress every 100 rows
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} rows.")

# Display the updated DataFrame
print("\nUpdated Dataset Preview:")
print(df.head())

# Define output path
output_path = 'dataset_filled_friday_normal.csv'  # Change as needed

# Save the updated DataFrame to a new CSV file
df.to_csv(output_path, index=False)

print(f"\nUpdated dataset saved to '{output_path}'.")

# Visualization (Optional)
# Set the style for seaborn
sns.set(style="whitegrid")

# Plot distributions of Purchase, Sell, and Bill
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['Purchase'], bins=50, kde=True, color='blue')
plt.title('Distribution of Purchase')
plt.xlabel('Purchase ($)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(df['Sell'], bins=50, kde=True, color='green')
plt.title('Distribution of Sell')
plt.xlabel('Sell ($)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(df['Bill'], bins=50, kde=True, color='red')
plt.title('Distribution of Bill')
plt.xlabel('Bill ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('financial_distributions.png')
plt.show()

# Plot distributions of Energy Allocations
energy_allocations = [
    'PV to Load',
    'PV to EV',
    'PV to Grid',
    'EV to Load',
    'EV to Grid',
    'Grid to Load',
    'Grid to EV'
]

plt.figure(figsize=(14, 7))
for allocation in energy_allocations:
    sns.histplot(df[allocation], bins=50, kde=True, label=allocation, alpha=0.6)
plt.title('Distribution of Energy Allocations')
plt.xlabel('Energy (kW)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('energy_allocations_distributions.png')
plt.show()
