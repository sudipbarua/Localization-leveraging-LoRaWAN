# Import necessary libraries
import time
import random
import numpy as np
from docplex.mp.model import Model # Import docplex
from datetime import datetime, timezone, timedelta
import re
import json
import pandas as pd


def offline_log_synchronization(recorded_data, num_events, num_gateways):
    """
    Performs offline log synchronization using a linear programming approach
    to estimate true event times, clock offsets, and clock rates.

    This implementation follows the Maximum Likelihood Estimator (MLE)
    formulation described in the paper "On the Time Synchronization of
    Distributed Log Files in Networks With Local Broadcast Media".

    Args:
        recorded_data (list of dict): A list of recorded observations.
            Each dictionary in the list must contain:
            - 'event_idx' (int): The index of the unique event (0 to num_events-1).
            - 'gateway_idx' (int): The index of the unique gateway (0 to num_gateways-1).
            - 'timestamp' (float): The timestamp recorded by the gateway for this event.
        num_events (int): The total number of unique events observed across all logs.
        num_gateways (int): The total number of unique gateways involved.

    Returns:
        tuple: (estimated_T, estimated_o_bar, estimated_r_bar) if successful,
               (None, None, None) otherwise.
            - estimated_T (np.array): A 1D array of estimated true times for each event.
            - estimated_o_bar (np.array): A 1D array of estimated transformed offsets
                                        (o_j / r_j) for each gateway.
            - estimated_r_bar (np.array): A 1D array of estimated inverse rates
                                        (1 / r_j) for each gateway.
    """

    # Create a new docplex model
    mdl = Model(name='log_synchronization')

    # --- 1. Define decision variables for the optimization problem ---
    # T_hat_i: Estimated true event times (continuous, no lower bound in the paper's formulation)
    t_hat_vars = [mdl.continuous_var(name=f'T_hat_{i}') for i in range(num_events)]

    # o_bar_j: Estimated transformed offsets (o_j / r_j) (continuous, no lower bound)
    o_bar_vars = [mdl.continuous_var(name=f'o_bar_{j}') for j in range(num_gateways)]

    # r_bar_j: Estimated inverse rates (1 / r_j) (continuous, must be strictly positive)
    # Using a small epsilon (1e-9) as a lower bound for numerical stability.
    r_bar_vars = [mdl.continuous_var(name=f'r_bar_{j}', lb=1e-9) for j in range(num_gateways)]

    # --- 2. Construct the objective function ---
    # The objective is to minimize sum_{(i,j) in R} (t_i,j * r_bar_j - o_bar_j - T_hat_i)
    objective = mdl.sum(
        obs['timestamp'] * r_bar_vars[obs['gateway_idx']] - o_bar_vars[obs['gateway_idx']] - t_hat_vars[obs['event_idx']]  
        for obs in recorded_data
    )

    mdl.minimize(objective)


    # --- 3. Construct inequality constraints (d_i,j >= 0) ---
    # Constraints are t_i,j * r_bar_j - o_bar_j - T_hat_i >= 0
    # In docplex, this is directly expressed.
    for obs in recorded_data:
        event_idx = obs['event_idx']
        gateway_idx = obs['gateway_idx']
        timestamp = obs['timestamp']

        # Constraint: T_recorded_i,j - (o_j + r_j * T_true_i) >= 0
        # Using the transformed variables: t_i,j - (o_bar_j * r_j + T_hat_i * r_j) >= 0
        # This is not quite right... the paper's formulation is d_i,j = t_i,j * r_bar_j - o_bar_j - T_hat_i >= 0
        mdl.add_constraint(
            timestamp * r_bar_vars[gateway_idx] - o_bar_vars[gateway_idx] - t_hat_vars[event_idx] >= 0
        )


    # --- 4. Construct equality constraints (to resolve ambiguities) ---

    # Constraint 1: Rate normalization
    # Sum_{j in J} r_bar_j = |J|
    mdl.add_constraint(
        mdl.sum(r_bar_vars[j] for j in range(num_gateways)) == num_gateways
    )

    # Constraint 2: Offset ambiguity resolution
    # o_bar_0 = 0 (fix the transformed offset of the first gateway to 0 as a reference)
    # This gateway index (0) is an arbitrary choice.
    mdl.add_constraint(o_bar_vars[0] == 0)


    # --- 5. Solve the Linear Program using Docplex ---
    print(f"Starting LP optimization with Docplex for {mdl.number_of_variables} variables and {mdl.number_of_constraints} constraints...")

    # Solve the model. Docplex will automatically use the available CPLEX runtime.
    solution = mdl.solve()


    if solution:
        print("LP optimization successful.")
        # Extract the estimated parameters from the solution
        estimated_T = np.array([v.solution_value for v in t_hat_vars])
        estimated_o_bar = np.array([v.solution_value for v in o_bar_vars])
        estimated_r_bar = np.array([v.solution_value for v in r_bar_vars])

        return estimated_T, estimated_o_bar, estimated_r_bar
    else:
        print(f"LP optimization failed: {mdl.get_solve_status()}")
        return None, None, None


if __name__ == "__main__":
    # Loading dataset
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/tuc_lora_metadata.mqtt_data_22-27_4gw.json') as file1:
        ds_json = json.load(file1)
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/tuc_lora_gateways.json') as file2:
        gw_loc = json.load(file2)

    # The GW IDs in the dataset are strings of unique identifiers,
    # we need to convert them to integers for processing
    gw_id_to_int = {gw_id: i for i, gw_id in enumerate(gw_loc)}
    int_to_gw_id = {idx+1: gw_id for idx, gw_id in enumerate(gw_loc.keys())}
    for i in ds_json:
        for j in i['rxInfo']:
            j['gatewayId'] = gw_id_to_int[j['gatewayId']]


    # Extracting recorded data from the dataset
    recorded_data = []
    for idx, event in enumerate(ds_json):
        for rx_info in event['rxInfo']:
            record = {}
            record['event_idx'] = idx
            record['gateway_idx'] = rx_info['gatewayId']
            
            # the timestamp since epoch are to long int values. So we set the 1st timestamp as the reference and
            # calculate the difference in seconds between the first event and the current event ::: Relative timestamp
            tsFirstEvent = pd.Timestamp(ds_json[0]['rxInfo'][0]['nsTime']).value
            tsCurrentEvent = pd.Timestamp(rx_info['nsTime']).value
            record['timestamp'] = ( tsFirstEvent - tsCurrentEvent ) / 1e9 + 1
            record['timestamp_actual_ns'] = tsCurrentEvent
            recorded_data.append(record)


    # Run the offline synchronization algorithm
    estimated_T, estimated_o_bar, estimated_r_bar = offline_log_synchronization(
        recorded_data=recorded_data, num_events=len(ds_json), num_gateways=len(gw_loc)
    )

    print("Estimated True Event Times (T):")

