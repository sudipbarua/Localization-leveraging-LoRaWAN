# Import necessary libraries
import numpy as np
# from scipy.optimize import linprog # No longer needed if using docplex
from collections import defaultdict
import random
from docplex.mp.model import Model # Import docplex

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

    # Combine variables into a single list for easier indexing if needed,
    # but docplex works directly with the created variables.
    # x_vars = t_hat_vars + o_bar_vars + r_bar_vars

    # --- 2. Construct the objective function ---
    # The objective is to minimize sum_{(i,j) in R} (t_i,j * r_bar_j - o_bar_j - T_hat_i)
    # which is equivalent to maximizing sum_{(i,j) in R} (T_hat_i + o_bar_j - t_i,j * r_bar_j)

    objective = mdl.sum(
        t_hat_vars[obs['event_idx']] + o_bar_vars[obs['gateway_idx']] - obs['timestamp'] * r_bar_vars[obs['gateway_idx']]
        for obs in recorded_data
    )

    mdl.maximize(objective)


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

# --- Helper function for simulating realistic data for testing ---
def generate_simulated_data(num_events, num_gateways, avg_true_time_diff=1.0,
                            max_offset_s=0.1, max_rate_ppm=100.0, avg_delay_s=1e-4,
                            observation_probability=0.8):
    """
    Generates simulated log data for events observed by multiple gateways.
    Includes realistic clock drift and timestamping delays.

    Args:
        num_events (int): Number of distinct events to simulate.
        num_gateways (int): Number of gateways.
        avg_true_time_diff (float): Average time difference between consecutive true events (seconds).
        max_offset_s (float): Maximum initial clock offset (seconds).
        max_rate_ppm (float): Maximum clock rate deviation in parts per million (ppm).
        avg_delay_s (float): Average timestamping delay (lambda^-1 for exponential) in seconds.
        observation_probability (float): Probability that a gateway observes a given event.

    Returns:
        tuple: (recorded_data, true_params_dict)
            - recorded_data (list of dict): Simulated observations ready for the LP solver.
            - true_params_dict (dict): Dictionary of true parameters used for simulation,
                                      useful for comparing with LP results.
    """
    random.seed(42) # For reproducibility of random choices
    np.random.seed(42) # For reproducibility of numpy random numbers

    # 1. Generate True Event Times
    # Start at true_time = 0. Events occur at increasing intervals.
    true_event_times = np.cumsum(np.random.rand(num_events) * avg_true_time_diff)

    # 2. Generate True Clock Parameters for each Gateway
    # Clock rates (r_j): Base rate is 1.0, with small random deviation
    # 1 ppm = 1e-6 deviation from ideal clock.
    true_rates = 1.0 + (np.random.rand(num_gateways) * 2 - 1) * max_rate_ppm / 1e6

    # Clock offsets (o_j): Random initial offsets
    true_offsets = (np.random.rand(num_gateways) * 2 - 1) * max_offset_s

    # Calculate true transformed parameters for later comparison
    true_r_bar = 1.0 / true_rates
    true_o_bar = true_offsets / true_rates

    recorded_data = []
    # (Optional) Store true delays if needed for specific analysis, but not directly used by the solver.
    # true_delays_per_observation = []

    # 3. Simulate Recorded Timestamps
    for i in range(num_events):
        for j in range(num_gateways):
            # Simulate if this gateway observes the event
            if random.random() < observation_probability:
                # Timestamping delay (d_i,j): Modeled as exponentially distributed
                # In real-world, might have a minimum delay, but exponential implies 0 is possible.
                delay = np.random.exponential(scale=avg_delay_s)
                # true_delays_per_observation.append(delay)

                # Calculate recorded timestamp based on the clock model:
                # T_recorded = C_j(T_true_event + d_i,j) = r_j * (T_true_event + d_i,j) + o_j
                recorded_timestamp = true_rates[j] * (true_event_times[i] + delay) + true_offsets[j]

                recorded_data.append({
                    'event_idx': i,
                    'gateway_idx': j,
                    'timestamp': recorded_timestamp
                })

    true_params = {
        'true_event_times': true_event_times,
        'true_rates': true_rates,
        'true_offsets': true_offsets,
        'true_r_bar': true_r_bar,
        'true_o_bar': true_o_bar,
        # 'true_delays': true_delays_per_observation # Uncomment if needed
    }

    return recorded_data, true_params

# --- Example Usage ---
if __name__ == "__main__":
    # Define simulation parameters
    NUM_EVENTS = 5000  # Number of simulated packet arrivals
    NUM_GATEWAYS = 10 # Number of LoRaWAN gateways
    AVG_TRUE_TIME_DIFF = 0.5 # seconds between events
    MAX_OFFSET_S = 1.0 # Max initial clock offset (e.g., up to 1 second)
    MAX_RATE_PPM = 500.0 # Max clock rate deviation (e.g., 500 ppm, significant)
    AVG_DELAY_S = 50e-6 # Average timestamping delay (50 microseconds)
    OBSERVATION_PROB = 0.9 # Probability a gateway successfully logs an event

    print("--- Starting Simulation and Synchronization ---")
    print(f"Simulating {NUM_EVENTS} events and {NUM_GATEWAYS} gateways...")
    print(f"Average timestamping delay (lambda^-1): {AVG_DELAY_S*1e6:.2f} us")
    print(f"Max clock rate deviation: {MAX_RATE_PPM} ppm")

    # Generate simulated data
    recorded_data_sim, true_params_sim = generate_simulated_data(
        NUM_EVENTS, NUM_GATEWAYS, AVG_TRUE_TIME_DIFF,
        MAX_OFFSET_S, MAX_RATE_PPM, AVG_DELAY_S, OBSERVATION_PROB
    )
    print(f"Simulated data generated. Total recorded observations: {len(recorded_data_sim)}")

    # Run the offline synchronization algorithm
    estimated_T, estimated_o_bar, estimated_r_bar = offline_log_synchronization(
        recorded_data_sim, NUM_EVENTS, NUM_GATEWAYS
    )

    if estimated_T is not None:
        print("\n--- Comparing Estimated Parameters with True Values ---")

        # Due to the normalization constraints (Sum(r_bar)=|J|, o_bar_0=0),
        # the LP solution provides relative values. To compare with the 'true'
        # parameters (which are on an arbitrary absolute scale), we need to
        # apply corresponding transformations to the estimated values.

        # 1. Adjust Estimated True Event Times (T_hat_i)
        # We align the first estimated true event time with the first true event time.
        # This resolves the global offset ambiguity.
        # Note: The LP implicitly chooses a time origin, this adjustment makes comparison easier.
        initial_t_hat_offset = estimated_T[0] - true_params_sim['true_event_times'][0]
        estimated_T_adjusted = estimated_T - initial_t_hat_offset

        # 2. Adjust Estimated Transformed Offsets (o_bar_j)
        # The LP fixed o_bar_0 = 0. We adjust based on the true o_bar_0.
        o_bar_adjustment = estimated_o_bar[0] - true_params_sim['true_o_bar'][0]
        estimated_o_bar_adjusted = estimated_o_bar - o_bar_adjustment

        # 3. Adjust Estimated Inverse Rates (r_bar_j)
        # The LP fixed Sum(r_bar_j) = |J|. We adjust based on the true average inverse rate.
        r_bar_scaling_factor = np.mean(true_params_sim['true_r_bar']) / np.mean(estimated_r_bar)
        estimated_r_bar_adjusted = estimated_r_bar * r_bar_scaling_factor

        # --- Display Results ---
        print("\nEstimated True Event Times (T_hat_i) vs True (first 5 events):")
        for i in range(min(5, NUM_EVENTS)):
            print(f"  Event {i}: True={true_params_sim['true_event_times'][i]:.6f}, Est={estimated_T_adjusted[i]:.6f}, Error={estimated_T_adjusted[i] - true_params_sim['true_event_times'][i]:.6e}")

        max_t_error = np.max(np.abs(estimated_T_adjusted - true_params_sim['true_event_times']))
        print(f"Max absolute error in Estimated True Event Times: {max_t_error:.6e} s")

        print("\nEstimated Transformed Offsets (o_bar_j) vs True:")
        for j in range(min(5, NUM_GATEWAYS)):
            print(f"  GW {j}: True o_bar={true_params_sim['true_o_bar'][j]:.6f}, Est o_bar={estimated_o_bar_adjusted[j]:.6f}, Error={estimated_o_bar_adjusted[j] - true_params_sim['true_o_bar'][j]:.6e}")
        max_o_bar_error = np.max(np.abs(estimated_o_bar_adjusted - true_params_sim['true_o_bar']))
        print(f"Max absolute error in Estimated Transformed Offsets: {max_o_bar_error:.6e}")

        print("\nEstimated Inverse Rates (r_bar_j) vs True:")
        for j in range(min(5, NUM_GATEWAYS)):
            print(f"  GW {j}: True r_bar={true_params_sim['true_r_bar'][j]:.6f}, Est r_bar={estimated_r_bar_adjusted[j]:.6f}, Error={estimated_r_bar_adjusted[j] - true_params_sim['true_r_bar'][j]:.6e}")
        max_r_bar_error = np.max(np.abs(estimated_r_bar_adjusted - true_params_sim['true_r_bar']))
        print(f"Max absolute error in Estimated Inverse Rates: {max_r_bar_error:.6e}")

        # Derive and compare original clock parameters (rates and offsets)
        estimated_rates_derived = 1.0 / estimated_r_bar_adjusted
        estimated_offsets_derived = estimated_o_bar_adjusted / estimated_r_bar_adjusted

        print("\nDerived Estimated Clock Rates (r_j) vs True:")
        for j in range(min(5, NUM_GATEWAYS)):
            print(f"  GW {j}: True r={true_params_sim['true_rates'][j]:.6f}, Est r={estimated_rates_derived[j]:.6f}, Error={estimated_rates_derived[j] - true_params_sim['true_rates'][j]:.6e}")
        max_rate_error = np.max(np.abs(estimated_rates_derived - true_params_sim['true_rates']))
        print(f"Max absolute error in Derived Clock Rates: {max_rate_error:.6e}")

        print("\nDerived Estimated Clock Offsets (o_j) vs True:")
        for j in range(min(5, NUM_GATEWAYS)):
            print(f"  GW {j}: True o={true_params_sim['true_offsets'][j]:.6f}, Est o={estimated_offsets_derived[j]:.6f}, Error={estimated_offsets_derived[j] - true_params_sim['true_offsets'][j]:.6e}")
        max_offset_error = np.max(np.abs(estimated_offsets_derived - true_params_sim['true_offsets']))
        print(f"Max absolute error in Derived Clock Offsets: {max_offset_error:.6e} s")

    else:
        print("Synchronization failed, no results to display.")

    print("\n--- Important Considerations ---")
    print("1. This is a general Python implementation using SciPy's linear programming solver (`linprog`).")
    print("2. For very large-scale problems (e.g., millions of events and hundreds of gateways), a specialized solver (like the one mentioned in the paper) that explicitly exploits the sparse matrix structure and highly optimized interior-point methods would likely be significantly faster and more memory-efficient than generic `scipy.optimize.linprog`.")
    print("3. The 'adjusted' values for T, o_bar, and r_bar are calculated here *after* the LP solution, solely for easier comparison with the 'true' simulated values. The linear program internally solves for values subject to its own chosen normalization constraints (sum of r_bar = |J|, o_bar_0 = 0).")
    print("4. The quality of synchronization depends heavily on the quantity and quality (e.g., how reliably 'shared events' can be identified across gateways) of the input data.")