"""
This is a TDoA based estimator or position solver based on levenberg and marquadt least-square esitmator
Levenberg-Marquardt least square-
 - Doesn’t handle bounds and sparse Jacobians.
 - Usually the most efficient method for small unconstrained problems.
 - It combines the Gauss Newton and Gradient descent methods.
"""

import json
import numpy as np
import scipy.optimize as opt
from datetime import datetime
import os

# Importing native libraries
import sys
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from tdoa_position_solver.estimator import Estimator
from general_functions import *   # We are going to use the 'map_plot_cartopy' method 
from RSSI_fingerprinting_TDoA_Estimation.performance_eval import calculate_pairwise_error_list

class Levenberg_marquadt_Estimator(Estimator):
    def __init__(self, reference_position, result_directory):
        # Speed of propagation (m/s)
        super().__init__(reference_position, result_directory)
        self.speed = 3e8
        self.ref_pos = reference_position
        self.result_dir = result_directory

    def estimate(self, packet, gateways, plot=False):
        self.get_gateway_positions_toa(packet, gateways)
        
        # Initital position for estimation: mean of the gateway coordinates 
        init_pos = np.mean(self.gateway_positions, axis=0) 

        if len(self.gateway_positions) >= 3:
            # The timestamps or the TOAs are string values so we convert them to Pandas tmestamp object
            ts = np.asarray([pd.Timestamp(t) for t in self.toa])
            # We also create a list called measurements
            # This contains the recieving gateway positions and the TOA
            measurements = np.c_[self.gateway_positions, ts]

            # Define functions and jacobian
            F = self.function(measurements)
            J = self.jacobian(measurements)
            lower_bounds = [init_pos[0] - 5000, init_pos[1] - 5000, 0]
            upper_bounds = [init_pos[0] + 5000, init_pos[1] + 5000, 1000]
            # Perform least squares optimization
            result = opt.least_squares(fun=F, jac=J, x0=init_pos[:2], bounds=(lower_bounds, upper_bounds), method='lm')        # For 2D positions

            print(f"Optimized (x, y, z): ({result.x})")
            # Estimated lat-lon
            lat_est, lon_est, alt_est = pm.enu2geodetic(e=result.x[0], n=result.x[1], u=0, **self.ref_pos)

            if plot==True:
                # Creating a list of results for plotting in the map
                results = {
                    'lat': [lat_est] + [self.gw_lat_lon[i][0] for i in range(len(self.gw_lat_lon))],
                    'lon': [lon_est] + [self.gw_lat_lon[i][1] for i in range(len(self.gw_lat_lon))],
                    'cat': [f'Estimated Pos of {packet['deviceInfo']['deviceName']}'] + [f'GW Positions' for i in range(len(self.gw_lat_lon))]
                }
                # map plot for individual predictions and estimations 
                directory = os.path.join(self.result_dir, 'map_plot')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_path = os.path.join(directory, f"{packet['deduplicationId']}.png")
                map_plot_cartopy(results, path=file_path)

        else:
            print('Position cannot be resolved. Not enough gws to for TDoA measurement')
            lat_est, lon_est, alt_est = None, None, None

        return [lat_est, lon_est, alt_est]


def main():
    ##################### Testing script ###########################
    with open('../data/tuc_lora_metadata.mqtt_data_20-22nov.json') as file1:
    # with open('data/tuc_lora_metadata.mqtt_data_20-22nov.json') as file1:         # for VScode
        ds_json = json.load(file1)
    # ds_json = pd.read_json('data/tuc_lora_metadata.mqtt_data_13-11.json')
    # gw_loc = pd.read_json('data/lorawan_antwerp_gateway_locations.json')
    # with open('data/tuc_lora_gateways.json') as file2:                            # for VScode
    with open('../data/tuc_lora_gateways.json') as file2:
        gw_loc = json.load(file2) 

    ref_pos = {'lat0': 50.814131,
            'lon0': 12.928044,
            'h0': 320}
    result_directory = f'results/LM_lstsq_2D_bounded/{datetime.now().strftime('%Y-%m-%d_%H-%M')}'
    # Ensure the result_directory exists
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    estimator = Levenberg_marquadt_Estimator(reference_position=ref_pos,
                          result_directory=result_directory)

    estimated_position = []
    for packet in ds_json:
        est = estimator.estimate(packet=packet,
                                 gateways=gw_loc, 
                                 plot=True)
        estimated_position.append(est)

    estimated_position = pd.DataFrame(estimated_position, columns=['lat', 'lon', 'alt'])

    # Save results
    file_path = os.path.join(result_directory, f"estimated_positions.csv")
    # Open or create the file
    with open(file_path, 'w') as file:
        estimated_position.to_csv(file)

    # Getting statistical information of the results 
    est_copy = estimated_position.copy()
    # Converting all the nonnumeric values to nan
    est_copy['lat'] = pd.to_numeric(estimated_position['lat'], errors='coerce')
    est_copy['lon'] = pd.to_numeric(estimated_position['lon'], errors='coerce')

    r_woNan = est_copy.dropna(subset=['lat'])
    actual_pos_2 = r_woNan[['lat', 'lon']].to_numpy()  # *****************add the actual coordinates ********************
    est_pos = r_woNan[['lat', 'lon']].to_numpy()

    estimation_error = calculate_pairwise_error_list(ground_truth=actual_pos_2, predictions=est_pos)

    # y = [x for x in estimation_error if x < 1000]
    y = estimation_error

    print(f'min {min(y)}')
    print(f'max {max(y)}')
    print(f'mean {np.mean(y)}')
    print(f'median {np.median(y)}')

if __name__=='__main__':
    main()
