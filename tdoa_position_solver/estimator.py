"""
This is a TDoA based estimator or position solver.
Implementing a child classes of the Least square estimator from "project_directory/RSSI_fingerprinting_TDoA_Estimation"
The main function and the jacobian of the master implementation remains the same. Although, here the 3rd dimension: z or altitude is added 
In this version we will make the data persing more modularized and suitable for reading the mqtt json data from chirpstack   
"""

import json
import numpy as np
import scipy.optimize as opt
from datetime import datetime
import os

# Importing native libraries
import sys
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from RSSI_fingerprinting_TDoA_Estimation.leastsq_estimator import Least_square_estimator
from general_functions import *   # We are going to use the 'map_plot_cartopy' method 
from RSSI_fingerprinting_TDoA_Estimation.performance_eval import calculate_pairwise_error_list

class Estimator():
    def __init__(self, reference_position, result_directory):
        # Speed of propagation (m/s)
        self.speed = 3e8
        self.ref_pos = reference_position
        self.result_dir = result_directory

    # method to generate the residuals for all hyperbolae
    def function(self, measurements):
        def fn(args):
            # Here the args are the arguments passed to the leastsq estimator method
            x, y = args[:2]  # Extract x, y and z coordinates from args
            residuals = []
            for i in range(1, len(measurements)):
                xi, yi, _, ti = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                t0 = measurements[0][3]
                # We use the pandas timestamp method in this case,
                # because it is the only one that can handle precision upto nanosecond
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(t0).value) * 1e-9  # the values are converted to seconds
                di = diff_seconds * self.speed
                ai = np.sqrt((x - xi)**2 + (y - yi)**2) - np.sqrt((x - x0)**2 + (y - y0)**2) - abs(di)                                 # for 2D
                residuals.append(ai)
            return residuals
        return fn

    # Function to generate the Jacobian matrix
    def jacobian(self, measurements):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            jac = []
            for i in range(1, len(measurements)):
                xi, yi, _, _ = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                adx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2) - (x - x0) / np.sqrt((x - x0)**2 + (y - y0)**2)
                ady = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2) - (y - y0) / np.sqrt((x - x0)**2 + (y - y0)**2)
                jac.append([adx, ady])
            return np.array(jac)
        return fn

    def get_gateway_positions_toa(self, packet, gateways):
        self.gateway_positions = []  # ENU coordinates
        self.gw_lat_lon = []         # Lat lon values
        self.toa = []

        for rxinfo in packet['rxInfo']:
            # gw_time = rxinfo.get('gwTime')  # none if the key or value is missing
            gw_time = rxinfo.get('nsTime')  # none if the key or value is missing
            if gw_time:
                self.toa.append(gw_time)
                lat = gateways[rxinfo['gatewayId']]['latitude']
                lon = gateways[rxinfo['gatewayId']]['longitude']
                alt = gateways[rxinfo['gatewayId']]['altitude']
                self.gw_lat_lon.append([lat, lon])
                x, y, z = pm.geodetic2enu(lat=lat, lon=lon, h=alt, **self.ref_pos)
                self.gateway_positions.append([x, y, z])

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

            # Perform least squares optimization
            x, y = opt.leastsq(func=F, x0=init_pos[:2], Dfun=J)        # For 2D positions

            print(f"Optimized (x, y, z): ({x}, {y})")
            # Estimated lat-lon
            lat_est, lon_est, alt_est = pm.enu2geodetic(e=x[0], n=x[1], u=30, **self.ref_pos)

            if plot==True:
                # Creating a list of results for plotting in the map
                result = {
                    'lat': [lat_est] + [self.gw_lat_lon[i][0] for i in range(len(self.gw_lat_lon))],
                    'lon': [lon_est] + [self.gw_lat_lon[i][1] for i in range(len(self.gw_lat_lon))],
                    'cat': ['Estimated Pos'] + [f'GW Positions' for i in range(len(self.gw_lat_lon))]
                }
                # map plot for individual predictions and estimations 
                directory = os.path.join(self.result_dir, 'map_plot')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_path = os.path.join(directory, f"{packet['deduplicationId']}.png")
                map_plot_cartopy(result, path=file_path)

        else:
            print('Position cannot be resolved. Not enough gws to for TDoA measurement')
            lat_est, lon_est, alt_est = None, None, None

        return [lat_est, lon_est, alt_est]


def main():
    ##################### Testing script ###########################
    with open('../data/tuc_lora_metadata.mqtt_data_13-11.json') as file1:
        ds_json = json.load(file1)
    # ds_json = pd.read_json('data/tuc_lora_metadata.mqtt_data_13-11.json')
    # gw_loc = pd.read_json('data/lorawan_antwerp_gateway_locations.json')
    with open('../data/tuc_lora_gateways.json') as file2:
        gw_loc = json.load(file2) 

    ref_pos = {'lat0': 50.814131,
            'lon0': 12.928044,
            'h0': 320}
    result_directory = f'results/lstsq_2D/{datetime.now().strftime('%Y-%m-%d_%H-%M')}'
    # Ensure the result_directory exists
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    estimator = Estimator(reference_position=ref_pos,
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
    actual_pos_2 = r_woNan[['lat', 'lon']].to_numpy()
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
