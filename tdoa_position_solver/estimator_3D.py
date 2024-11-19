"""
This is a TDoA based estimator or position solver.
Implementing a child classes of the Least square estimator from "project_directory/RSSI_fingerprinting_TDoA_Estimation"
The main function and the jacobian of the master implementation remains the same. Although, here the 3rd dimension: z or altitude is added 
In this version we will make the data parsing more modularized and suitable for reading the mqtt json data from chirpstack
"""

# Importing native libraries
import sys
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from tdoa_position_solver.estimator import *
from general_functions import *   # We are going to use the 'map_plot_cartopy' method
from RSSI_fingerprinting_TDoA_Estimation.performance_eval import calculate_pairwise_error_list

class Estimator_3d(Estimator):
    def __init__(self, reference_position):
        # Speed of propagation (m/s)
        super().__init__(reference_position)
        self.speed = 3e8
        self.ref_pos = reference_position

    # method to generate the residuals for all hyperbolae
    def function(self, measurements):
        def fn(args):
            # Here the args are the arguments passed to the leastsq estimator method
            x, y, z = args[:3]  # Extract x, y and z coordinates from args
            residuals = []
            for i in range(1, len(measurements)):
                xi, yi, zi, ti = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                z0 = measurements[0][2]
                # We use the pandas timestamp method in this case, 
                # because it is the only one that can handle precision upto nanosecond
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(
                    measurements[0][2]).value) * 1e-9  # the values are converted to seconds
                di = diff_seconds * self.speed
                ai = np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0**2)) - abs(di)   # for 3D
                residuals.append(ai)
            return residuals

        return fn

    # Function to generate the Jacobian matrix
    def jacobian(self, measurements):
        def fn(args):
            x, y, z = args[:3]  # Extract x and y coordinates from args
            jac = []
            for i in range(1, len(measurements)):
                xi, yi, zi, _ = measurements[i]
                x0 = measurements[0][0]
                z0 = measurements[0][2]
                y0 = measurements[0][1]
                adx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - (x - x0) / np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0**2))
                ady = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - (y - y0) / np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0**2))
                adz = (z - zi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - (z - z0) / np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0**2))
                jac.append([adx, ady, adz])
            return np.array(jac)
        return fn


    def estimate(self, packet, gateways, plot=False):
        toa = []
        for rxinfo in packet['rxInfo']:
            toa.append(rxinfo['gwTime'])
        
        self.get_gateway_positions(packet, gateways)
        
        # Initital position for estimation: mean of the gateway coordinates 
        init_pos = np.mean(self.gateway_positions, axis=0) 

        if len(self.gateway_positions) >= 4:
            # The timestamps or the TOAs are string values so we convert them to Pandas tmestamp object
            ts = np.asarray([pd.Timestamp(t) for t in toa])
            # We also create a list called measurements
            # This contains the recieving gateway positions and the TOA
            measurements = np.c_[self.gateway_positions, ts]

            # Define functions and jacobian
            F = self.function(measurements)
            J = self.jacobian(measurements)# Define functions and jacobian

            # Perform least squares optimization
            x, y = opt.leastsq(func=F, x0=init_pos, Dfun=J)

            print(f"Optimized (x, y, z): ({x}, {y})")
            # Estimated lat-lon
            lat_est, lon_est, alt_est = pm.enu2geodetic(e=x[0], n=x[1], u=0, **self.ref_pos)
            # lat_est, lon_est, alt_est = pm.enu2geodetic(e=x[0], n=x[1], u=x[2], **self.ref_pos)

            if plot==True:
                # Creating a list of results for plotting in the map
                result = {
                    'lat': [lat_est] + [self.gw_lat_lon[i][0] for i in range(len(self.gw_lat_lon))],
                    'lon': [lon_est] + [self.gw_lat_lon[i][1] for i in range(len(self.gw_lat_lon))],
                    'cat': ['Estimated Pos'] + [f'GW Positions' for i in range(len(self.gw_lat_lon))]
                }
                # map plot for individual predictions and estimations 
                map_plot_cartopy(result, ref=packet['deduplicationId'])

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
            'h0': 0}

    estimator = Estimator_3d(reference_position=ref_pos)
    
    for packet in ds_json:
        est = estimator.estimate(packet=packet,
                                 gateways=gw_loc, 
                                 plot=False)

    est.to_csv('files/position_estimation_comb_gps_time.csv')

    # Getting statistical information of the results 
    est_copy = est.copy()

    est_copy['lat_est'] = pd.to_numeric(est['lat_est'], errors='coerce')
    est_copy['lon_est'] = pd.to_numeric(est['lon_est'], errors='coerce')

    r_woNan = est_copy.dropna(subset=['lat_est'])
    actual_pos_2 = r_woNan[['lat', 'lon']].to_numpy()
    est_pos = r_woNan[['lat_est', 'lon_est']].to_numpy()    

    estimation_error = calculate_pairwise_error_list(ground_truth=actual_pos_2, predictions=est_pos)

    # y = [x for x in estimation_error if x < 1000]
    y = estimation_error

    print(f'min {min(y)}')
    print(f'max {max(y)}')
    print(f'mean {np.mean(y)}')
    print(f'median {np.median(y)}')

if __name__=='__main__':
    main()
