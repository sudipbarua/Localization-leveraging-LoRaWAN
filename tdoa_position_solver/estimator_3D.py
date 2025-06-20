"""
This is a TDoA based estimator or position solver.
Implementing a child classes of the Least square estimator from "project_directory/RSSI_fingerprinting_TDoA_Estimation"
The main function and the jacobian of the master implementation remains the same. Although, here the 3rd dimension: z or altitude is added 
In this version we will make the data parsing more modularized and suitable for reading the mqtt json data from chirpstack
"""

# Importing native libraries
import sys
from tkinter import NO
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from tdoa_position_solver.estimator import *
from general_functions import *   # We are going to use the 'map_plot_cartopy' method
from RSSI_fingerprinting_TDoA_Estimation.performance_eval import calculate_pairwise_error_list

class Estimator_3d(Estimator):
    def __init__(self, reference_position, result_directory):
        # Speed of propagation (m/s)
        super().__init__(reference_position, result_directory)
        self.speed = 3e8
        self.ref_pos = reference_position
        self.result_dir = result_directory

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
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(measurements[0][3]).value) * 1e-9  # the values are converted to seconds
                di = diff_seconds * self.speed
                ai = np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2) - abs(di)   # for 3D
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
                y0 = measurements[0][1]
                z0 = measurements[0][2]
                adx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - (x - x0) / np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
                ady = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - (y - y0) / np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
                adz = (z - zi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) - (z - z0) / np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
                jac.append([adx, ady, adz])
            return np.array(jac)
        return fn


    def estimate(self, packet, gateways, plot=False, minimizing_algorithm='lm', tr_solver=None, distance_from_init_pos=None):
        self.get_gateway_positions_toa(packet, gateways)
        
        # Initital position for estimation: mean of the gateway coordinates 
        init_pos = np.mean(self.gateway_positions, axis=0) 

        if len(self.gateway_positions) >= 4:        # *++++++++++Change in the condition
            # The timestamps or the TOAs are string values so we convert them to Pandas tmestamp object
            ts = np.asarray([pd.Timestamp(t) for t in self.toa])
            # We also create a list called measurements
            # This contains the recieving gateway positions and the TOA
            measurements = np.c_[self.gateway_positions, ts]

            # Define functions and jacobian
            F = self.function(measurements)
            J = self.jacobian(measurements)# Define functions and jacobian

            if distance_from_init_pos is not None:
                lower_bounds = [init_pos[0] - distance_from_init_pos, init_pos[1] - distance_from_init_pos, 0]
                upper_bounds = [init_pos[0] + distance_from_init_pos, init_pos[1] + distance_from_init_pos, 1000]
                bounds = (lower_bounds, upper_bounds) 
            else:
                bounds = None  # No bounds if range is not specified

            # Perform least squares optimization
            try:
                solution = opt.least_squares(fun=F, jac=J, x0=init_pos, method=minimizing_algorithm,
                                             tr_solver=tr_solver,
                                             bounds=bounds) 
            except Exception as e:
                print(f"Error during optimization: {e}")
                return [None, None, None]

            print(f"Optimized (x, y, z): ({solution.x[0]}, {solution.x[1]}, {solution.x[2]})")
            # Estimated lat-lon
            lat_est, lon_est, alt_est = pm.enu2geodetic(e=solution.x[0], n=solution.x[1], u=solution.x[2], **self.ref_pos)     # *****Change*****
            print(f'Estimated latitude: {lat_est}, longitude: {lon_est}, altitude: {alt_est}')
            
            ############################################ Plotting in map ############################################
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
            #############################################################################################################

        else:
            print('Position cannot be resolved. Not enough gws to for TDoA measurement')
            lat_est, lon_est, alt_est = None, None, None

        return [lat_est, lon_est, alt_est]


def main():
    ##################### Testing script ###########################
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/tuc_lora_metadata.mqtt_data_22-27_4gw.json') as file1:
        ds_json = json.load(file1)
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/tuc_lora_gateways.json') as file2:
        gw_loc = json.load(file2)

    ################ Initializing the estimator ####################
    # Reference position for the ENU to geodetic conversion
    ref_pos = {'lat0': 50.814131,
            'lon0': 12.928044,
            'h0': 320}
    result_dir_identifier = 'trf_exact_100sqkm' # A special identifier for the result directory 
    result_directory = f'D:/work_dir/Datasets/LoRa_anomaly-detection/tdoa_position_solver/results/lstsq_3D/{result_dir_identifier}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}'
    # Ensure the result_directory exists
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    estimator = Estimator_3d(reference_position=ref_pos,
                             result_directory=result_directory)
    distance_from_initial_estimation = 5000  # meters, for TRF estimator this defines the region of interest for the optimization
    ################################################################

    estimated_position = []
    for packet in ds_json:
        est = estimator.estimate(packet=packet,
                                 gateways=gw_loc,
                                 plot=False,
                                 minimizing_algorithm='trf',  # 'trf' for Trust Region Reflective, 'lm' for Levenberg-Marquardt
                                 tr_solver='exact',  # 'exact' for exact Jacobian, None for numerical Jacobian
                                 distance_from_init_pos=distance_from_initial_estimation)  # Optional, can be None
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
