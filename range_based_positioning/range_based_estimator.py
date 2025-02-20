import requests
import re
import numpy as np
import pandas as pd
import pymap3d as pm
import scipy.optimize as opt
import json
import os
from haversine import haversine
from datetime import datetime
import sys
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from general_functions import *   # We are going to use the 'map_plot_cartopy' method
from RSSI_fingerprinting_TDoA_Estimation.performance_eval import calculate_pairwise_error_list


class RangeBasedEstimator:
    def __init__(self, reference_position, gateways, path_loss_exponent, reference_distance,
                 reference_rssi, result_directory):
        self.reference_position = reference_position
        self.gateways = gateways  # The list of known coordinates of the receiving gateways
        self.path_loss_exponent = path_loss_exponent
        self.reference_distance = reference_distance
        self.reference_rssi = reference_rssi
        self.result_dir = result_directory

    def packet_perser(self):
        pass
    
    def gw_cord_collector(self, pkt):
        gw_pos_enu = []
        gw_lat_lon = []
        for gateway in pkt['gateways']:
            lat = self.gateways[gateway['id']]['latitude']
            lon = self.gateways[gateway['id']]['longitude']
            x, y, z = pm.geodetic2enu(lat=lat, lon=lon, h=30, **self.reference_position)
            gw_pos_enu.append([x,y,z])
            gw_lat_lon.append([lat, lon])
        
        self.gateway_coordinates_enu = np.asarray(gw_pos_enu)
        self.gateway_lat_lon = np.asarray(gw_lat_lon)

    def link_budget_param_collector(self):
        pass

    
    def osm_query_buildier(self, lat1, lon1, lat2=None, lon2=None, region="coverage"):
        if region=="path":
            query = f"""
            [out:json];
            (
            way["building"]["height"]({lat1},{lon1},{lat2},{lon2});
            );
            out body;
            """
        elif region=="coverage":
            query = f"""
            [out:json];
            (
            way["building"]["height"](around:3000, {lat1},{lon1});
            );
            out body;
            """
        return query
 

    def get_average_building_height(self, lat1, lon1, lat2, lon2):
        # Define the Overpass API query
        query = self.osm_query_buildier(lat1, lon1)
        
        # Send the request to the Overpass API
        response = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
        data = response.json()
        
        # Extract building heights
        heights = []
        for element in data['elements']:
            if 'tags' in element and 'height' in element['tags']:
                h = element['tags']['height']
                # The height elements are strings and may contain the units like "150 m"
                # So with the help of regular expression check we only take the value
                match = re.match(r"([-+]?\d*\.\d+|[-+]?\d+)", h)
                height = match.group(0)
                heights.append(float(height))
        
        # Calculate average height
        if heights:
            average_height = sum(heights) / len(heights)
            return average_height
        else:
            return None  # No building height data available
        
    
    def get_average_street_width(self, lat1, lon1, lat2, lon2):
        # Define the Overpass API query to get streets with width
        query = f"""
        [out:json];
        (
        way["highway"]["width"]({lat1},{lon1},{lat2},{lon2});
        );
        out body;
        """
        
        # Send the request to the Overpass API
        response = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
        
        # Check if the request was successful
        if response.status_code != 200:
            print("Error fetching data from Overpass API")
            return None
        
        data = response.json()
        
        # Extract street widths
        widths = []
        for element in data['elements']:
            if 'tags' in element and 'width' in element['tags']:
                try:
                    widths.append(float(element['tags']['width']))  # Convert width to float
                except ValueError:
                    continue  # Skip if conversion fails

        # Calculate average width
        if widths:
            average_width = np.mean(widths)
            return average_width
        else:
            return None  # No width data available


    def residual_function(self, measurements, distances):
        """
        input measurements format = [[x0, y0, z0],
                                     [x1, y1, z1],
                                     [x2, y2, z2]
                                    ]
        range format = [r0, r1, r2]
        """
        def fn(args):
            # Here the args are the arguments passed to the leastsq estimator method
            x, y = args[:2]  # Extract x and y coordinates from args. Usually these are the initial coordinates (initialization for least sq estimation)
            residuals = []
            for i in range(len(measurements)):
                xi, yi, _ = measurements[i]
                res_i = np.sqrt((x - xi)**2 + (y - yi)**2) - distances[i]
                residuals.append(res_i)
            return residuals
        return fn

    def jacobian_of_residual(self, measurements):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            jac = []
            for i in range(len(measurements)):
                xi, yi, _ = measurements[i]
                res_dx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2) 
                res_dy = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2) 
                jac.append([res_dx, res_dy])
            return np.array(jac)
        return fn

    def weather_info_collector(self):
        pass


    def channel_mapper(self, channel):
        # Maps the channel number to frequency
        # Useful for Antwerp dataset 
        match channel:
            case 1:
                return 868.1
            case 2:
                return 868.3
            case 3:
                return 868.5
            case 4:
                return 867.1
            case 5:
                return 867.3
            case 6:
                return 867.5
            case 7:
                return 867.7
            case 8:
                return 867.9
            case _:
                return 868.8


    def range_calculator(self, pkt):
        # From reference RSSI and distance: using a reference based path loss model   
        R = []
        beta = self.path_loss_exponent
        rssi_0 = self.reference_rssi
        d_0 = self.reference_distance

        for gw in pkt['gateways']:
            rssi = gw['rssi']
            d = d_0 * 10 ** ( (rssi_0 - rssi) / 10 * beta )
            R.append(d)
        
        return R

    def initial_position(self, gw_pos):
        # center of the polygon created by the GWs 
        return np.mean(gw_pos, axis=0)
    
    def plot_in_map(self, lat_est, lon_est, packet_ref):
        # Creating a list of results for plotting in the map
        result = {
            'lat': [lat_est] + [self.gateway_lat_lon[i][0] for i in range(len(self.gateway_lat_lon))],
            'lon': [lon_est] + [self.gateway_lat_lon[i][1] for i in range(len(self.gateway_lat_lon))],
            'cat': ['Estimated Pos'] + [f'GW Positions' for i in range(len(self.gateway_lat_lon))]
        }
        # map plot for individual predictions and estimations
        directory = os.path.join(self.result_dir, 'map_plot')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{packet_ref}.png")
        map_plot_cartopy(result, path=file_path)

    def estimate(self, packet, packet_ref, plot=False):
        # list of position of the gateways in ENU format
        try:
            self.gw_cord_collector(packet)
        except:
            print('Gateway coordinates are unavailable')
            return None, None, None

        # Initital position for estimation
        init_pos = self.initial_position(self.gateway_coordinates_enu)
        distances = self.range_calculator(packet)  # returns an array of ranges from
        # Define functions and jacobian
        F = self.residual_function(self.gateway_coordinates_enu, distances)
        J = self.jacobian_of_residual(self.gateway_coordinates_enu)
        # Perform least squares optimization for 2D
        x, _ = opt.leastsq(func=F, x0=init_pos[:2], Dfun=J)
        print(f"Optimized (x, y): ({x})")
        # Estimated lat-lon
        lat_est, lon_est, alt_est = pm.enu2geodetic(e=x[0], n=x[1], u=30, **self.reference_position)
        if plot == True:
            self.plot_in_map(lat_est, lon_est, packet_ref)

        return [lat_est, lon_est, alt_est]


class RangeBasedEstimator_3d(RangeBasedEstimator):
    def __init__(self, reference_position, gateways, path_loss_exponent, reference_distance,
                 reference_rssi, result_directory):
        super().__init__(reference_position, gateways, path_loss_exponent, reference_distance,
                 reference_rssi, result_directory)

    def residual_function(self, measurements, distances):
        def fn(args):
            # Here the args are the arguments passed to the leastsq estimator method
            x, y, z = args[:3]  # Extract x and y coordinates from args. Usually these are the initial coordinates (initialization for least sq estimation)
            residuals = []
            for i in range(len(measurements)):
                xi, yi, zi = measurements[i]
                res_i = np.sqrt((x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2) - distances[i]
                residuals.append(res_i)
            return residuals
        return fn

    def jacobian_of_residual(self, measurements):
        def fn(args):
            x, y, z = args[:3]  # Extract x and y coordinates from args
            jac = []
            for i in range(len(measurements)):
                xi, yi, zi = measurements[i]
                res_dx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2)
                res_dy = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2)
                res_dz = (z - zi) / np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2)
                jac.append([res_dx, res_dy, res_dz])
            return np.array(jac)
        return fn

    def estimate(self, packet, packet_ref, plot=False):
        # list of position of the gateways in ENU format
        try:
            self.gw_cord_collector(packet)
        except:
            print('Gateway coordinates are unavailable')
            return None, None, None

        # Initital position for estimation
        init_pos = self.initial_position(self.gateway_coordinates_enu)
        distances = self.range_calculator(packet)  # returns an array of ranges from
        # Define functions and jacobian for 3D
        F = self.residual_function(self.gateway_coordinates_enu, distances)
        J = self.jacobian_of_residual(self.gateway_coordinates_enu)
        # x, _ = opt.leastsq(func=F, x0=init_pos, Dfun=J)
        # print(f"Optimized (x, y): ({x})")
        # lower_bounds = [-np.inf, -np.inf, 0]
        lower_bounds = [init_pos[0] - 2500, init_pos[1] - 2500, 0]
        # upper_bounds = [np.inf, np.inf, 1000]
        upper_bounds = [init_pos[0] + 2500, init_pos[1] + 2500, 1000]
        try:
            solution = opt.least_squares(fun=F, x0=init_pos, jac=J, bounds=(lower_bounds, upper_bounds))
        except Exception as e:
            print(e)
            return [None, None, None]
        print(f"Optimized (x, y): ({solution.x})")
        # Estimated lat-lon
        # lat_est, lon_est, alt_est = pm.enu2geodetic(e=x[0], n=x[1], u=x[2], **self.reference_position)
        lat_est, lon_est, alt_est = pm.enu2geodetic(e=solution.x[0], n=solution.x[1], u=solution.x[2],
                                                        **self.reference_position)

        if plot == True:
            self.plot_in_map(lat_est, lon_est, packet_ref)
        return [lat_est, lon_est, alt_est]

            
def main():
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_2019_dataset.json', 'r') as file1:
        data = json.load(file1)

    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_gateway_locations.json', 'r') as file2:
        gateways = json.load(file2)

    ref_pos = {'lat0': 51.260644,
        'lon0': 4.370656,
        'h0': 10}
    result_directory = f'results/pl_model_friss/{datetime.now().strftime('%Y-%m-%d_%H-%M')}'
    # Ensure the result_directory exists
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    estimator = RangeBasedEstimator(reference_position=ref_pos, gateways=gateways, 
                                    reference_distance=4.709445557884708,
                                    reference_rssi=-60,
                                    path_loss_exponent=0.4057,
                                    result_directory=result_directory)
    
    for i, packet in enumerate(data):
        if len(packet['gateways']) >= 3:
            lat, lon, _ = estimator.estimate(packet, i)

        else: 
            print('Position cannot be solved: expected Number of reciveing gateways is at least 3')

    

if __name__=='__main__':
    main()
    