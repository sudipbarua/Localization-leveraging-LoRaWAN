import requests
import re
import numpy as np
import pandas as pd
import pymap3d as pm
import scipy.optimize as opt

import sys
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from RSSI_fingerprinting.data_preprocess import DataPreprocess

class RangeBasedEstimator:
    def __init__(self, reference_position):
        self.reference_position = reference_position
    
    def packet_perser(self):
        pass
    
    def gw_cord_collector(self):
        pass

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


    def path_loss_range(self):
        # Free space path loss 

        return R

    def residual_function(self, measurements, range):
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
                xi, yi = measurements[i]
                res_i = np.sqrt((x - xi)**2 + (y - yi)**2) - range[i]
                residuals.append(res_i)
            return residuals
        return fn
    
    def jacobian_of_residual(self, measurements):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            jac = []
            for i in range(1, len(measurements)):
                xi, yi = measurements[i]
                res_dx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2) 
                res_dy = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2) 
                jac.append([res_dx, res_dy])
            return np.array(jac)
        return fn


    def weather_info_collector(self):
        pass

    def haversine_distance(self):
        pass

    def range_predictor(self):
        pass

    def estimator(self, data, ds_json, gateway_locations):
        for idx, _ in data.iterrows():
            row = data.iloc[idx]
            
            # Initital position for estimation 
            init_pos = [row['x_i'], row['y_i']]
            
            _, gw_pos, _, gw_lat_lon = DataPreprocess().get_gw_cord_tdoa(row['gw_ref'], ds_json, gateway_locations, self.reference_position)

            # We create a list called measurements 
            # This contains the recieving gateway positions 
            measurements = np.c_[gw_pos[:, [0,2]]]

            distance = self.path_loss_range()  # returns an array of ranges from 

            # Define functions and jacobian
            F = self.residual_function(measurements, distance)
            J = self.jacobian_of_residual(measurements)

            # Perform least squares optimization
            x, y = opt.leastsq(func=F, x0=init_pos, Dfun=J)

            print(f"Optimized (x, y): ({x}, {y})")
            # Estimated lat-lon 
            lat_est, lon_est, _ = pm.enu2geodetic(e=x[0], n=x[1], u=0, **self.reference_position)

        


    