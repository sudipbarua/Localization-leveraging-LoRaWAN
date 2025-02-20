"""
In this modified version we consider the combination of all receiving gateways for make the set of error equations
and subsequently the error function.
"""

import numpy as np
import pandas as pd
from leastsq_estimator_with_gps_time import Least_square_estimator_gps_timer
import pymap3d as pm
from itertools import combinations


class Least_square_estimator_combinationGWs(Least_square_estimator_gps_timer):
    def __init__(self):
        super().__init__()

    def function(self, measurements, *args, **kwargs):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            residuals = []
            for i, j in combinations(range(len(measurements)), 2):
                xi, yi, ti = measurements[i]
                xj, yj, tj = measurements[j]
                # We use the pandas timestamp method in this case,
                # because it is the only one that can handle precision upto nano second
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(tj).value) * 1e-9  # the values are converted to seconds
                dij = diff_seconds * self.speed
                aij = np.sqrt((x - xj)**2 + (y - yj)**2) - np.sqrt((x - xi)**2 + (y - yi)**2) - abs(dij)
                residuals.append(aij)
            return residuals
        return fn

    # Function to generate the Jacobian matrix
    def jacobian(self, measurements, *args, **kwargs):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            jac = []
            for i, j in combinations(range(len(measurements)), 2):
                xi, yi, ti = measurements[i]
                xj, yj, tj = measurements[j]
                # We use the pandas timestamp method in this case,
                # because it is the only one that can handle precision upto nano second
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(tj).value) * 1e-9  # the values are converted to seconds
                adx = (x - xj) / np.sqrt((x - xj)**2 + (y - yj)**2) - (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2)
                ady = (y - yj) / np.sqrt((x - xj)**2 + (y - yj)**2) - (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2)
                jac.append([adx, ady])
            return np.array(jac)
        return fn


def main():
    ##################### Testing script ###########################
    ds_json = pd.read_json('../data/lorawan_antwerp_2019_dataset.json')
    gw_loc = pd.read_json('../data/lorawan_antwerp_gateway_locations.json')

    # Loading initial position coordinates form machine learning predictions
    pos_pred_rssi = pd.read_csv('files/position_pred_RSSI.csv', index_col=0)
    pos_pred_comb = pd.read_csv('files/position_pred_weather-comb.csv', index_col=0)


    ref_pos = {'lat0': 51.260644,
            'lon0': 4.370656,
            'h0': 0}

    pos_pred_rssi['x'], pos_pred_rssi['y'], pos_pred_rssi['z'] = pm.geodetic2enu(lat=pos_pred_rssi['lat'], lon=pos_pred_rssi['lon'], h=0, **ref_pos)
    pos_pred_rssi['x_i'], pos_pred_rssi['y_i'], pos_pred_rssi['z_i'] = pm.geodetic2enu(lat=pos_pred_rssi['pred_lat'], lon=pos_pred_rssi['pred_lon'], h=0, **ref_pos)

    pos_pred_comb['x'], pos_pred_comb['y'], pos_pred_comb['z'] = pm.geodetic2enu(lat=pos_pred_comb['lat'], lon=pos_pred_comb['lon'], h=0, **ref_pos)
    pos_pred_comb['x_i'], pos_pred_comb['y_i'], pos_pred_comb['z_i'] = pm.geodetic2enu(lat=pos_pred_comb['pred_lat'], lon=pos_pred_comb['pred_lon'], h=0, **ref_pos)
    estimator = Least_square_estimator_combinationGWs()
    est_rssi = estimator.estimate(data=pos_pred_rssi,
                                 reference_position=ref_pos,
                                 ds_json=ds_json,
                                 gateway_locations=gw_loc)
    
    est_rssi.to_csv('files/position_estimation_rssi_gw_comb.csv')
    
    est_comb = estimator.estimate(data=pos_pred_comb, 
                                 reference_position=ref_pos, 
                                 ds_json=ds_json, 
                                 gateway_locations=gw_loc, plot=True)

    est_comb.to_csv('files/position_estimation_comb_gw_comb.csv')



if __name__=='__main__':
    main()
