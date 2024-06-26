import plotly.express as px
import numpy as np
import scipy.optimize as opt
import pymap3d as pm
import pandas as pd
from data_preprocess import get_gw_cord_tdoa

class Least_square_estimator:
    def __init__(self):
        # Speed of propagation (m/s)
        self.speed = 3e8

    # method to generate the residuals for all hyperbolae
    def functions(self, measurements, speeds):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            residuals = []
            for i in range(len(measurements)):
                xi, yi, ti = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                # We use the pandas timestamp method in this case, 
                # because it is the only one that can handle precision upto nano second
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(measurements[0][2]).value) * 1e-9  # the values are converted to seconds
                di = diff_seconds * speeds[i]
                ai = np.sqrt((x - xi)**2 + (y - yi)**2) - np.sqrt((x - x0)**2 + (y - y0)**2) - abs(di)
                residuals.append(ai)
            return residuals
        return fn

    # Function to generate the Jacobian matrix
    def jacobian(self, measurements, speeds):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            jac = []
            for i in range(len(measurements)):
                xi, yi, ti = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                # We use the pandas timestamp method in this case, 
                # because it is the only one that can handle precision upto nano second
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(measurements[0][2]).value) * 1e-9  # the values are converted to seconds
                di = diff_seconds * speeds[i]
                adx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2) - (x - x0) / np.sqrt((x - x0)**2 + (y - y0)**2)
                ady = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2) - (y - y0) / np.sqrt((x - x0)**2 + (y - y0)**2)
                jac.append([adx, ady])
            return np.array(jac)
        return fn


    def estimate(self, data, ref_pos, ds_json, gw_loc):
        for idx, _ in data.iterrows():
            row = data.iloc[idx]
            # Initital position for estimation 
            init_pos = [row['x_i'], row['y_i']]

            toa, gw_pos, tdoa, gw_lat_lon = get_gw_cord_tdoa(row['gw_ref'], ds_json, gw_loc)
            
            ts = np.asarray([pd.Timestamp(t) for t in toa])
            measurements = np.c_[gw_pos[:, [0,2]], ts]


            # Extract measurements for use in functions and jacobian
            speeds = [self.speed] * len(measurements)

            # Define functions and jacobian
            F = self.functions(measurements, speeds)
            J = self.jacobian(measurements, speeds)

            # Perform least squares optimization
            x, y = opt.leastsq(func=F, x0=init_pos, Dfun=J)

            print(f"Optimized (x, y): ({x}, {y})")

            lat_est, lon_est, _ = pm.enu2geodetic(e=x[0], n=x[1], u=0, **ref_pos)

            result = {
                'lat': [row['lat'].values[0], lat_est, row['pred_lat_comb'].values[0]] + [gw_lat_lon[i][0] for i in range(len(gw_lat_lon))],
                'lon': [row['lon'].values[0], lon_est, row['pred_lon_comb'].values[0]] + [gw_lat_lon[i][1] for i in range(len(gw_lat_lon))],
                'cat': ['Actual Pos', 'Estimated Pos', 'ML Predicted Pos'] + [f'GW Positions' for i in range(len(gw_lat_lon))]
            }
            self.map_plot(result, row['gw_ref'])
            data.loc[row, 'lat_est'] = lat_est
            data.loc[row, 'lon_est'] = lon_est


    def map_plot(self, result, gw_ref):

        fig = px.scatter_mapbox(result, 
                                lat='lat', 
                                lon='lon',
                                color='cat', 
                                zoom=11, 
                                height=800,
                                width=1400)
        # Update the marker size
        fig.update_traces(marker=dict(size=15))  # Adjust the size value as needed

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()
        fig.write_image(f"fig/map_plots/gw{gw_ref}.png")
