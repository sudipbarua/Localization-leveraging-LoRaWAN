import plotly.express as px
import numpy as np
import scipy.optimize as opt
import pymap3d as pm
import pandas as pd
from data_preprocess import DataPreprocess


class Least_square_estimator:
    def __init__(self):
        # Speed of propagation (m/s)
        self.speed = 3e8

    # method to generate the residuals for all hyperbolae
    def function(self, measurements, speeds):
        def fn(args):
            # Here the args are the arguments passed to the leastsq estimator method
            x, y = args[:2]  # Extract x and y coordinates from args
            residuals = []
            for i in range(1, len(measurements)):
                xi, yi, ti = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                # We use the pandas timestamp method in this case, 
                # because it is the only one that can handle precision upto nanosecond
                diff_seconds = (pd.Timestamp(ti).value - pd.Timestamp(measurements[0][2]).value) * 1e-9  # the values are converted to seconds
                di = diff_seconds * speeds[i]
                ai = np.sqrt((x - xi)**2 + (y - yi)**2) - np.sqrt((x - x0)**2 + (y - y0)**2) - abs(di)
                residuals.append(ai)
            return residuals
        return fn

    # Function to generate the Jacobian matrix
    def jacobian(self, measurements):
        def fn(args):
            x, y = args[:2]  # Extract x and y coordinates from args
            jac = []
            for i in range(1, len(measurements)):
                xi, yi, ti = measurements[i]
                x0 = measurements[0][0]
                y0 = measurements[0][1]
                adx = (x - xi) / np.sqrt((x - xi)**2 + (y - yi)**2) - (x - x0) / np.sqrt((x - x0)**2 + (y - y0)**2)
                ady = (y - yi) / np.sqrt((x - xi)**2 + (y - yi)**2) - (y - y0) / np.sqrt((x - x0)**2 + (y - y0)**2)
                jac.append([adx, ady])
            return np.array(jac)
        return fn


    def estimate(self, data, reference_position, ds_json, gateway_locations, plot=False):
        for idx, _ in data.iterrows():
            row = data.iloc[idx]
            
            # Initital position for estimation 
            init_pos = [row['x_i'], row['y_i']]

            # Now we collect the TOA and gateway lat-lon and calculate the TDoA and positions  
            toa, gw_pos, _, gw_lat_lon = DataPreprocess().get_gw_cord_tdoa(row['gw_ref'], ds_json, gateway_locations, reference_position)
            
            # The timestamps or the TOAs are string values so we convert them to Pandas tmestamp object
            ts = np.asarray([pd.Timestamp(t) for t in toa])
            # We also create a list called measurements 
            # This contains the recieving gateway positions and the TOA 
            measurements = np.c_[gw_pos[:, [0,2]], ts]

            speeds = [self.speed] * len(measurements)

            # Define functions and jacobian
            F = self.function(measurements, speeds)
            J = self.jacobian(measurements, speeds)

            # Perform least squares optimization
            x, y = opt.leastsq(func=F, x0=init_pos, Dfun=J)

            print(f"Optimized (x, y): ({x}, {y})")
            # Estimated lat-lon 
            lat_est, lon_est, _ = pm.enu2geodetic(e=x[0], n=x[1], u=0, **reference_position)

            # Creating a list of results for plotting in the map 
            result = {
                'lat': [row['lat'], lat_est, row['pred_lat']] + [gw_lat_lon[i][0] for i in range(len(gw_lat_lon))],
                'lon': [row['lon'], lon_est, row['pred_lon']] + [gw_lat_lon[i][1] for i in range(len(gw_lat_lon))],
                'cat': ['Actual Pos', 'Estimated Pos', 'ML Predicted Pos'] + [f'GW Positions' for i in range(len(gw_lat_lon))]
            }
            
            if plot==True:
                # map plot for individual predictions and estimations 
                self.map_plot(result, gw_ref=row['gw_ref'])
            data.loc[idx, 'lat_est'] = lat_est
            data.loc[idx, 'lon_est'] = lon_est
        return data


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
        # fig.write_image(f"fig/map_plots/gw{gw_ref}.jpg")  # Saving image


def main():
    ##################### Testing script ###########################
    ds_json = pd.read_json('RSSI-fingerprinting_TDoA-Estimation/data/lorawan_antwerp_2019_dataset.json')
    gw_loc = pd.read_json('RSSI-fingerprinting_TDoA-Estimation/data/lorawan_antwerp_gateway_locations.json')

    # Loading initial position coordinates form machine learning predictions
    pos_pred_rssi = pd.read_csv('RSSI-fingerprinting_TDoA-Estimation/files/position_pred_RSSI.csv', index_col=0)
    pos_pred_comb = pd.read_csv('RSSI-fingerprinting_TDoA-Estimation/files/position_pred_weather-comb.csv', index_col=0)


    ref_pos = {'lat0': 51.260644,
            'lon0': 4.370656,
            'h0': 0}

    pos_pred_rssi['x'], pos_pred_rssi['y'], pos_pred_rssi['z'] = pm.geodetic2enu(lat=pos_pred_rssi['lat'], lon=pos_pred_rssi['lon'], h=0, **ref_pos)
    pos_pred_rssi['x_i'], pos_pred_rssi['y_i'], pos_pred_rssi['z_i'] = pm.geodetic2enu(lat=pos_pred_rssi['pred_lat'], lon=pos_pred_rssi['pred_lon'], h=0, **ref_pos)

    pos_pred_comb['x'], pos_pred_comb['y'], pos_pred_comb['z'] = pm.geodetic2enu(lat=pos_pred_comb['lat'], lon=pos_pred_comb['lon'], h=0, **ref_pos)
    pos_pred_comb['x_i'], pos_pred_comb['y_i'], pos_pred_comb['z_i'] = pm.geodetic2enu(lat=pos_pred_comb['pred_lat'], lon=pos_pred_comb['pred_lon'], h=0, **ref_pos)
    estimator = Least_square_estimator()
    # est_rssi = estimator.estimate(data=pos_pred_rssi, 
    #                              reference_position=ref_pos, 
    #                              ds_json=ds_json, 
    #                              gateway_locations=gw_loc)
    
    # est_rssi.to_csv('RSSI-fingerprinting_TDoA-Estimation/files/position_estimation_rssi.csv')
    
    est_comb = estimator.estimate(data=pos_pred_comb, 
                                 reference_position=ref_pos, 
                                 ds_json=ds_json, 
                                 gateway_locations=gw_loc, plot=True)

    est_comb.to_csv('RSSI-fingerprinting_TDoA-Estimation/files/position_estimation_comb.csv')



if __name__=='__main__':
    main()
