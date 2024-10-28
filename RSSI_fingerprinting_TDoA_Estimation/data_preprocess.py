from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import pandas as pd
from itertools import combinations
import numpy as np
import pymap3d as pm 


def get_row_raw_gw_info(df, raw_ds):
    # Create a dictionary to map timestamps to row IDs in raw_ds
    timestamp_to_row_id = {}

    for id, row in raw_ds.iterrows():
        for gateway in row['gateways']:
            timestamp = gateway['rx_time']['time']
            if timestamp not in timestamp_to_row_id:
                timestamp_to_row_id[timestamp] = []
            timestamp_to_row_id[timestamp].append(id)

    # Initialize the gw_info column in df
    df['gw_info_row_id'] = 0

    # For each timestamp in df, find the corresponding row ID in raw_ds
    for i, t in df.iterrows():
        timestamp = t['RX Time']
        if timestamp in timestamp_to_row_id:
            # If there are multiple matches, you may need to decide how to handle them.
            # Here, we'll just take the first match.
            df.loc[i, 'gw_info_row_id'] = timestamp_to_row_id[timestamp][0]
            print(f'row number {timestamp_to_row_id[timestamp][0]}')
    
    return df

class DataPreprocess:
    def __init__(self):
        pass

    def data_spliting(self, x_scaled, y, random_state, train_size, val_size):
        # Train, validation, test set splitting, (70%/15%/15%)
        x_train, x_test_val, y_train, y_test_val = train_test_split(x_scaled, y.values, train_size=train_size, random_state=random_state)
        x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=(1-val_size), random_state=random_state)
        print(f'Training shape:      {x_train.shape}')
        print(f'Test shape:          {x_test.shape}')
        print(f'Validation shape:    {x_val.shape}')

        return x_train, x_test, x_val, y_train, y_test, y_val

    def scaling(self, scaler, x):
        if scaler=='MinMax':
            min_max_scaler = MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
        elif scaler=='powed':
            pass
        elif scaler=='exponential':
            pass
        return x_scaled_df

    def data_cleaning(self, ds, wathear_data=True):
        # Removing duplicates
        ds = ds.drop_duplicates()
        #### Remove entries with less than 3 gateways ####
        columns = ds.columns
        # x = ds[columns[0:68]]  # Get basestations' RSS readings
        x = ds[columns[0:72]]  # Get basestations' RSS readings
        c = (x == -200).astype(int).sum(axis=1) # counting the amount of not-receiving gateways per message
        # c = 68 - c  # counting the amount of receiving gateways per message
        c = 72 - c  # counting the amount of receiving gateways per message
        c = c.tolist()

        # finding indices of messages with less than 3 receiving gateways, and dropping these messages from the dataset ds
        indices = list()
        for i in range(len(c)):
            element = c[i]
            if element <3:
                indices.append(i)  # appending all indices of messages with fewer than 3 receiving gateways

        print('Size before cleaning: ',ds.shape) # size before...
        ds = ds.drop(indices) # dropping all entries with fewer than 3 receiving gateways
        ds.reset_index(drop=True, inplace=True)  # Reset the index
        print('Size after cleaning: ',ds.shape) # ... and size after the dropping


        #### Dataset preparation for ML pipeline
        columns = ds.columns
        # x1 = ds[columns[0:68]] #features (RSS receptions)
        # x2 = ds[columns[69:71]] # SF and HDOP
        # x3 = ds[columns[75:]] #weather data
        x1 = ds[columns[0:73]] #features (RSS receptions with timestamps)
        x2 = ds[columns[73:75]] # SF and HDOP
        x3 = ds[columns[79:]] #weather data
        if wathear_data==True:
            x = pd.concat([x1, x2, x3], axis=1)
        else:
            x = pd.concat([x1, ds['gw_info_row_id']], axis=1)

        y = ds[columns[75:77]] # target (locations)

        return x, y

    def get_gw_meta(self):
        gw_meta = self.ds_json.loc[self.index, ['gateways']]
        for gws in gw_meta:
            for gw in gws:
                gateway_id = gw['id']
                if gateway_id in self.gw_loc:
                    # Check wheather GW location information is actually availaable in the list or not
                    self.gw_ids.append(gateway_id)
                    self.toa.append(gw['rx_time']['time'])
                else:
                    print(f'Gateway {gateway_id} location information is not available')


    # Collecting GW coordinates, ToA and TDoA 
    def get_gw_cord_tdoa(self, index, ds_json, gw_loc, ref_pos):
        """
        Inputs:
            - index: This is the reference index of the raw data json file 'RSSI_fingerprinting_TDoA_Estimation\data\lorawan_antwerp_2019_dataset.json'
                     from which we are going to take the GW meta data
            - ds_json: Raw data file read by pandas as dataframe
            - gw_loc: Gateway locations from the raw data file
            - ref_pos: Reference position for geodetic to ENU conversion
        """

        self.ds_json = ds_json
        self.gw_loc = gw_loc
        self.ref_pos = ref_pos
        self.index = index
        self.gw_ids = []
        self.toa = []

        self.get_gw_meta()

        print(f'Receiving gateways: {self.gw_ids}')
        print(f'Time of arrivals: {self.toa}')

        gw_pos = []
        gw_lat_lon = []

        for i in self.gw_ids:
            gw = self.gw_loc[i]
            x, y, z = pm.geodetic2enu(lat=gw['latitude'], lon=gw['longitude'], h=0, **self.ref_pos)
            gw_lat_lon.append([gw['latitude'], gw['longitude']])
            gw_pos.append([x,y,z])
        gw_positions = np.asarray(gw_pos)

        print('Gateway coordinates (enu): ')
        print(gw_positions)

        # Calulating TDoA value
        # We use the pandas timestamp method in this case,
        # because it is the only one that can handle precision upto nano second
        tdoa = []
        # Generate unique pairs using combinations
        for i, j in combinations(range(len(self.toa)), 2):
            diff = pd.Timestamp(self.toa[i]).value - pd.Timestamp(self.toa[j]).value
            diff_seconds = diff*1e-9
            tdoa.append([i, j, diff_seconds])
        print(f'Time difference of arrival: {tdoa}')

        return self.toa, gw_positions, tdoa, gw_lat_lon


class DataProcessingGPStimer(DataPreprocess):
    def __init__(self):
        super().__init__()

    def get_gw_meta(self):
    # Here only the gateways with GPS capability is taken into consideration
        gw_meta = self.ds_json.loc[self.index, ['gateways']]
        for gws in gw_meta:
            for gw in gws:
                gateway_id = gw['id']
                timestamp_type = gw['rx_time']['ts_type']
                if (gateway_id in self.gw_loc) & (timestamp_type == 'GPS_RADIO'):
                    # Check whether GW location information is actually available in the list or not
                    # Additionally check if the timestamp is GPS time or not
                    self.gw_ids.append(gateway_id)
                    self.toa.append(gw['rx_time']['time'])
                else:
                    print(f'Gateway {gateway_id} location information is not available or not usable for TDoA measurements')


def main():
    # collecting the row number of the matched data sample based on the TOA timestamp info in the gateways subsection
    ds = pd.read_csv('../data/antwerp_ds_weather-data_2019.csv', index_col=0)
    ds_raw = pd.read_json('../data/lorawan_antwerp_2019_dataset.json')
    ds_row_ref = get_row_raw_gw_info(ds, ds_raw)
    ds_row_ref.to_csv('data/antwerp_combo_raw_ref.csv')  # Saving the dataframe containing the corresponding index or row number from the raw jason file


if __name__=='__main__':
    main()