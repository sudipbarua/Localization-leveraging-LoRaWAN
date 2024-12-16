import math
from range_based_estimator import *


class PLPositioningEngine_COST231(RangeBasedEstimator):
    def __init__(self):
        super().__init__()
    
    def packet_perser(self):
        pass
    
    # def gw_cord_collector(self, pkt):
        # pass

    def link_budget_param_collector(self):
        pass

    def weather_info_collector(self):
        pass

    def range_calculator(self, pkt):
        pass  


class PLPositioningEngine_OkumuraHata(RangeBasedEstimator):
    def __init__(self, reference_position, gateways, result_directory):
        super().__init__(reference_position, gateways, None, None, None,
                         result_directory)

    def range_calculator(self, pkt):
        R = []
        # Collect the frequency from the packet 
        
        #!!!!!!!!!!!!try to do dynamic h_b and h_m extraction from the pkt. pkt should contain all the necessary information in JSON format!!!!!!!!
        h_b = 30                                 # The height of the base station 
        h_m = 1                                  # height of the mobile station
        f = self.channel_mapper(pkt['channel'])  # frequency in MHz

        for gw in pkt['gateways']:
            RSSI = gw['rssi']
            SNR = gw['snr']

            # implementing range calculation from Okumura-Hata path loss model
            A = 69.55 + 26.161 * math.log10(f) - 13.82 * math.log10(h_b) - 3.2*(math.log10(11.75 * h_m))**2 + 4.97
            B = 44.9 - 6.55 * math.log10(h_b)
            c = 5.4 + 2 * (math.log10(f/28))**2
            D = 40.94 + 4.78 * (math.log10(f))**2 - 18.33*math.log10(f)

            # Some fixed parameters
            P_tx = 14
            G_rx = 3
            G_tx = 2
            # To avoid divided by zero error we use an adjustment value epsilon to SNR
            if SNR==0:
                SNR = 1e-10
            # We use the pathloss calculated from the RSSI SNR since we cannot measure pathloss
            L = P_tx + G_tx + G_rx + 10 * np.log10(1 + 1/SNR) - RSSI

            d = 10 ** ((L-A)/B) * 1000
            R.append(d)
        return R


def main():
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_2019_dataset.json', 'r') as file1:
        data = json.load(file1)

    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_gateway_locations.json', 'r') as file2:
        gateways = json.load(file2)

    ref_pos = {'lat0': 51.260644,
        'lon0': 4.370656,
        'h0': 10}  # The alitude of Antwerp Belgium is 10m above sea level
    result_directory = f'results/pl_model_okumura_hata/{datetime.now().strftime('%Y-%m-%d_%H-%M')}'
    # Ensure the result_directory exists
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    estimator = PLPositioningEngine_OkumuraHata(reference_position=ref_pos, gateways=gateways,
                                                result_directory=result_directory)

    result_json = []
    for i, packet in enumerate(data):
        if len(packet['gateways']) >= 3:
            ################ Estimation starts #####################
            lat_est, lon_est, _ = estimator.estimate(packet, packet_ref=i, plot=False)
            # Collecting ground truth values
            lat, lon = packet['latitude'], packet['longitude']
            # calculate the estimation error
            try:
                est_er = haversine(tuple([lat, lon]),tuple([lat_est, lon_est]))*1000  # multiplying by 1000 to transform from Km to m
            except:
                est_er = None
            print(f'Estimation error: {est_er}')
            new_data = {'lat': lat, 'lon': lon, 'lat_est': lat_est, 'lon_est': lon_est, 'Estimation Error': est_er}
            # Collect the coordinates of the receiving gateways
            for i, gw in enumerate(estimator.gateway_lat_lon):
                new_data[f'gw_{i}'] = {}
                new_data[f'gw_{i}']['lat'], new_data[f'gw_{i}']['lon'] = gw
            result_json.append(new_data)
        else: 
            print('Position cannot be solved: expected Number of reciveing gateways is at least 3')

    # Save the results
    with open(f"{result_directory}/result.json", "w") as file:
        json.dump(result_json, file, indent=4)  # 'indent=4' makes the file human-readable

    

if __name__=='__main__':
    main()
        

    