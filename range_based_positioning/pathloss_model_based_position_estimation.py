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
    def __init__(self, reference_position, gateways, path_loss_exponent, reference_distance, reference_rssi):
        super().__init__(reference_position, gateways, path_loss_exponent, reference_distance, reference_rssi)

    def range_calculator(self, pkt):
        R = []
        # Collect the frequency from the packet 
        
        #!!!!!!!!!!!!try to do dynamic h_b and h_m extraction from the pkt. pkt should contain all the necessary information in JSON format!!!!!!!!
        h_b = 30                                 # The height of the base station 
        h_m = 1                                  # height of the mobile station
        f = self.channel_mapper(pkt['channel'])  # frequency in MHz

        for gw in pkt['gateways']:
            RSSI = gw('rssi') 
            SNR = gw('snr')

            # implementing range calculation from Okumura-Hata path loss model
            A = 69.55 + 26.161 * math.log10(f) - 13.82 * math.log10(h_b) - 3.2*(math.log10(11.75 * h_m))**2 + 4.97
            B = 44.9 - 6.55 * math.log10(h_b)
            c = 5.4 + 2(math.log10(f/28))**2
            D = 40.94 + 4.78(math.log10(f))**2 - 18.33*math.log10(f)

            L = P_tx + G_tx + G_rx + 10 * math.log10(1 + 1/SNR) - RSSI

            d = 10 ** ((L-A)/B)
            R.append(d)



def main():
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_2019_dataset.json', 'r') as file1:
        data = json.load(file1)

    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_gateway_locations.json', 'r') as file2:
        gateways = json.load(file2)

    ref_pos = {'lat0': 51.260644,
        'lon0': 4.370656,
        'h0': 0}
    
    estimator = PLPositioningEngine_OkumuraHata(reference_position=ref_pos, gateways=gateways, 
                                    reference_distance=4.709445557884708,
                                    reference_rssi=-60,
                                    path_loss_exponent=0.4057)
    
    for packet in data:
        if len(packet['gateways']) >= 3:
            lat, lon, _ = estimator.estimate(packet)

        else: 
            print('Position cannot be solved: expected Number of reciveing gateways is at least 3')

    

if __name__=='__main__':
    main()
        

    