from range_based_estimator import *


class PLPositioningEngine(RangeBasedEstimator):
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


def main():
    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_2019_dataset.json', 'r') as file1:
        data = json.load(file1)

    with open('D:/work_dir/Datasets/LoRa_anomaly-detection/data/lorawan_antwerp_gateway_locations.json', 'r') as file2:
        gateways = json.load(file2)

    ref_pos = {'lat0': 51.260644,
        'lon0': 4.370656,
        'h0': 0}
    
    estimator = PLPositioningEngine(reference_position=ref_pos, gateways=gateways, 
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
        

    