import sys
sys.path.append('D:/work_dir/Datasets/LoRa_anomaly-detection')
from general_functions import *   # We are going to use the 'map_plot_cartopy' method
import json
import os

# with open('D:/work_dir/Datasets/LoRa_anomaly-detection/range_based_positioning/results/pl_model_okumura_hata/2024-12-16_16-49/result.json', 'r') as file:
with open(
        'D:/work_dir/Datasets/LoRa_anomaly-detection/range_based_positioning/results/pl_model_okumura_hata_suburban/2024-12-17_15-57/result.json',
        'r') as file:
        r = json.load(file)

for i, d in enumerate(r):
    rows = []
    # Add the estimated position
    rows.append({'lat': d['lat'], 'lon': d['lon'], 'cat': 'Actual Pos'})
    rows.append({'lat': d['lat_est'], 'lon': d['lon_est'], 'cat': 'Estimated Pos'})

    # Add the gateway positions
    for key, value in d.items():
        if key.startswith('gw_'):
            rows.append({'lat': value['lat'], 'lon': value['lon'], 'cat': 'GW Positions'})


    file_dir = f'D:/work_dir/Datasets/LoRa_anomaly-detection/range_based_positioning/results/pl_model_okumura_hata_suburban/2024-12-17_15-57/map_plot/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = os.path.join(file_dir, f'{i}.png')
    map_plot_cartopy(rows, path=file_path)