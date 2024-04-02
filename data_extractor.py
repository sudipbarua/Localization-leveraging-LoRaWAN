import csv
from influxdb_client import InfluxDBClient
import pandas as pd

# InfluxDB credentials: bucket, organization, and token.
bucket = "lorawan_mqtt_collector"
org = "TU Chemnitz"
token = "qhYr-5pT9wSBB4_eou5bWqdZcjei3fJFzPMOFP7zFli1he5VDis_oB6CdAVVTRpgR7o1wJqxRyq6WzXIIl0y1g=="
url="http://localhost:8086"

# INfluxDB client instantiation
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# Flux query, and then format it as a Python string.
# query = '''
# from(bucket: "test")
# |> range(start: 2024-03-24T11:06:00Z, stop: 2024-03-27T11:06:00Z)
# |> filter(fn: (r) => r["_measurement"] == "Uplink_data")
# |> filter(fn: (r) => r["_field"] == "data_size" or r["_field"] == "dr" or r["_field"] == "fCnt" or r["_field"] == "txInfo_modulation_spreadingFactor" or r["_field"] == "txInfo_modulation_bandwidth" or r["_field"] == "txInfo_frequency" or r["_field"] == "rxInfo_uplinkId" or r["_field"] == "rxInfo_snr" or r["_field"] == "rxInfo_rssi" or r["_field"] == "rxInfo_location_longitude" or r["_field"] == "rxInfo_location_latitude" or r["_field"] == "rxInfo_location_altitude" or r["_field"] == "rxInfo_channel" or r["_field"] == "rxInfo_board" or r["_field"] == "fPort")
# '''

query = '''
from(bucket: "lorawan_mqtt_collector")
|> range(start: -1h)
|> filter(fn: (r) => r["_measurement"] == "Uplink_data")
|> filter(fn: (r) => r["_field"] == "data_size")
'''


print(query)
# Pass the query() method two named parameters: org and query.
result = query_api.query_data_frame(org=org, query=query)

all_data = pd.DataFrame(data=result)  # Creating an empty dataframe

# Write the combined data to a CSV file
all_data.to_csv('all_tables2.csv', index=False)
