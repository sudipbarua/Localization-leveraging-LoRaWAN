import json
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import paho.mqtt.client as mqtt
import base64

# InfluxDB settings
influxdb_url = "http://localhost:8086"
influxdb_token = "qhYr-5pT9wSBB4_eou5bWqdZcjei3fJFzPMOFP7zFli1he5VDis_oB6CdAVVTRpgR7o1wJqxRyq6WzXIIl0y1g=="  # The API token
influxdb_org = "TU Chemnitz"
influxdb_bucket = "lorawan_mqtt_collector"

# MQTT settings
mqtt_broker = "134.109.5.110"
mqtt_port = 1883
# mqtt_topic = "eu868/gateway/7076ff005607214b/event/up "
# mqtt_topic = "eu868/application/#"
mqtt_topic = "#"


def write_to_influxdb(line):
    # Write the data to InfluxDB
    print('Authenticating to InfluxDB')
    client = influxdb_client.InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
    print('Authentication successful. Writing DB via API')
    # try:
    #     write_api = client.write_api(write_options=SYNCHRONOUS)
    #     write_api.write(influxdb_bucket, influxdb_org, line)
    #     print('Write successful')

    # except:
    #     print('Error writing to DB')
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(influxdb_bucket, influxdb_org, line)
    print('Write successful')


def format_GW_data(data):
    # Formatting the data into InfluxDB Line Protocol
    line = (
        f'gateway,Id={data["gatewayId"]} '
        f'latitude={data["location"]["latitude"]},'
        f'longitude={data["location"]["longitude"]},'
        f'altitude={data["location"]["altitude"]},'
        f'rxPacketsReceived={data["rxPacketsReceived"]},'
        f'rxPacketsReceivedOk={data["rxPacketsReceivedOk"]},'
        f'txPacketsReceived={data["txPacketsReceived"]},'
        f'txPacketsEmitted={data["txPacketsEmitted"]} '
        f'{int(data["time"])}'
    )

    return line

def format_uplink_data(data):
    # Influxdb line protoocl message accepted format f"{measurement_name},{tag_set} {field_set}" 
    # Tags are the columns in the raw table
    tags = ["tenantId", "tenantName", "applicationId", "applicationName", "deviceProfileId", "deviceProfileName", "deviceName", "devEui", "deviceClassEnabled"]
    tag_set = ",".join(f"{tag}={data['deviceInfo'][tag]}" for tag in tags)

    tags_2 = ["devAddr", "adr", "dr", "fCnt", "fPort", "confirmed"]
    tag_set_2 = ",".join(f"{tag}={data[tag]}" for tag in tags_2)
    # Preprocess the data field value. We consider the data size in bytes  
    decoded_data = base64.b64decode(data["data"])
    data_size = len(decoded_data)

    # Add rxInfo and txInfo as tags
    # The key 'context' is ignored since it contained unsupported charecters like '=' equal sign
    rxInfo_tag_set = ",".join(f"rxInfo_{key}={value}" for key, value in data['rxInfo'][0].items() if isinstance(value, (str, int, float)) and key != 'context')
    txInfo_tag_set = ",".join(f"txInfo_{key}={value}" for key, value in data['txInfo'].items() if isinstance(value, (str, int, float)))

    modulation_tag_set = ",".join(f"txInfo_modulation_{key}={value}" for key, value in data['txInfo']['modulation']['lora'].items())
    location_tag_set = ",".join(f"rxInfo_location_{key}={value}" for key, value in data['rxInfo'][0]['location'].items())  

    line = (
        f"Uplink_data,test_data=tuc_lab {tag_set},{tag_set_2}," 
        f"{rxInfo_tag_set},"
        f"{txInfo_tag_set},"
        f"{modulation_tag_set},"
        f"{location_tag_set},"
        f"data_size={data_size}"
    )
    return line 

# MQTT callback function
def on_message(client, userdata, message):
    # Check for the validity of the message payload
    try:
        # Parse the JSON data
        data = json.loads(message.payload)
        print(f'Received message payload of topic {message.topic} in JSON format: {data}')
    except json.JSONDecodeError:
        print(f"Received a non-JSON message: {message.payload}")

    topic_split = message.topic.split('/')
    if topic_split[1]=='application' and topic_split[-1]=='up':
        # Get the uplink data in influxdb line format     
        line = format_uplink_data(data)    
        print(line)
        write_to_influxdb(line)
    else:
        print('Ignoring irrelevant data')
            


if __name__ == '__main__':
    # Create a new MQTT client
    client = mqtt.Client()

    # Assign the callback function
    client.on_message = on_message

    # Connect to the MQTT broker
    print('Connceting to client')
    client.connect(mqtt_broker, mqtt_port)
    print('Connceted')

    # Subscribe to the topic
    print('Subscribing to topic')
    client.subscribe(mqtt_topic)
    print('Subscribed')

    # Start the MQTT loop
    client.loop_forever()

