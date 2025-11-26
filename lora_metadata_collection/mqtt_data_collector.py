import paho.mqtt.client as mqtt
from pymongo import MongoClient
import json
import datetime

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tuc_lora_metadata"
COLLECTION_NAME = "mqtt_data"

# MQTT Configuration
MQTT_BROKER = ""  # Replace with your MQTT broker
MQTT_PORT = 1883  # Default MQTT port
MQTT_TOPIC = "eu868/application/58e920e9-56d0-4ee1-8fbf-0134826738fc/device/+/event/up"  # Replace with your topic or topic pattern

# Initialize MongoDB client and get the collection
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
    else:
        print("Failed to connect, return code %d\n", rc)

# Callback when a message is received
def on_message(client, userdata, msg):
    try:
        # Parse the MQTT message payload
        payload = msg.payload.decode("utf-8")
        message = json.loads(payload)

        # Add a timestamp
        message["timestamp"] = datetime.datetime.now()

        # Insert the message into MongoDB
        collection.insert_one(message)
        print(f"Stored message to MongoDB: {message}")
    except Exception as e:
        print(f"Failed to store message: {e}")

# Set up MQTT client and callbacks
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to the MQTT broker and start listening
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_forever()
