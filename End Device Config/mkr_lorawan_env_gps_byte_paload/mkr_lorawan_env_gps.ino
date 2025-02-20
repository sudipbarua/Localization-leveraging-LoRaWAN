/*
Sends environmental and GPS data every x minutes over LoRaWAN

Modues
- MKR WAN 1310
- MKR GPS Shield
    GPS Module
      u-blox SAM-M8Q
- MKR ENV Shield
    Sensors
      Temperature (HTS221)	Range	15-40 (celsius)
      Humidity (HTS221)	Range	20-80% rH (relative humidity)
      Barometric pressure (LPS22HB)	Range	260-1260 hPa (hectopascal)
      Ambient light (TEMT6000)	Range	Max 650 LUX.

Parameters
  - GPS (Lat, lon, alt, speed, number of Satellietes...)
  - Env (Temp, hum, pressure, illumination, UVA, UVB, UV index...)
*/

#include <MKRWAN.h>
#include "secrets.h"
#include <Arduino_MKRENV.h>
#include <Arduino_MKRGPS.h>


LoRaModem modem;

// Secret data are stored in secrets.h file
String appEui = SECRET_APP_EUI;
String appKey = SECRET_APP_KEY;

long lastSendTime = 0;        // last send time
int interval = 20;          // interval between sends in seconds
String msg = "no data"

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Start the GPS
  if (!GPS.begin(GPS_MODE_SHIELD)) {
    Serial.println("Failed to initialize GPS!");
    while (1);
  }
  Serial.println("GPS initialized.");
  
  // Start the ENV sensor  
  if (!ENV.begin()) {
    Serial.println("Failed to initialize MKR ENV Shield!");
    while (1)
      ;
  } else Serial.println("Env sensor Ready");
  
  // change this to your regional band (eg. US915, AS923, ...)
  if (!modem.begin(EU868)) {
    Serial.println("Failed to start module");
    while (1) {}
  };
  Serial.println("Modem initialized.");
  Serial.print("Module version is: ");
  Serial.println(modem.version());
  Serial.print("Device EUI is: ");
  Serial.println(modem.deviceEUI());

  if (!modem.joinOTAA(appEui, appKey)) {
    Serial.println("Modem failed to connect!!!");
    while (1) {}
  }
  // Set poll interval to 60 secs.
  modem.minPollInterval(60);
  // NOTE: independent of this setting, the modem will
  // not allow sending more than one message every 2 minutes,
  // this is enforced by firmware and can not be changed.
}

void loop() {
  // Sending messages
  if (millis() / 1000 - lastSendTime / 1000 > interval) {
    // Wait for GPS fix
    // while (!GPS.available()) {
    //   Serial.print(".");
    //   delay(1000);
    // }

    // // Get GPS data
    // double latitude = GPS.latitude();
    // double longitude = GPS.longitude();
    // float altitude = GPS.altitude();

    double latitude = 50.8138275;
    double longitude = 12.9279842;
    float altitude = 30.5;

    // read all the env sensor values
    uint8_t temperature = ENV.readTemperature();
    uint8_t humidity = ENV.readHumidity();
    uint8_t pressure = ENV.readPressure();
    uint8_t illuminance = ENV.readIlluminance();
    uint8_t uva = ENV.readUVA();
    uint8_t uvb = ENV.readUVB();
    uint8_t uvIndex = ENV.readUVIndex();

    // Print GPS data to serial
    Serial.print("Latitude: ");
    Serial.println(latitude, 7);
    Serial.print("Longitude: ");
    Serial.println(longitude, 7);
    Serial.print("Altitude: ");
    Serial.println(altitude, 2);

    // print each of the sensor values
    Serial.print("Temperature = ");
    Serial.print(temperature);
    Serial.println(" °C");

    Serial.print("Humidity    = ");
    Serial.print(humidity);
    Serial.println(" %");

    Serial.print("Pressure    = ");
    Serial.print(pressure);
    Serial.println(" kPa");

    Serial.print("Illuminance = ");
    Serial.print(illuminance);
    Serial.println(" lx");

    Serial.print("UVA         = ");
    Serial.println(uva);

    Serial.print("UVB         = ");
    Serial.println(uvb);

    Serial.print("UV Index    = ");
    Serial.println(uvIndex);

    // Create payload
    byte payload[24];
    memcpy(payload, &latitude, 8);
    memcpy(payload + 8, &longitude, 8);
    memcpy(payload + 16, &altitude, 4);
    memcpy(payload + 20, &temperature, 1);
    memcpy(payload + 21, &humidity, 1);
    memcpy(payload + 22, &pressure, 1);
    memcpy(payload + 23, &illuminance, 1);
    
    // Send payload
    modem.beginPacket();
    modem.write(payload, sizeof(payload));
    
    // End packet check
    int err = modem.endPacket(true);
    if (err > 0) {
      Serial.println("Message sent correctly!");
      lastSendTime = millis();  
    } else {
      Serial.println("Error sending message :(");
    }

  }
}
