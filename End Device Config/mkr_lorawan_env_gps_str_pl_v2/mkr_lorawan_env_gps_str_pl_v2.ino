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
#include <TinyGPS++.h>
// #define IS_USB;  // Uncomment and upload if USB connection is available

TinyGPSPlus tinygps;

LoRaModem modem;

// Secret data are stored in secrets.h file
String appEui = SECRET_APP_EUI;
String appKey = SECRET_APP_KEY;

long lastSendTime = 0;        // last send time
int interval = 60;          // interval between sends in seconds
String msg = "no data";

static const int RXPin = 13, TXPin = 14;

void setup() {
  // during the initialinzation stage, if there is no serial monitor, the program gets stuck at this phase.
  // To handle this issue, we defined a variable called IS_USB and we will initialize the serial communication only when we connect the USB
  // Or even never uncommenting the #define IS_USB; line at all can still work by skipping the serial initialization.
  #if defined(IS_USB)
    Serial.begin(9600);
    while (!Serial);
  #endif
    
  // Start the GPS
  if (!GPS.begin(GPS_MODE_SHIELD)) {
    Serial.println("Failed to initialize GPS!");
    while (1);
  }
  Serial.println("GPS initialized.");
  
  // Start the ENV sensor  
  if (!ENV.begin()) {
    Serial.println("Failed to initialize MKR ENV Shield!");
    while (1);
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
  Serial.println("Joining was successfull");
  // Set poll interval to 60 secs.
  modem.minPollInterval(60);
  // NOTE: independent of this setting, the modem will
  // not allow sending more than one message every 2 minutes,
  // this is enforced by firmware and can not be changed.
}

void loop() {
  // Sending messages
  double lat;
  double lon;
  float alt;
  if (GPS.available()){
    // Get GPS data
    double lat = GPS.latitude();
    double lon = GPS.longitude();
    float alt = GPS.altitude();
    
    // read all the env sensor values
    float temp = ENV.readTemperature();
    float hum = ENV.readHumidity();
    float p = ENV.readPressure();
    float lum = ENV.readIlluminance();
    float uva = ENV.readUVA();
    float uvb = ENV.readUVB();
    float uvI = ENV.readUVIndex();

    print_data(lat, lon, alt, temp, hum, p, lum, uva, uvb, uvI);

    msg = String(lat, 7) + ";" + String(lon, 7) + ";" + String(alt) +";" + String(temp)+";" + String(hum)+";" + String(p)+";" + String(lum);//+" , " + String(uva)+" , " + String(uvb);//+" , " + String(uvI); 

    Serial.println(msg);   
    // Send payload
    modem.beginPacket();
    modem.print(msg);
    
    // End packet check
    int err = modem.endPacket(true);
    if (err > 0) {
      Serial.println("Message sent correctly!");
      lastSendTime = millis();  
    } else {
      Serial.println("Error sending message");
    }
    delay(1000);
    if (!modem.available()) {
        Serial.println("No downlink message received at this time.");
        return;
      }
    // Receiving
    char rcv[64];
    int i = 0;
    while (modem.available()) {
      rcv[i++] = (char)modem.read();
    }
    Serial.print("Received: ");
    for (unsigned int j = 0; j < i; j++) {
      Serial.print(rcv[j] >> 4, HEX);
      Serial.print(rcv[j] & 0xF, HEX);
      Serial.print(" ");
    }
    Serial.println();

    delay(1000);

  }
  
}

void print_data(double lat, double lon, float alt, float temp, float hum,float p, float lum, float uva, float uvb, float uvI){
  // Print GPS data to serial
  Serial.print("Latitude: ");
  Serial.println(lat, 7);
  Serial.print("Longitude: ");
  Serial.println(lon, 7);
  Serial.print("Altitude: ");
  Serial.println(alt, 2);

  // print each of the sensor values
  Serial.print("Temperature = ");
  Serial.print(temp);
  Serial.println(" °C");

  Serial.print("Humidity    = ");
  Serial.print(hum);
  Serial.println(" %");

  Serial.print("Pressure    = ");
  Serial.print(p);
  Serial.println(" kPa");

  Serial.print("Illuminance = ");
  Serial.print(lum);
  Serial.println(" lx");

  // Serial.print("UVA         = ");
  // Serial.println(uva);

  // Serial.print("UVB         = ");
  // Serial.println(uvb);

  // Serial.print("UV Index    = ");
  // Serial.println(uvI);

}
