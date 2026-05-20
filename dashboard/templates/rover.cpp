/*
========================================
DHT11 TEST FOR ESP32
PIN: GPIO13
========================================
*/

#include <DHT.h>

// =====================================

#define DHT_PIN   13
#define DHT_TYPE  DHT11

DHT dht(DHT_PIN, DHT_TYPE);

// =====================================

void setup()
{
  Serial.begin(115200);

  Serial.println("DHT11 TEST START");

  dht.begin();

  delay(2000);
}

// =====================================

void loop()
{
  float temperature = dht.readTemperature();

  float humidity = dht.readHumidity();

  // ===================================
  // CHECK IF SENSOR FAILED
  // ===================================

  if (isnan(temperature) || isnan(humidity))
  {
    Serial.println("DHT11 READ FAILED");

    delay(2000);

    return;
  }

  // ===================================
  // PRINT VALUES
  // ===================================

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print(" °C");

  Serial.print(" | Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");

  delay(2000);
}