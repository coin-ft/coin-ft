#include <Arduino.h>

// Use built-in IntervalTimer for Teensy 4.0
IntervalTimer myTimer;

#define CFT1  Serial2
#define CFT2  Serial3

// variables
byte data1[24] = { 0 };
byte data2[24] = { 0 };

// variable shared with the interrupt
volatile bool newData1 = false;
volatile bool newData2 = false;

// Function Prototypes
void transmitData(void);

void setup() {
  CFT1.begin(1000000);
  CFT2.begin(1000000);
  Serial.begin(115200); // Teensy USB ignores this number and runs at max USB speed

  // Wait for hardware serials to be ready
  while (!CFT1 && millis() < 2000);
  while (!CFT2 && millis() < 2000);
}

void loop() {
  // 1. Listen for PC Commands
  if (Serial.available()) {
    char readChar = Serial.read();

    if (readChar == 'i') {
      // Idle Command
      CFT1.write('i');
      CFT2.write('i');
      myTimer.end(); // Stop the checker
      
      // Reset flags so we don't start with stale data next time
      newData1 = false;
      newData2 = false;
    }
    else if (readChar == 's') {
      // Start Command
      CFT1.write('s');
      CFT2.write('s');
      
      // Check for data every 1 milliseconds (1000 Hz)
      // This minimizes latency once the data actually arrives.
      myTimer.begin(transmitData, 1000); 
    }
    else if (readChar == 't') {
      CFT1.write('t');
      CFT2.write('t');
    }
  }

  // 2. Listen for Sensor 1 Data
  if (CFT1.available() > 25) {
    if (CFT1.read() == 0x02) { // Start Byte
      CFT1.readBytes(data1, 24);
      newData1 = true; // Raise flag for Sensor 1
      
      // Flush until End Byte to keep stream clean
      unsigned long startWait = micros();
      while (CFT1.read() != 0x03 && (micros() - startWait < 1000)); 
    }
  }

  // 3. Listen for Sensor 2 Data
  if (CFT2.available() > 25) {
     if (CFT2.read() == 0x02) { // Start Byte
       CFT2.readBytes(data2, 24);
       newData2 = true; // Raise flag for Sensor 2
       
       // Flush until End Byte
       unsigned long startWait = micros();
       while (CFT2.read() != 0x03 && (micros() - startWait < 1000));
     }
  }
}

// Interrupt Service Routine (Runs every 1ms)
void transmitData(void) {
  // Only send if BOTH sensors have reported in since the last send.
  if (newData1 && newData2) {
    
    Serial.write(0x00);
    Serial.write(0x00);
    Serial.write(data1, 24);
    Serial.write(data2, 24);

    newData1 = false;
    newData2 = false;
  }
}