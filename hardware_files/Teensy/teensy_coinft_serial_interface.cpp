#include <Arduino.h>

// ==========================================
// USER CONFIGURATION
// ==========================================
#define NUM_COINFTS  2   // Currently supports 1 or 2. You can easily extend to more CoinFTs by modifying the code.
// ==========================================

// Use built-in IntervalTimer for Teensy 4.0
IntervalTimer myTimer;

// Hardware Serial Definitions
#define CFT1  Serial2
#if NUM_COINFTS == 2
  #define CFT2  Serial3
#endif

// Variables
byte data1[24] = { 0 };
volatile bool newData1 = false;

#if NUM_COINFTS == 2
  byte data2[24] = { 0 };
  volatile bool newData2 = false;
#endif

// Function Prototypes
void transmitData(void);

void setup() {
  CFT1.begin(1000000);
  #if NUM_COINFTS == 2
    CFT2.begin(1000000);
  #endif
  
  Serial.begin(115200);

  // Wait for hardware serials to be ready
  while (!CFT1 && millis() < 2000);
  #if NUM_COINFTS == 2
    while (!CFT2 && millis() < 2000);
  #endif
}

void loop() {
  // 1. Listen for PC Commands
  if (Serial.available()) {
    char readChar = Serial.read();

    if (readChar == 'i') {
      // Idle Command
      CFT1.write('i');
      #if NUM_COINFTS == 2
        CFT2.write('i');
      #endif
      myTimer.end(); // Stop the checker
      
      // Reset flags
      newData1 = false;
      #if NUM_COINFTS == 2
        newData2 = false;
      #endif
    }
    else if (readChar == 's') {
      // Start Command
      CFT1.write('s');
      #if NUM_COINFTS == 2
        CFT2.write('s');
      #endif
      
      // Check for data every 1ms (1000 Hz)
      myTimer.begin(transmitData, 1000); 
    }
    else if (readChar == 't') {
      // Tare Command
      CFT1.write('t');
      #if NUM_COINFTS == 2
        CFT2.write('t');
      #endif
    }
  }

  // 2. Listen for Sensor 1 Data
  if (CFT1.available() > 25) {
    if (CFT1.read() == 0x02) { // Start Byte
      CFT1.readBytes(data1, 24);
      newData1 = true;
      
      // Flush until End Byte
      unsigned long startWait = micros();
      while (CFT1.read() != 0x03 && (micros() - startWait < 1000)); 
    }
  }

  // 3. Listen for Sensor 2 Data (Only if enabled)
  #if NUM_COINFTS == 2
    if (CFT2.available() > 25) {
       if (CFT2.read() == 0x02) { // Start Byte
         CFT2.readBytes(data2, 24);
         newData2 = true;
         
         // Flush until End Byte
         unsigned long startWait = micros();
         while (CFT2.read() != 0x03 && (micros() - startWait < 1000));
       }
    }
  #endif
}

// Interrupt Service Routine (Runs every 1ms)
void transmitData(void) {
  
  #if NUM_COINFTS == 1
    // --- SINGLE SENSOR MODE ---
    if (newData1) {
      Serial.write(0x00); // Header
      Serial.write(0x00);
      Serial.write(data1, 24);
      
      newData1 = false;
    }

  #elif NUM_COINFTS == 2
    // --- DUAL SENSOR MODE (Synchronized) ---
    // Only send if BOTH sensors have reported in
    if (newData1 && newData2) {
      Serial.write(0x00); // Header
      Serial.write(0x00);
      Serial.write(data1, 24);
      Serial.write(data2, 24);

      newData1 = false;
      newData2 = false;
    }
  #endif
}

