#include <Arduino.h>
#include <TimerOne.h>
#include <TimerThree.h>

#define CFT1  Serial2
#define CFT2  Serial3

// variables
byte data1[24] = { 0 };
byte data2[24] = { 0 };
bool newData1 = false;
bool newData2 = false;

// put function declarations here:
void transmitData(void);

void setup() {
  // put your setup code here, to run once:
  CFT1.begin(1000000);
  CFT2.begin(1000000);
  Serial.begin(115200);
  while (!CFT1);
  while (!CFT2);
  while (!Serial);
  // Serial.println("Setup done");
  Timer1.initialize(2778); // 10000 -> 100Hz  || 2778 -> 360Hz
  Timer1.attachInterrupt(transmitData);
  Timer1.stop();
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()) {
    char readChar = Serial.read();
    // Serial.println(readChar);
    if (readChar == 'i') {
      // Serial.println("i command received");
      CFT1.write('i');
      CFT2.write('i');
      Timer1.stop();
    }
    else if (readChar == 's') {
      // Serial.println("s command received");
      CFT1.write('s');
      CFT2.write('s');
      Timer1.start();
    }
    else if (readChar == 't') {
      CFT1.write('t');
      CFT2.write('t');
    }
  }
  if (CFT1.available())
    // Serial.println("CFT1 Available");
  if (CFT1.available() > 25) {

    if (CFT1.read() == 0x02) {
      CFT1.readBytes(data1, 24);
      newData1 = true;
      // Serial.println("CFT1 Read");
      if (CFT1.read() != 0x03) {
        while (CFT1.read() != 0x03);
      }
    }
  }
  if (CFT2.available() > 25 && CFT2.read() == 0x02) {
    CFT2.readBytes(data2, 24);
    // Serial.println("CFT2 detected");
    newData2 = true;
    if (CFT2.read() != 0x03) {
      while (CFT2.read() != 0x03);
    }
  }
}

// put function definitions here:
void transmitData(void) {
  // if (!newData1 || !newData2) return;
  Serial.write(0x00);
  Serial.write(0x00);
  Serial.write(data1, 24);
  Serial.write(data2, 24);
  newData1 = false;
  newData2 = false;
}