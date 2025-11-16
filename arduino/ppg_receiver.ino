#include <PulseSensorPlayground.h>


int Signal;
int Output = 0;
int PulseSensorPurplePin = 0;
int LedPin = 11;
int packageNum = 0;
int thisTime = 0;

// Pause control
bool isPaused = false;
String inputString = "";
bool stringComplete = false;

// Collection control
bool isCollecting = false;
unsigned long collectStartTime = 0;
const unsigned long collectDuration = 10000; // 10秒 / 10 seconds

void setup() {
  pinMode(LedPin, OUTPUT);
  Serial.begin(115200);
  inputString.reserve(200);
  
  // 启动信息 / Startup message
  Serial.println("=== Pulse Sensor Debug Mode ===");
  Serial.println("System initialized");
  Serial.println("Commands:");
  Serial.println("  - Type 'pause' to pause monitoring");
  Serial.println("  - Type 'start' to resume monitoring");
  Serial.println("  - Type 'collect' (when paused) to collect data for 10 seconds");
  Serial.println("  - Send numbers (0-255) to control LED brightness");
  Serial.println("Status: RUNNING\n");
}

void loop() {
  // Process serial commands
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
  
  // Process complete command
  if (stringComplete) {
    inputString.trim(); // 移除空格和换行 / Remove spaces and newlines
    inputString.toLowerCase(); // 转为小写 / Convert to lowercase
    
    // Debug: 显示接收到的命令 / Display received command
    Serial.print("[DEBUG] Received command: '");
    Serial.print(inputString);
    Serial.print("' (length: ");
    Serial.print(inputString.length());
    Serial.println(")");
    
    if (inputString == "pause") {
      if (!isCollecting) {
        isPaused = true;
        Serial.println("\n[SYSTEM] ⏸️  PAUSED - Monitoring stopped");
        Serial.println("Type 'start' to resume or 'collect' to collect for 10s\n");
      } else {
        Serial.println("[ERROR] Cannot pause during collection");
      }
    }
    else if (inputString == "start") {
      if (!isCollecting) {
        isPaused = false;
        Serial.println("\n[SYSTEM] ▶️  STARTED - Monitoring resumed\n");
      } else {
        Serial.println("[ERROR] Cannot start during collection");
      }
    }
    else if (inputString == "collect") {
      if (isPaused && !isCollecting) {
        isCollecting = true;
        collectStartTime = millis();
        Serial.println("\n[SYSTEM] 📊 COLLECTION STARTED - 10 seconds");
        Serial.println("Format: [COLLECT] Timestamp(ms) | Arduino_millis | Signal | Package Number");
        Serial.println("-----------------------------------------------------------");
      } else if (isCollecting) {
        Serial.println("[ERROR] Collection already in progress");
      } else {
        Serial.println("[ERROR] Please pause first before collecting (type 'pause')");
      }
    }
    else if (inputString.length() > 0) {
      // 尝试解析数字 / Try to parse number
      int value = inputString.toInt();
      if (value >= 0 && value <= 255) {
        Output = value;
        Serial.print("[DEBUG] LED Output set to: ");
        Serial.print(Output);
        Serial.print(" (");
        Serial.print((Output * 100.0) / 255.0, 1);
        Serial.println("%)");
      } else {
        Serial.println("[ERROR] Invalid command or value. Use 'pause', 'start', 'collect', or 0-255");
      }
    }
    
    // 清空字符串 / Clear string
    inputString = "";
    stringComplete = false;
  }
  
  // 检查采集是否结束 / Check if collection is finished
  if (isCollecting) {
    unsigned long elapsed = millis() - collectStartTime;
    
    
    if (elapsed >= collectDuration) {
      isCollecting = false;
      Serial.println("-----------------------------------------------------------");
      Serial.println("[SYSTEM] ✅ COLLECTION COMPLETED - 10 seconds elapsed");
      Serial.println("Status: PAUSED (type 'start' to resume or 'collect' to collect again)\n");

      // Once you finish collecting, reset package number.
      packageNum = 0;
    } 
    else 
    {
      //Read data during collection
      Signal = analogRead(PulseSensorPurplePin);
      unsigned long currentMillis = millis();
      
      // Output format: Timestamp request | Arduino time | Signal | Package Number
      Serial.print("[COLLECT] TIMESTAMP_REQUEST | ");
      Serial.print(currentMillis);
      Serial.print(" | ");
      Serial.print(Signal);
      Serial.print(" | ");
      Serial.println(packageNum);
      
      analogWrite(LedPin, Output);
      delay(20);
      thisTime += 20;
      if(thisTime == 1000){
        thisTime = 0;
        packageNum++;
      }
    }
  }
  // 只在未暂停且未采集时读取和显示数据 / Only read and display when not paused and not collecting
  else if (!isPaused) {
    // 读取脉搏传感器 / Read pulse sensor
    Signal = analogRead(PulseSensorPurplePin);
    
    // Debug: 详细信息 / Detailed information
    Serial.print("[SENSOR] Signal: ");
    Serial.print(Signal);
    Serial.print(" | LED Output: ");
    Serial.print(Output);
    Serial.print(" | Package Number: ");
    Serial.print(0);
    Serial.println("%");
    
    // 控制 LED / Control LED
    analogWrite(LedPin, Output);
    
    delay(20);
  } else {
    //  Update LED when paused but don't read sensor
    analogWrite(LedPin, Output);
    delay(100); // Longer delay when paused
  }
}