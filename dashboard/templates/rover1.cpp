// ================================================================
//  LINE FOLLOWER â€” 2 SENSOR VERSION
//  ESP32 + 2x L298N
//  S1 = LEFT (GPIO34)   S3 = RIGHT (GPIO36)
//  CENTER SENSOR REMOVED COMPLETELY
//  BLACK LINE = LOW (raw) â†’ inverted to 1 in code
// ================================================================

// ================= IR SENSOR PINS =================

#define S1  34   // LEFT
#define S3  36   // RIGHT

// ================= FRONT L298N =================

#define RF_EN   25
#define RF_IN1  26
#define RF_IN2  27

#define LF_EN   33
#define LF_IN1  32
#define LF_IN2  23

// ================= REAR L298N =================

#define RR_EN   18
#define RR_IN1  19
#define RR_IN2  21

#define LR_EN    4
#define LR_IN1  16
#define LR_IN2  17

// ================= PWM CHANNELS =================

#define CH_RF  0
#define CH_LF  1
#define CH_RR  2
#define CH_LR  3

// ================= ULTRASONIC =================

#define TRIG_PIN  5
#define ECHO_PIN  22

// ================================================================
// SPEED
// ================================================================

int baseSpeed    = 110;
int maxSpeed     = 155;
int recoverSpeed = 150;  // aggressive recovery turn speed

// ================================================================
// PID â€” boosted for 2-sensor
// ================================================================

float Kp = 55.0;   // higher â€” only 2 discrete states, need strong response
float Ki =  0.0;
float Kd = 20.0;   // higher â€” damps overshoot on aggressive corrections

float error         = 0;
float previousError = 0;
float integral      = 0;
float derivative    = 0;
float correction    = 0;
float lastError     = 0;

// ================================================================
// OBSTACLE
// ================================================================

const int  OBSTACLE_CM = 15;
const int  CLEAR_CM    = 20;
const long HOLD_MS     = 400;

bool          obstacleActive = false;
unsigned long obstacleTimer  = 0;

// ================================================================
// MOTOR CONTROL
// ================================================================

void moveForward(int leftSpd, int rightSpd) {
  leftSpd  = constrain(leftSpd,  0, maxSpeed);
  rightSpd = constrain(rightSpd, 0, maxSpeed);
  ledcWrite(CH_LF, leftSpd);
  ledcWrite(CH_LR, leftSpd);
  ledcWrite(CH_RF, rightSpd);
  ledcWrite(CH_RR, rightSpd);
}

void stopMotors() {
  ledcWrite(CH_RF, 0);
  ledcWrite(CH_LF, 0);
  ledcWrite(CH_RR, 0);
  ledcWrite(CH_LR, 0);
}

// ================================================================
// RECOVERY â€” aggressive, based on lastError only
// Fully stops inner side, blasts outer side
// ================================================================

void recoverLine() {
  if (lastError <= 0) {
    // Line was last seen LEFT â†’ hard left turn
    moveForward(recoverSpeed, 0);
    Serial.println("RECOVER â†’ LEFT");
  } else {
    // Line was last seen RIGHT â†’ hard right turn
    moveForward(0, recoverSpeed);
    Serial.println("RECOVER â†’ RIGHT");
  }
}

// ================================================================
// ULTRASONIC
// ================================================================

int getDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 25000);
  if (dur == 0) return 999;
  return (int)(dur * 0.034f / 2);
}

bool handleObstacle(int dist) {
  if (!obstacleActive && dist > 0 && dist < OBSTACLE_CM) {
    obstacleActive = true;
    obstacleTimer  = millis();
    stopMotors();
    integral = 0; previousError = 0;
    Serial.print("OBSTACLE: "); Serial.print(dist); Serial.println("cm");
  }
  if (obstacleActive) {
    if (millis() - obstacleTimer < HOLD_MS) return true;
    int recheck = getDistance();
    if (recheck >= CLEAR_CM || recheck == 0) {
      obstacleActive = false;
      Serial.println("PATH CLEAR");
    } else {
      obstacleTimer = millis();
      return true;
    }
  }
  return false;
}

// ================================================================
// SETUP
// ================================================================

void setup() {
  Serial.begin(115200);

  pinMode(S1, INPUT);
  pinMode(S3, INPUT);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(RF_IN1, OUTPUT); digitalWrite(RF_IN1, HIGH);
  pinMode(RF_IN2, OUTPUT); digitalWrite(RF_IN2, LOW);

  pinMode(LF_IN1, OUTPUT); digitalWrite(LF_IN1, HIGH);
  pinMode(LF_IN2, OUTPUT); digitalWrite(LF_IN2, LOW);

  pinMode(RR_IN1, OUTPUT); digitalWrite(RR_IN1, HIGH);
  pinMode(RR_IN2, OUTPUT); digitalWrite(RR_IN2, LOW);

  pinMode(LR_IN1, OUTPUT); digitalWrite(LR_IN1, HIGH);
  pinMode(LR_IN2, OUTPUT); digitalWrite(LR_IN2, LOW);

  ledcSetup(CH_RF, 5000, 8); ledcAttachPin(RF_EN, CH_RF);
  ledcSetup(CH_LF, 5000, 8); ledcAttachPin(LF_EN, CH_LF);
  ledcSetup(CH_RR, 5000, 8); ledcAttachPin(RR_EN, CH_RR);
  ledcSetup(CH_LR, 5000, 8); ledcAttachPin(LR_EN, CH_LR);

  stopMotors();
  Serial.println("READY â€” 2 SENSOR MODE");
  delay(3000);

  // ================================================================
  // [ADDED] SENSOR NODE WIFI + UDP RECEIVE INIT
  // ================================================================
  sensor_setup();
}

// ================================================================
// LOOP
// ================================================================

void loop() {

  // â”€â”€ OBSTACLE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (handleObstacle(getDistance())) return;

  // â”€â”€ READ SENSORS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Invert: raw LOW on black â†’ 1, raw HIGH on white â†’ 0
  int s1 = !digitalRead(S1);   // LEFT
  int s3 = !digitalRead(S3);   // RIGHT

  // ================================================================
  // STATE MACHINE
  //
  //  s1  s3  |  meaning
  //  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  //   1   1  |  junction / both on line â†’ go straight
  //   1   0  |  line on LEFT  â†’ steer left
  //   0   1  |  line on RIGHT â†’ steer right
  //   0   0  |  line lost     â†’ recover
  // ================================================================

  // â”€â”€ JUNCTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (s1 == 1 && s3 == 1) {
    integral      = 0;
    previousError = 0;
    moveForward(baseSpeed, baseSpeed);
    Serial.println("JUNCTION");
    return;
  }

  // â”€â”€ LINE LOST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (s1 == 0 && s3 == 0) {
    recoverLine();
    return;
  }

  // â”€â”€ ERROR FROM 2 SENSORS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â°