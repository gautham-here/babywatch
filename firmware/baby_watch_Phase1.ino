// EdgeGuard Phase 1 - AIoT Infant Safety Monitor
// Hardware Demo: Potentiometer = Audio, PIR = Motion, Button = Diaper Change

#define AUDIO_PIN 34     // Potentiometer (audio intensity)
#define MOTION_PIN 35    // PIR sensor
#define LED_SAFE 32      // Green LED
#define LED_MONITOR 33   // Yellow LED
#define LED_ATTENTION 26 // Red LED
#define DIAPER_BUTTON_PIN 25 // Push button for "diaper changed"

// --------- TIME SCALING FOR DEMO ---------
// 1 real second = TIME_SCALE simulated minutes
// Example: TIME_SCALE = 10 → 1s = 10min, so 90min ~ 9s
const float TIME_SCALE = 10.0;

// Global state
float audio_rms = 0.0;
float motion_pct = 0.0;
int motion_transitions = 0;
unsigned long last_diaper_change = 0;
unsigned long distress_start = 0;
bool last_motion = false;
bool last_button_state = HIGH; // for edge detection (using INPUT_PULLUP)

enum AlertLevel { SAFE, MONITOR, ATTENTION };
AlertLevel current_state = SAFE;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Setup pins
  pinMode(LED_SAFE, OUTPUT);
  pinMode(LED_MONITOR, OUTPUT);
  pinMode(LED_ATTENTION, OUTPUT);
  pinMode(MOTION_PIN, INPUT);
  pinMode(DIAPER_BUTTON_PIN, INPUT_PULLUP); // Button to GND

  // Turn off all LEDs initially
  digitalWrite(LED_SAFE, LOW);
  digitalWrite(LED_MONITOR, LOW);
  digitalWrite(LED_ATTENTION, LOW);

  Serial.println("=== EDGEGUARD ===");
  Serial.println("Potentiometer = Audio Intensity");
  Serial.println("PIR Sensor    = Motion Detection");
  Serial.println("Button (PIN 25) = Diaper Changed");
  Serial.println("LEDs: Green=SAFE, Yellow=MONITOR, Red=ATTENTION");
  Serial.print("Time scale: 1s real = ");
  Serial.print(TIME_SCALE);
  Serial.println(" min simulated");
  Serial.println("=================================\n");

  last_diaper_change = millis();
}

void loop() {
  // --- Handle diaper button (edge trigger) ---
  bool button_now = digitalRead(DIAPER_BUTTON_PIN); // LOW when pressed
  if (last_button_state == HIGH && button_now == LOW) {
    reset_diaper_timer();
    delay(200); // simple debounce
  }
  last_button_state = button_now;

  // Read sensors every 500ms
  static unsigned long last_read = 0;
  if (millis() - last_read < 500) return;
  last_read = millis();

  // === STEP 1: Read Audio (Potentiometer) ===
  int audio_raw = analogRead(AUDIO_PIN);  // 0-4095
  audio_rms = audio_raw / 4095.0;         // 0-1

  // === STEP 2: Read Motion (PIR) ===
  bool motion_now = digitalRead(MOTION_PIN);
  if (motion_now != last_motion) {
    motion_transitions++;
    last_motion = motion_now;
  }
  motion_pct = motion_now ? 100.0 : 0.0;

  // === STEP 3: Decision Logic ===
  AlertLevel new_state = decide_alert();

  // === STEP 4: Update if state changed ===
  if (new_state != current_state) {
    update_leds(new_state);
    log_event(new_state);
    current_state = new_state;
  }

  // === STEP 5: Debug output ===
  Serial.printf("RMS:%.2f | Motion:%.0f%% | Transitions:%d | State:%s\n",
                audio_rms, motion_pct, motion_transitions,
                new_state == SAFE ? "SAFE" :
                new_state == MONITOR ? "MONITOR" : "ATTENTION");
}

AlertLevel decide_alert() {
  unsigned long now = millis();

  // ATTENTION: High audio RMS sustained
  if (audio_rms > 0.7) {
    if (distress_start == 0) distress_start = now;
    if ((now - distress_start) > 5000) {  // 5 seconds sustained
      return ATTENTION;
    }
  } else {
    distress_start = 0;
  }

  // --- Convert elapsed millis to "simulated minutes" using TIME_SCALE ---
  float elapsed_real_ms = (float)(now - last_diaper_change);
  float diaper_min = (elapsed_real_ms / 60000.0f) * TIME_SCALE;
  // Example: TIME_SCALE=10 → 6000ms (~6s) ≈ 1 simulated hour

  // 1. Diaper overdue + fussy (use simulated minutes)
  if (diaper_min > 90.0 && audio_rms > 0.4) {
    return MONITOR;
  }

  // 2. Unusual silence (no motion + very quiet)
  if (motion_pct < 10 && audio_rms < 0.15) {
    return MONITOR;
  }

  // 3. Restlessness (many motion changes)
  if (motion_transitions > 7 && audio_rms > 0.3) {
    return MONITOR;
  }

  // Default: SAFE
  return SAFE;
}

void update_leds(AlertLevel state) {
  digitalWrite(LED_SAFE, state == SAFE ? HIGH : LOW);
  digitalWrite(LED_MONITOR, state == MONITOR ? HIGH : LOW);
  digitalWrite(LED_ATTENTION, state == ATTENTION ? HIGH : LOW);
}

void log_event(AlertLevel state) {
  // Compute simulated minutes for logging
  float elapsed_real_ms = (float)(millis() - last_diaper_change);
  float diaper_min = (elapsed_real_ms / 60000.0f) * TIME_SCALE;

  Serial.println("\n🚨 ALERT CHANGE! 🚨");
  Serial.printf("State: %s\n",
                state == SAFE ? "✅ SAFE" :
                state == MONITOR ? "⚠️ MONITOR" : "🚨 ATTENTION");
  Serial.printf("Audio RMS: %.2f\n", audio_rms);
  Serial.printf("Motion: %.0f%%\n", motion_pct);
  Serial.printf("Simulated time since diaper: %.1f min\n", diaper_min);
  Serial.println("====================\n");
}

void reset_diaper_timer() {
  last_diaper_change = millis();
  Serial.println("\n👶 Diaper changed — timer reset to 0 min\n");
}
