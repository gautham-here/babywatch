# BabyWatch – AIoT Infant Attention Monitor

![Our Logo](./img/logo.png)

**BabyWatch** is a **privacy-first, edge-based infant monitoring system** that guides caregiver attention using audio and motion patterns instead of cameras or cloud streaming.

Phase 1 implements a working hardware demo on ESP32 with **rule-based intelligence** and a Wokwi simulation.

---

## Problem

- Parents and daycare staff juggle multiple infants and spaces with limited attention.
- Existing Wi‑Fi baby monitors and CCTV systems stream video to the cloud, creating **privacy, security, and surveillance** concerns.
- Caregivers face **alert fatigue** and cognitive overload from raw feeds instead of clear guidance.

---

## Solution Overview

BabyWatch runs entirely on an **ESP32 microcontroller**, continuously monitoring:

- **Audio intensity** (simulated via potentiometer)
- **Motion** (PIR sensor)
- **Time context** (e.g., time since last diaper change)

It classifies each state into:

- **SAFE**
- **MONITOR**
- **ATTENTION**

and surfaces **simple, explainable alerts** rather than video streams.

Phase 1 focuses on a **single-zone prototype** with a clear upgrade path to TinyML and multi-zone orchestration.

---

## Phase 1 – What’s Implemented

**Hardware (Wokwi Simulation)**

- ESP32 DevKit
- Potentiometer → simulates audio intensity (quiet → crying)
- PIR motion sensor → simulates infant/body movement
- 3 LEDs:
  - Green → SAFE  
  - Yellow → MONITOR  
  - Red → ATTENTION  
- Push button → resets “time since diaper change”

**Logic**

- Reads audio (0–1 normalized) and motion (present/absent).
- Tracks:
  - `audio_rms`
  - `motion_pct`
  - `motion_transitions`
  - `time since diaper change` (with time scaling for demo).
- Computes alert level:

  - **ATTENTION**  
    - High audio for sustained duration (simulated “crying”).

  - **MONITOR**  
    - Diaper overdue + fussy (simulated via timer + moderate audio).  
    - Unusual silence (very quiet + no motion).  
    - Restlessness (many motion transitions + moderate audio).

  - **SAFE**  
    - None of the above conditions.

- Updates LEDs and logs rich context over Serial:
  - Current state
  - Audio RMS
  - Motion %
  - Simulated time since last diaper change

---

## Demo Scenarios

1. **SAFE – Normal Sleep**
   - Potentiometer low (quiet).
   - PIR idle (no motion).
   - → Green LED ON, state = SAFE.

2. **MONITOR – Diaper Overdue / Discomfort**
   - Simulated “90+ minutes since diaper change” (time-scaled).
   - Potentiometer at moderate level (fussy).
   - → Yellow LED ON, state = MONITOR.

3. **ATTENTION – Sustained Distress**
   - Potentiometer high (loud) for several seconds.
   - → Red LED ON, state = ATTENTION.

A push button press resets the diaper timer during the demo.

---

## AI / ML Roadmap

**Phase 1 – Rule-Based (Current MVP)**  
- Time-series thresholds on audio RMS and motion.  
- State machine: SAFE → MONITOR → ATTENTION.  
- Temporal reasoning: sustained distress duration, time since diaper change.

**Phase 2 – TinyML (Planned)**  
- On-device audio classifier (cry vs. non‑cry).  
- TensorFlow Lite Micro on ESP32.  
- Sub‑100 ms inference, combined with rule-based logic.

**Phase 3+ – Future**  
- Multi-label behavior (sleep, distress types).  
- Federated learning for privacy-preserving model updates.

---

## Hardware Stack

- **Microcontroller**: ESP32 DevKit (dual-core, Wi‑Fi capable, suitable for TinyML).  
- **Sensors**:
  - MEMS microphone (real hardware; potentiometer in Wokwi).  
  - PIR motion sensor.  
- **Outputs**:
  - 3 status LEDs (SAFE / MONITOR / ATTENTION).  
  - Optional OLED display and Bluetooth for local notifications (Phase 2).  

Deployment form factor: compact wall/ceiling node in infant spaces, powered via USB or battery.

---

## Testing the implementation in Wokwi - Steps

1. Open the Wokwi project link (or import `wokwi-diagram.json`).
2. Upload `babywatch_phase1.ino`.
3. Click **Start Simulation**.
4. Open **Serial Monitor** to observe state changes.
5. Use:
   - Potentiometer → adjust “audio intensity”.
   - PIR → toggle motion.
   - Button → simulate diaper change (timer reset).

---

## Notes & Limitations

- This is a **concept and prototype**, not a medical device.
- It is designed as an **attention aid**, not a guarantee against critical events.
- No real baby data or personal data is stored or transmitted in Phase 1.

---

## Contact

- Team: **Data Dynamos**  
- Track: **AIoT / Edge AI** 
- Members: Sriram SV [See GitHub Here](https://github.com/IamRasengan) and Gautham R [See GitHub Here](https://github.com/gautham-here)

“**Guidance over surveillance. Intelligence at the edge.**”
