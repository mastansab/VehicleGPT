# VehicleGPT — On-Device Vehicle Intelligence & HealthEngine

**VehicleGPT** is an on-device automotive intelligence platform for Software Defined Vehicles (SDVs). It combines the **HealthEngine** predictive diagnostics core with a localized Small Language Model (SLM), providing real-time telemetry access, proactive failure warnings, and root-cause fault diagnosis in plain natural language — **100% offline with zero cloud reliance**.

> **Built for TELIPORT Season 4**

---

## 🌟 Key Architecture & Highlights

- **100% On-Device Processing**: Runs entirely local on in-vehicle compute (Raspberry Pi 5 + AI accelerator, Cuttlefish/QEMU virtual testbed, AWS Graviton, or Automotive Cockpit HPC domain controllers such as Snapdragon Ride or NVIDIA DRIVE).
- **Unified HealthEngine Core**: A single deterministic evaluation loop monitoring 9 key vehicle subsystems:
  1. `High-Voltage Battery & BMS` — Pack voltage, cell balance, internal resistance, undervoltage shutdown risk
  2. `Thermal Management & Cooling` — Coolant temp, thermal slope drift, boil-over prediction
  3. `Electric Motor & Powertrain` — RPM, throttle responsiveness, torque envelopes, thermal derating
  4. `Friction Brakes & ABS` — Individual wheel speeds, slip ratios, brake pad thickness prediction
  5. `Tyres & Rolling Resistance` — Diagonal wheel speed variances, pressure loss, rolling radius drift
  6. `ADAS & Forward Sensors` — Camera/radar occlusion, lane-keep disengagements, sensor latency
  7. `Body Latches & Closures` — Door latches, security locks, microswitch wear
  8. `12V Auxiliary Power` — Alternator ripple, battery resting voltage, cold-crank health
  9. `Gateway & Telematics Bus` — CAN bus load %, frame drop bursts, network health
- **Hardware & Software Safety Diode**: Physical/logical read-only telemetry tap. Inference models cannot write back to safety-critical steering, braking, or powertrain networks.
- **Strict Resource Budget**:
  - **Memory**: `< 2.0 GB RAM` total footprint (SLM 4-bit 1.18 GB, Telemetry Buffers 340 MB, HealthEngine Math 40 MB, UI 180 MB).
  - **Latency**: `< 1.4 s` time-to-first-token inference latency.
- **Intent Classifier**: Automatically classifies queries into `GENERAL` (e.g. "What is the capital of France?"), `VEHICLE_KNOWLEDGE` (e.g. "Explain regenerative braking"), `DIAGNOSTIC`, `PREDICTIVE`, or `VEHICLE_DATA`.
- **Session Time Machine**: Scrubbable timeline to review past telemetry envelopes and correlate causal chains.

---

## 🗺️ Website Route Map

VehicleGPT provides a client-side routed multi-page web application:

| Route | Page | Description |
|---|---|---|
| `/` | **Overview** | Product showcase: hero, trust badges, 4-step data pipeline, predict & pinpoint diagnostic story, and embedded live test console |
| `/demo` (or `/dashboard`) | **Interactive Console** | Full interactive digital twin dashboard with live SSE telemetry, 9 hotspot nodes, causal chain diagnosis, chat, scenarios, and timeline |
| `/architecture` | **System Blueprint** | Interactive SVG diagram with clickable block inspector (Data Ingestion, Safety Diode, Storage, HealthEngine, SLM) and Latency & RAM budget bars |
| `/use-cases` | **Operational Personas** | Dedicated workflows and live sample queries for Driver, Service Technician, and Fleet Operations |
| `/privacy` | **Privacy & Safety** | Technical details on zero-cloud guarantees, AES-256 storage, and functional safety domain isolation |
| `/docs` | **Developer Guide** | Quickstart commands, edge hardware matrix, and offline REST & SSE API references |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.com) running `llama3.2:3b` for enhanced natural-language generation:
  ```bash
  ollama pull llama3.2:3b
  ```
  *(If Ollama is not present, VehicleGPT automatically falls back to its built-in rule and statistical on-device reasoning engine).*

### 2. Installation & Run
```bash
# Clone and enter repository
cd Peykan-main

# Set up virtual environment
python -m venv venv
.\venv\Scripts\activate       # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install in editable mode
pip install -e .

# Launch VehicleGPT Edge Server on port 6278
peykan.exe demo --port 6278   # On Windows
# peykan demo --port 6278     # On Linux/macOS
```

Open your browser to:
👉 **[http://localhost:6278/](http://localhost:6278/)** (Overview)
👉 **[http://localhost:6278/demo](http://localhost:6278/demo)** (Interactive Digital Twin)
live demo URL : https://vehiclegpt-1.onrender.com

---

## 📊 Offline REST & SSE API

- **`GET /dashboard/stream`**: Server-Sent Events (SSE) streaming live vehicle telemetry snapshots every 500 ms.
- **`POST /dashboard/query`**: Query engine endpoint. Accepts JSON: `{"query": "Why did my range drop?", "persona": "driver"}`.
- **`GET /api/components`**: Returns health scores (0–100), statuses, live signals, and correlated causal chains.
- **`POST /api/scenarios`**: Injects test telemetry envelopes:
  - `normal`: Baseline steady highway driving
  - `battery_overheat`: Triggers thermal overtemp, undervoltage, and multi-subsystem causal chain
  - `coolant_leak`: Simulates radiator flow degradation
  - `highway_loss`: Simulates aerodynamic drag and battery SOC decline
  - `brake_wear`: Simulates asymmetric friction pad degradation
  - `adas_disengage`: Simulates radar optical occlusion and disengagement logging

---

## 🎨 Design System

- **Background**: `#F6F4EF` (warm editorial off-white) with dark mode `#0E1217`
- **Surfaces**: Crisp white cards with `#E5E2DB` 1px hairline borders and subtle depth shadows
- **Brand Accent**: Deep Teal `#0F766E` (`#14B8A6` in dark mode)
- **Status Colors**:
  - Green (`#15803D`): Nominal baseline
  - Amber (`#B45309`): Watch & precautionary alert
  - Red (`#B91C1C`): Critical fault & threshold breach
- **Typography**:
  - Display / Headlines: Newsreader / Editorial Serif
  - Interface / Body: Clean Humanist Sans (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto`)
  - Telemetry: Monospace (`ui-monospace, "JetBrains Mono", Consolas`) with tabular figures (`font-variant-numeric: tabular-nums`).

---

## 🛡️ License

Licensed under the MIT License. Built for TELIPORT Season 4.
