---
title: "VehicleAI — On-edge SDV assistant"
description: "Local vehicle intelligence: live CAN, OBD-II, UDS and J1939 telemetry searchable with natural language. No cloud required."
---

# VehicleAI

VehicleAI is an **AI-on-edge** assistant for Software Defined Vehicles. It
consumes live ECU streams, events and logs, and answers driver or service
questions in plain language — entirely on the vehicle.

## What it does

- Streams and decodes live **CAN frames** from a simulated multi-ECU bus.
- Runs **OBD-II**, **UDS** and **SAE J1939** diagnostic tools.
- Injects named **fault scenarios** for training and demos.
- Analyses and replays **recorded CAN logs**.
- Serves a **live dashboard** with KPI gauges and a natural-language assistant.
- Optionally uses a **local Ollama SLM**; otherwise an onboard retriever answers from live + 14-day trip memory.

## Quickstart

```bash
pip install -e .
peykan demo --port 6278
```

Open http://localhost:6278/dashboard and ask:

- Why has my driving range reduced over the last two weeks?
- When did fuel efficiency show the largest drop?
- Show all ADAS disengagement events from yesterday.
- What happened before the battery overheating warning?
- Compare energy consumption between city and highway driving.
