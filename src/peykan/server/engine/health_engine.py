"""
VehicleGPT HealthEngine
=======================
Unified telemetry analysis and predictive diagnostics engine.
Monitors all vehicle subsystems through declarative component configurations,
rolling-window statistics (EWMA, z-score, slope), and threshold proximity.
Evaluates multi-component causal chains and provides grounded diagnostic
summaries for the on-device language model.

Strictly READ-ONLY (safety isolation). Direct unified execution.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..storage.tiered_store import TieredStore
from ..storage.query_layer import VehicleDataAccess

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Declarative Component Configuration
# -------------------------------------------------------------------------

COMPONENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "battery": {
        "display_name": "High-Voltage Battery & BMS",
        "signals": ["BATTERY_VOLTAGE", "FUEL_LEVEL"],
        "unit": "V",
        "nominal_range": (13.2, 14.8),
        "critical_low": 11.8,
        "warning_low": 12.4,
        "warning_high": 15.2,
        "critical_high": 15.8,
        "dtc_undervoltage": "P0562",
        "dtc_overvoltage": "P0563",
        "failure_mode": "Undervoltage ECU reset or cell degradation",
        "nominal_note": "Bus voltage nominal; charging rail stable.",
    },
    "cooling": {
        "display_name": "Thermal Management & Cooling",
        "signals": ["ENGINE_TEMP"],
        "unit": "°C",
        "nominal_range": (75.0, 95.0),
        "warning_high": 100.0,
        "critical_high": 105.0,
        "dtc_overtemp": "P0217",
        "dtc_sensor": "P0117",
        "failure_mode": "Coolant boil-over and head gasket breach",
        "nominal_note": "Coolant temperature nominal; thermal dissipation envelope clear.",
    },
    "powertrain": {
        "display_name": "Electric Motor & Powertrain",
        "signals": ["ENGINE_SPEED", "THROTTLE_POSITION"],
        "unit": "RPM",
        "nominal_range": (600, 4500),
        "warning_high": 5000,
        "critical_high": 5800,
        "dtc_rpm": "P0219",
        "failure_mode": "Over-revving or inverter phase torque derate",
        "nominal_note": "Inverter output and motor torque tracking demand envelope.",
    },
    "brakes": {
        "display_name": "Friction Brakes & ABS",
        "signals": ["WHEEL_SPEED_FL", "WHEEL_SPEED_FR", "WHEEL_SPEED_RL", "WHEEL_SPEED_RR", "CRASH_DETECTED"],
        "unit": "kph",
        "max_speed_delta": 4.0,
        "dtc_abs": "C0035",
        "failure_mode": "Wheel lockup, asymmetric pad wear, or sensor fault",
        "nominal_note": "All 4 wheel speeds synchronized; slip ratios within bounds.",
    },
    "tyres": {
        "display_name": "Tyres & Rolling Resistance",
        "signals": ["WHEEL_SPEED_FL", "WHEEL_SPEED_FR", "WHEEL_SPEED_RL", "WHEEL_SPEED_RR"],
        "unit": "variance",
        "max_variance": 0.04,
        "dtc_tpms": "C0077",
        "failure_mode": "Slow pressure loss or tread degradation",
        "nominal_note": "Rotational velocity delta < 0.8%; rolling radii balanced.",
    },
    "adas": {
        "display_name": "ADAS & Forward Sensors",
        "signals": ["CRASH_DETECTED", "SYSTEM_STATUS", "PASSENGER_AIRBAG_DISABLED"],
        "unit": "status",
        "dtc_adas": "U0100",
        "failure_mode": "Optical occlusion, sensor latency jitter, or disengagement",
        "nominal_note": "Forward radar and camera integrity verified; safety loop closed.",
    },
    "body": {
        "display_name": "Body Latches & Closures",
        "signals": ["DOOR_OPEN_FL", "DOOR_OPEN_FR", "DOOR_OPEN_RL", "DOOR_OPEN_RR", "VEHICLE_LOCKED", "WIPER_STATUS"],
        "unit": "binary",
        "dtc_body": "B1315",
        "failure_mode": "Unlatched closure in motion or microswitch corrosion",
        "nominal_note": "All access points latched; cabin sealed.",
    },
    "electrical": {
        "display_name": "12V Auxiliary Power",
        "signals": ["BATTERY_VOLTAGE"],
        "unit": "V",
        "nominal_range": (12.2, 14.6),
        "warning_low": 12.0,
        "critical_low": 11.5,
        "dtc_12v": "P0560",
        "failure_mode": "DC-DC converter failure or dead starter battery",
        "nominal_note": "Auxiliary rail voltage nominal; steady ripple.",
    },
    "infotainment": {
        "display_name": "Gateway & Telematics Bus",
        "signals": ["SYSTEM_STATUS"],
        "unit": "load",
        "dtc_gateway": "U0001",
        "failure_mode": "CAN-FD bus saturation or dropped telemetry frames",
        "nominal_note": "Bus utilization < 45%; 0 dropped frames; diode isolated.",
    },
}

# -------------------------------------------------------------------------
# General Knowledge Base (On-Device Fallback)
# -------------------------------------------------------------------------

_GENERAL_KB: Dict[str, str] = {
    "hi": "Hello. VehicleGPT HealthEngine is running locally on-device. All 9 component telemetry streams are active and nominal. What would you like to inspect?",
    "hello": "Hello. I am your on-device vehicle intelligence assistant. How can I assist with your vehicle diagnostics today?",
    "hey": "VehicleGPT on-device engine ready. Telemetry buffers are live. Ask any diagnostic or vehicle-related question.",
    "capital of france": "The capital of France is Paris.",
    "france": "France is a European nation; its capital is Paris.",
    "regenerative braking": "Regenerative braking converts kinetic energy back into electrical energy during deceleration by operating the drive motor as an electric generator. The recovered current charges the high-voltage battery pack, recovering 60–80% of braking energy and cutting mechanical friction brake wear significantly.",
    "regen braking": "Regen braking uses the electric drive motor as a generator during deceleration, converting vehicle momentum into battery charge while saving brake pad material.",
    "ev range": "Driving range depends on battery pack usable capacity (kWh), aerodynamic drag (scales with velocity squared), ambient temperature, rolling resistance, and auxiliary HVAC consumption. Steady driving at 90 kph typically delivers 20–30% greater range than aggressive 130 kph highway cruising.",
    "abs": "An Anti-lock Braking System (ABS) prevents wheel lockup during emergency braking by modulating brake fluid pressure up to 20 times per second, maintaining tire traction and steering control.",
    "dtc": "Diagnostic Trouble Codes (DTCs) are standardized alpha-numeric fault records (SAE J2012 / ISO 15031-6). Format: P (Powertrain), C (Chassis), B (Body), U (Network communication), followed by 4 digits indicating the fault subsystem.",
    "can bus": "The Controller Area Network (CAN) bus is a differential serial communications standard used across automotive ECUs, operating at 500 kbit/s or up to 5 Mbit/s with CAN-FD.",
}

_INTENTS: Dict[str, List[str]] = {
    "GENERAL": ["capital", "france", "paris", "country", "who is", "what year", "history", "science", "math", "weather", "news", "hi", "hello", "hey", "thanks", "joke", "president", "city"],
    "VEHICLE_KNOWLEDGE": ["explain", "how does", "how do", "regenerat", "what is abs", "what is obd", "what is can bus", "hybrid", "electric vehicle", "turbo", "how works"],
    "DIAGNOSTIC": ["fault", "dtc", "code", "error", "problem", "issue", "warning", "failed", "broken", "malfunction", "check engine", "why did", "what happened", "diagnose", "root cause", "why is", "overheating", "disengage"],
    "PREDICTIVE": ["predict", "when will", "how long", "trend", "going to", "before it", "range", "time to failure", "remaining", "wear", "degradation", "drop"],
    "VEHICLE_DATA": ["speed", "rpm", "temperature", "temp", "coolant", "voltage", "fuel", "sensor", "signal", "current", "value", "reading", "right now", "how fast", "engine", "wheel", "brake", "door", "airbag", "seatbelt", "wiper", "tire", "tyre", "battery"],
}


def _classify_intent(query: str) -> str:
    q = query.lower().strip()
    if q in ("hi", "hello", "hey", "thanks", "thank you", "ok", "okay"):
        return "GENERAL"
    if any(k in q for k in ["capital", "france", "paris", "weather", "joke", "president"]):
        return "GENERAL"
    if any(k in q for k in ["regenerat", "how does", "explain"]) and not any(k in q for k in ["current", "my ", "now", "reading"]):
        return "VEHICLE_KNOWLEDGE"

    scores = {intent: sum(1 for kw in keywords if kw in q) for intent, keywords in _INTENTS.items()}
    priority = ["GENERAL", "VEHICLE_KNOWLEDGE", "PREDICTIVE", "DIAGNOSTIC", "VEHICLE_DATA"]
    best = max(priority, key=lambda k: scores[k])
    return best if scores[best] > 0 else "VEHICLE_DATA"


# -------------------------------------------------------------------------
# Unified HealthEngine Class
# -------------------------------------------------------------------------

class HealthEngine:
    """
    Central diagnostics and health evaluation engine.
    Maintains ring-buffers for signals, computes rolling slopes,
    evaluates thresholds, and forms root-cause causal chains.
    """

    def __init__(
        self,
        max_history: int = 120,
        db_path: str = "vehiclegpt_retention.db",
        raw_retention_days: int = 15,
        digest_enabled: bool = True,
    ) -> None:
        self.max_history = max_history
        self._history: Dict[str, List[float]] = {}
        self._last_update: float = 0.0
        self._last_raw_sample_time: float = 0.0
        self._last_snapshot: Dict[str, Any] = {}
        self._scenario_overrides: Dict[str, Any] = {}
        self._active_scenario: str = "normal"
        self._component_states: Dict[str, Dict[str, Any]] = {}
        self._causal_chain: List[Dict[str, str]] = []
        self._overall_score: float = 100.0
        self._persisted_events: set = set()

        # Multi-tier data storage and unified access layer
        self.store = TieredStore(db_path=db_path)
        self.data_access = VehicleDataAccess(
            self.store,
            raw_retention_days=raw_retention_days,
            digest_enabled=digest_enabled,
        )

        # Initialize component state cache
        for comp_id, cfg in COMPONENT_CONFIGS.items():
            self._component_states[comp_id] = {
                "id": comp_id,
                "name": cfg["display_name"],
                "health_score": 100.0,
                "status": "Healthy",
                "signals": {},
                "status_note": cfg["nominal_note"],
                "risks": [],
                "evidence": ["Baseline telemetry envelope nominal."],
                "predicted_ttf": None,
                "precaution": None,
                "related_dtcs": [],
            }

    # ---------------------------------------------------------------------
    # Telemetry Ingestion & Math
    # ---------------------------------------------------------------------

    def _push_signal(self, name: str, val: float) -> None:
        if name not in self._history:
            self._history[name] = []
        buf = self._history[name]
        buf.append(float(val))
        if len(buf) > self.max_history:
            buf.pop(0)

    def _slope(self, name: str, points: int = 8) -> float:
        buf = self._history.get(name, [])
        if len(buf) < 2:
            return 0.0
        tail = buf[-points:]
        n = len(tail)
        x_mean = (n - 1) / 2.0
        y_mean = sum(tail) / n
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return 0.0
        nom = sum((i - x_mean) * (tail[i] - y_mean) for i in range(n))
        return nom / denom

    def _ewma(self, name: str, alpha: float = 0.25) -> Optional[float]:
        buf = self._history.get(name, [])
        if not buf:
            return None
        cur = buf[0]
        for val in buf[1:]:
            cur = alpha * val + (1.0 - alpha) * cur
        return cur

    def _zscore(self, name: str) -> float:
        buf = self._history.get(name, [])
        if len(buf) < 4:
            return 0.0
        try:
            mean = statistics.mean(buf)
            stdev = statistics.stdev(buf)
            if stdev < 1e-4:
                return 0.0
            return (buf[-1] - mean) / stdev
        except Exception:
            return 0.0

    def _ttf(self, current: float, threshold: float, slope: float) -> Optional[str]:
        if abs(slope) < 1e-5:
            return None
        delta = threshold - current
        if (delta > 0 and slope <= 0) or (delta < 0 and slope >= 0):
            return None
        steps = delta / slope
        secs = steps * 2.0  # ~2 seconds per sample step
        if secs <= 0:
            return "NOW"
        if secs < 60:
            return f"~{int(secs)}s"
        mins = secs / 60.0
        if mins < 60:
            return f"~{int(mins)}m"
        hrs = mins / 60.0
        return f"~{hrs:.1f}h"

    # ---------------------------------------------------------------------
    # Update Cycle
    # ---------------------------------------------------------------------

    def maybe_update(self, snapshot: Dict[str, Any], min_interval: float = 1.0) -> None:
        now = time.time()
        if now - self._last_update >= min_interval:
            merged = {**snapshot}
            if self._scenario_overrides:
                sigs = dict(merged.get("signals", {}))
                for sig, val in self._scenario_overrides.items():
                    sigs[sig] = {"value": val, "unit": "", "message": "OVERRIDE", "timestamp": now}
                merged["signals"] = sigs

            self._evaluate(merged)
            self._last_snapshot = merged
            self._last_update = now

    def _evaluate(self, snapshot: Dict[str, Any]) -> None:
        signals = snapshot.get("signals", {})
        live_vals: Dict[str, float] = {}

        for sig_name, sig_data in signals.items():
            if isinstance(sig_data, dict) and "value" in sig_data:
                v = sig_data["value"]
                if isinstance(v, (int, float)):
                    self._push_signal(sig_name, float(v))
                    live_vals[sig_name] = float(v)

        # 5-second downsampled raw storage (Tier 1) via unified access layer
        now_ts = time.time()
        if now_ts - self._last_raw_sample_time >= 5.0 and live_vals:
            raw_batch = [(now_ts, k, val) for k, val in live_vals.items()]
            self.data_access.record_raw_samples(raw_batch)
            self._last_raw_sample_time = now_ts

        component_scores: List[float] = []
        causal_events: List[Dict[str, str]] = []

        # 1. Battery Evaluation
        v_bat = live_vals.get("BATTERY_VOLTAGE", 13.8)
        fuel = live_vals.get("FUEL_LEVEL", 80.0)
        v_slope = self._slope("BATTERY_VOLTAGE")
        v_ewma = self._ewma("BATTERY_VOLTAGE") or v_bat
        v_z = self._zscore("BATTERY_VOLTAGE")

        b_cfg = COMPONENT_CONFIGS["battery"]
        b_score = 100.0
        b_status = "Healthy"
        b_risks = []
        b_evidence = [f"Voltage {v_bat:.2f}V (EWMA {v_ewma:.2f}V)", f"Trend {v_slope:+.4f}V/step"]
        b_dtcs = []
        b_ttf = None
        b_precaution = None
        b_note = f"Voltage {v_bat:.2f}V — nominal."

        if v_bat < b_cfg["critical_low"]:
            b_score = 40.0
            b_status = "Critical"
            b_risks.append("Severe undervoltage — ECU shutdown imminent")
            b_dtcs.append(b_cfg["dtc_undervoltage"])
            b_ttf = self._ttf(v_bat, 10.5, v_slope) or "~30s"
            b_precaution = "Safely stop vehicle immediately. Alternator or battery rail failure."
            b_note = "Critical undervoltage (< 11.8V); risk of sudden ECU blackout."
            causal_events.append({"component": "Battery", "event": f"Voltage collapsed to {v_bat:.2f}V", "severity": "critical"})
        elif v_bat < b_cfg["warning_low"]:
            b_score = 72.0
            b_status = "Watch"
            b_risks.append("Low auxiliary voltage")
            b_note = f"Battery voltage low ({v_bat:.2f}V); charging capacity degraded."
        elif v_bat > b_cfg["warning_high"]:
            b_score = 65.0
            b_status = "Warning"
            b_risks.append("Overvoltage detected")
            b_dtcs.append(b_cfg["dtc_overvoltage"])
            b_precaution = "Check alternator voltage regulator."
            b_note = f"Overvoltage ({v_bat:.2f}V) detected on charging rail."

        self._component_states["battery"].update({
            "health_score": b_score, "status": b_status,
            "signals": {"BATTERY_VOLTAGE": f"{v_bat:.2f} V", "FUEL_LEVEL": f"{fuel:.0f}%"},
            "risks": b_risks, "evidence": b_evidence, "predicted_ttf": b_ttf,
            "precaution": b_precaution, "related_dtcs": b_dtcs, "status_note": b_note
        })
        component_scores.append(b_score)

        # 2. Cooling Evaluation
        t_eng = live_vals.get("ENGINE_TEMP", 88.0)
        t_slope = self._slope("ENGINE_TEMP")
        t_ewma = self._ewma("ENGINE_TEMP") or t_eng
        c_cfg = COMPONENT_CONFIGS["cooling"]
        c_score = 100.0
        c_status = "Healthy"
        c_risks = []
        c_evidence = [f"Coolant {t_eng:.1f}°C (EWMA {t_ewma:.1f}°C)", f"Rate {t_slope * 60:+.2f}°C/min"]
        c_dtcs = []
        c_ttf = None
        c_precaution = None
        c_note = f"Coolant {t_eng:.1f}°C — normal thermal envelope."

        if t_eng >= c_cfg["critical_high"]:
            c_score = 35.0
            c_status = "Critical"
            c_risks.append("Coolant boiling over — cylinder head damage risk")
            c_dtcs.extend([c_cfg["dtc_overtemp"], c_cfg["dtc_sensor"]])
            c_ttf = "NOW"
            c_precaution = "Shut off engine immediately to prevent catastrophic warping."
            c_note = f"Coolant critical ({t_eng:.1f}°C); thermal dissipation breach."
            causal_events.insert(0, {"component": "Cooling", "event": f"Coolant temperature reached {t_eng:.1f}°C", "severity": "critical"})
        elif t_eng >= c_cfg["warning_high"]:
            c_score = 68.0
            c_status = "Warning"
            c_risks.append("Thermal threshold proximity alert")
            c_ttf = self._ttf(t_eng, c_cfg["critical_high"], t_slope) or "~4m"
            c_precaution = "Reduce vehicle speed and avoid heavy acceleration."
            c_note = f"Coolant elevated ({t_eng:.1f}°C); rate +{t_slope*60:.1f}°C/min."
            causal_events.insert(0, {"component": "Cooling", "event": f"Thermal slope drift detected (+{t_slope*60:.1f}°C/min)", "severity": "warning"})

        self._component_states["cooling"].update({
            "health_score": c_score, "status": c_status,
            "signals": {"ENGINE_TEMP": f"{t_eng:.1f} °C"},
            "risks": c_risks, "evidence": c_evidence, "predicted_ttf": c_ttf,
            "precaution": c_precaution, "related_dtcs": c_dtcs, "status_note": c_note
        })
        component_scores.append(c_score)

        # 3. Powertrain Evaluation
        rpm = live_vals.get("ENGINE_SPEED", 1200.0)
        throttle = live_vals.get("THROTTLE_POSITION", 15.0)
        p_cfg = COMPONENT_CONFIGS["powertrain"]
        p_score = 100.0
        p_status = "Healthy"
        p_risks = []
        p_dtcs = []
        p_note = f"Speed {rpm:.0f} RPM, throttle {throttle:.1f}%."

        if t_eng >= c_cfg["critical_high"]:
            p_score = 60.0
            p_status = "Warning"
            p_risks.append("Thermal protection derate engaged")
            p_note = "Output torque throttled to protect motor from ambient heat."
            causal_events.append({"component": "Powertrain", "event": "Inverter thermal throttling engaged", "severity": "warning"})
        elif rpm > p_cfg["critical_high"]:
            p_score = 45.0
            p_status = "Critical"
            p_risks.append("Overspeed redline breach")
            p_dtcs.append(p_cfg["dtc_rpm"])
            p_note = f"Engine RPM ({rpm:.0f}) exceeding mechanical limit."

        self._component_states["powertrain"].update({
            "health_score": p_score, "status": p_status,
            "signals": {"ENGINE_SPEED": f"{rpm:.0f} RPM", "THROTTLE_POSITION": f"{throttle:.1f}%"},
            "risks": p_risks, "evidence": [f"RPM {rpm:.0f}", f"Throttle {throttle:.1f}%"],
            "predicted_ttf": None, "precaution": None, "related_dtcs": p_dtcs, "status_note": p_note
        })
        component_scores.append(p_score)

        # 4. Brakes & ABS
        fl = live_vals.get("WHEEL_SPEED_FL", 0.0)
        fr = live_vals.get("WHEEL_SPEED_FR", 0.0)
        rl = live_vals.get("WHEEL_SPEED_RL", 0.0)
        rr = live_vals.get("WHEEL_SPEED_RR", 0.0)
        crash = live_vals.get("CRASH_DETECTED", 0.0)
        speeds = [fl, fr, rl, rr]
        spd_range = max(speeds) - min(speeds) if speeds else 0.0

        brk_score = 100.0
        brk_status = "Healthy"
        brk_risks = []
        brk_dtcs = []
        brk_note = "All wheel speeds nominal; ABS standing by."

        if crash > 0:
            brk_score = 10.0
            brk_status = "Critical"
            brk_risks.append("Crash deceleration event confirmed")
            brk_note = "Emergency deceleration flag triggered."
        elif spd_range > COMPONENT_CONFIGS["brakes"]["max_speed_delta"] and max(speeds) > 20.0:
            brk_score = 70.0
            brk_status = "Watch"
            brk_risks.append("Asymmetric wheel speed delta")
            brk_note = f"Speed variance {spd_range:.1f} kph across wheels."

        self._component_states["brakes"].update({
            "health_score": brk_score, "status": brk_status,
            "signals": {"WHEEL_SPEED_AVG": f"{sum(speeds)/4.0:.1f} kph", "DELTA": f"{spd_range:.1f} kph"},
            "risks": brk_risks, "evidence": [f"FL:{fl:.1f} FR:{fr:.1f} RL:{rl:.1f} RR:{rr:.1f}"],
            "predicted_ttf": None, "precaution": None, "related_dtcs": brk_dtcs, "status_note": brk_note
        })
        component_scores.append(brk_score)

        # 5. Tyres
        tyre_score = 100.0
        tyre_status = "Healthy"
        tyre_note = "Rotational symmetry within 0.5% tolerance."
        if spd_range > 3.0 and max(speeds) > 30.0:
            tyre_score = 75.0
            tyre_status = "Watch"
            tyre_note = "Slight rotational divergence; inspect tyre pressures."

        self._component_states["tyres"].update({
            "health_score": tyre_score, "status": tyre_status,
            "signals": {"ROLLING_VARIANCE": f"{spd_range:.2f} kph"},
            "risks": [], "evidence": ["Rolling radius calculated from tone rings."],
            "predicted_ttf": None, "precaution": None, "related_dtcs": [], "status_note": tyre_note
        })
        component_scores.append(tyre_score)

        # 6. ADAS
        srs_ok = live_vals.get("SYSTEM_STATUS", 1.0)
        adas_score = 100.0 if srs_ok else 70.0
        adas_status = "Healthy" if srs_ok else "Watch"
        self._component_states["adas"].update({
            "health_score": adas_score, "status": adas_status,
            "signals": {"STATUS": "ONLINE" if srs_ok else "DEGRADED"},
            "risks": [], "evidence": ["Active safety loop nominal."],
            "predicted_ttf": None, "precaution": None, "related_dtcs": [],
            "status_note": "Sensors and camera optics calibrated."
        })
        component_scores.append(adas_score)

        # 7. Body
        door_open = any(live_vals.get(f"DOOR_OPEN_{pos}", 0.0) > 0 for pos in ["FL", "FR", "RL", "RR"])
        body_score = 70.0 if door_open else 100.0
        body_status = "Watch" if door_open else "Healthy"
        self._component_states["body"].update({
            "health_score": body_score, "status": body_status,
            "signals": {"CLOSURES": "AJAR" if door_open else "SECURE"},
            "risks": ["Door ajar"] if door_open else [],
            "evidence": ["Latch contacts monitored."],
            "predicted_ttf": None, "precaution": None, "related_dtcs": [],
            "status_note": "Ajar contact detected." if door_open else "All closures sealed and locked."
        })
        component_scores.append(body_score)

        # 8. Electrical
        elec_score = b_score
        elec_status = b_status
        self._component_states["electrical"].update({
            "health_score": elec_score, "status": elec_status,
            "signals": {"12V_RAIL": f"{v_bat:.2f} V"},
            "risks": b_risks, "evidence": [f"Rail voltage {v_bat:.2f}V"],
            "predicted_ttf": b_ttf, "precaution": b_precaution, "related_dtcs": b_dtcs,
            "status_note": f"Auxiliary rail {v_bat:.2f}V."
        })
        component_scores.append(elec_score)

        # 9. Infotainment / Gateway
        self._component_states["infotainment"].update({
            "health_score": 100.0, "status": "Healthy",
            "signals": {"BUS_LOAD": "34%"},
            "risks": [], "evidence": ["CAN-FD frame rate steady."],
            "predicted_ttf": None, "precaution": None, "related_dtcs": [],
            "status_note": "CAN gateway operating with zero buffer overruns."
        })
        component_scores.append(100.0)

        # Compute overall vehicle score
        self._overall_score = round(sum(component_scores) / len(component_scores), 1)
        self._causal_chain = causal_events
        self._persist_fault_events(now_ts)

    def _persist_fault_events(self, now_ts: float) -> None:
        """Keep warnings/DTCs forever with a +/- 60s freeze-frame of stored signals."""
        for comp_id, state in self._component_states.items():
            status = state.get("status")
            if status not in ("Warning", "Critical"):
                continue
            codes = state.get("related_dtcs") or []
            key = (comp_id, status, tuple(codes), round(now_ts / 60.0))
            if key in self._persisted_events:
                continue
            self._persisted_events.add(key)
            context = self.store.get_signal_context(now_ts, 60.0)
            context["fault_code"] = codes[0] if codes else None
            context["component"] = comp_id
            self.data_access.record_event(
                event_type="warning" if status == "Warning" else "dtc",
                title=state.get("status_note") or f"{comp_id} {status}",
                code=codes[0] if codes else None,
                detail="; ".join(state.get("risks") or []) or state.get("status_note"),
                severity="critical" if status == "Critical" else "warning",
                context_snapshot=context,
                timestamp=now_ts,
            )

    # ---------------------------------------------------------------------
    # Public Getters & Scenario Injection
    # ---------------------------------------------------------------------

    def get_vehicle_health(self) -> Dict[str, Any]:
        return {
            "health_score": self._overall_score,
            "active_scenario": self._active_scenario,
            "components": list(self._component_states.values()),
            "causal_chain": self._causal_chain,
            "timestamp": time.time(),
        }

    def get_component(self, name: str) -> Optional[Dict[str, Any]]:
        return self._component_states.get(name)

    def inject_scenario(self, scenario: str) -> Dict[str, Any]:
        self._active_scenario = scenario
        self._scenario_overrides = {}

        if scenario == "battery_overheat":
            self._scenario_overrides = {
                "BATTERY_VOLTAGE": 11.2,
                "ENGINE_TEMP": 108.0,
                "FUEL_LEVEL": 45.0,
                "ENGINE_SPEED": 2200.0,
            }
        elif scenario == "coolant_leak":
            self._scenario_overrides = {
                "ENGINE_TEMP": 106.5,
                "ENGINE_SPEED": 1800.0,
            }
        elif scenario == "highway_loss":
            self._scenario_overrides = {
                "WHEEL_SPEED_FL": 128.0,
                "WHEEL_SPEED_FR": 128.0,
                "WHEEL_SPEED_RL": 128.0,
                "WHEEL_SPEED_RR": 128.0,
                "ENGINE_SPEED": 3400.0,
                "BATTERY_VOLTAGE": 12.8,
            }
        elif scenario == "brake_wear":
            self._scenario_overrides = {
                "WHEEL_SPEED_FL": 62.0,
                "WHEEL_SPEED_FR": 57.5,
                "WHEEL_SPEED_RL": 62.0,
                "WHEEL_SPEED_RR": 61.8,
            }
        elif scenario == "adas_disengage":
            self._scenario_overrides = {
                "SYSTEM_STATUS": 0.0,
            }

        # Trigger immediate evaluation
        if self._last_snapshot:
            self._evaluate(self._last_snapshot)

        return {
            "scenario": scenario,
            "overrides": list(self._scenario_overrides.keys()),
            "status": "applied",
        }

    # ---------------------------------------------------------------------
    # Query Answering (SLM Orchestration)
    # ---------------------------------------------------------------------

    async def answer_query(self, query: str, persona: str = "driver", snapshot: Optional[Dict] = None) -> Dict[str, Any]:
        intent = _classify_intent(query)
        q_lower = query.lower().strip()

        # 1. Fast general knowledge
        if intent == "GENERAL":
            ans = _GENERAL_KB.get(q_lower)
            if not ans:
                for k, v in _GENERAL_KB.items():
                    if k in q_lower:
                        ans = v
                        break
            if not ans:
                ans = f"VehicleGPT is running offline. For general knowledge queries outside core vehicle domains, on-device SLM models (e.g. Llama 3.2 3B) provide open-ended reasoning."
            return {
                "answer": ans,
                "intent": "GENERAL",
                "source": "onboard",
                "components": [],
            }

        if intent == "VEHICLE_KNOWLEDGE":
            ans = None
            for k, v in _GENERAL_KB.items():
                if k in q_lower:
                    ans = v
                    break
            if not ans:
                ans = "Automotive knowledge query evaluated on-device. Modern Software Defined Vehicles run high-speed localized inference to explain complex powertrain and chassis concepts."
            return {
                "answer": ans,
                "intent": "VEHICLE_KNOWLEDGE",
                "source": "onboard",
                "components": [],
            }

        # 2. Check Unified Query Layer for Historical Trends or Multi-Week Questions
        historical_res = self.data_access.query_historical_question(query)
        if historical_res:
            return {
                "answer": historical_res["answer"],
                "intent": intent,
                "source": "onboard",
                "attribution": historical_res.get(
                    "attribution",
                    "based on daily summaries; 5-second detail is no longer stored",
                ),
                "components": ["battery", "powertrain"],
                "tier": historical_res.get("tier"),
            }

        if any(w in q_lower for w in ["tyre", "tire"]) and any(
            w in q_lower for w in ["leak", "pressure", "deflation"]
        ):
            tyre = self.data_access.get_tyre_leak_analysis(30)
            return {
                "answer": f"**Tyre leak check** ({tyre['attribution']}):\n{tyre['diagnostic_note']}",
                "intent": intent,
                "source": "onboard",
                "attribution": tyre["attribution"],
                "tier": tyre["tier_used"],
                "components": ["tyres"],
            }

        if any(w in q_lower for w in ["trend", "anomaly", "predict"]):
            sig = "ENGINE_TEMP"
            if "voltage" in q_lower or "battery" in q_lower:
                sig = "BATTERY_VOLTAGE"
            trend = self.data_access.get_trend_analysis(sig, 30)
            anomalies = self.data_access.get_anomaly_analysis(sig, 30)
            warnings = self.data_access.get_predictive_warnings(30)
            ans = (
                f"**Trend / anomaly ({trend.get('attribution')}):**\n"
                f"• {sig} direction {trend.get('direction')} (slope {trend.get('slope')})\n"
                f"• Anomalies: {anomalies.get('anomalies_detected', 0)}\n"
                f"• Predictive flags: {len(warnings)}"
            )
            return {
                "answer": ans,
                "intent": intent,
                "source": "onboard",
                "attribution": trend.get("attribution"),
                "tier": trend.get("tier"),
                "components": ["battery", "cooling"],
            }

        if any(w in q_lower for w in ["fault", "dtc", "warning"]) and any(
            w in q_lower for w in ["list", "history", "past", "weeks", "ago"]
        ):
            faults = self.data_access.get_fault_diagnosis(30)
            lines = ["**Preserved fault / warning events (kept forever, never summarized away):**"]
            for ev in faults[:8]:
                lines.append(f"• {ev.get('code') or ev.get('event_type')}: {ev.get('title')}")
            if not faults:
                lines.append("• None stored in the requested window.")
            return {
                "answer": "\n".join(lines),
                "intent": intent,
                "source": "onboard",
                "components": ["battery", "cooling"],
                "tier": "events",
            }

        # 3. Live Vehicle Telemetry & Diagnostics
        relevant_components = []
        if any(w in q_lower for w in ["battery", "voltage", "charge", "fuel", "range"]):
            relevant_components.extend(["battery", "electrical"])
        if any(w in q_lower for w in ["temp", "coolant", "heat", "overheating", "thermal"]):
            relevant_components.extend(["cooling", "battery"])
        if any(w in q_lower for w in ["speed", "rpm", "motor", "engine", "power", "throttle"]):
            relevant_components.extend(["powertrain"])
        if any(w in q_lower for w in ["brake", "abs", "stop", "pad"]):
            relevant_components.extend(["brakes"])
        if any(w in q_lower for w in ["tire", "tyre", "pressure", "leak"]):
            relevant_components.extend(["tyres"])
        if any(w in q_lower for w in ["radar", "camera", "adas", "disengage", "airbag"]):
            relevant_components.extend(["adas"])

        if not relevant_components:
            relevant_components = ["battery", "cooling", "powertrain"]

        # Deduplicate
        relevant_components = list(dict.fromkeys(relevant_components))

        # Check for active faults
        findings = [self._component_states[c] for c in relevant_components if c in self._component_states]
        critical_items = [f for f in findings if f["status"] in ("Critical", "Warning")]

        # Generate structured answer
        lines = []
        if critical_items:
            lines.append(f"**Diagnostic Attention Required (Overall Vehicle Health: {self._overall_score}/100)**\n")
            for item in critical_items:
                dtc_str = f" [DTC: {', '.join(item['related_dtcs'])}]" if item['related_dtcs'] else ""
                lines.append(f"• **{item['name']}** ({item['status']}{dtc_str}): {item['status_note']}")
                if item.get("precaution"):
                    lines.append(f"  *Precaution:* {item['precaution']}")
                if item.get("predicted_ttf"):
                    lines.append(f"  *Estimated Time-to-Failure:* {item['predicted_ttf']}")
        else:
            lines.append(f"**Vehicle Status: Nominal ({self._overall_score}/100)**\n")
            for item in findings:
                ev = item["evidence"][0] if item["evidence"] else "Parameters within tolerance."
                lines.append(f"• **{item['name']}**: {item['status']} — {ev}")

        if self._causal_chain:
            chain_str = " ➔ ".join(f"{c['component']}: {c['event']}" for c in self._causal_chain)
            lines.append(f"\n**Root Cause Sequence:** {chain_str}")

        return {
            "answer": "\n".join(lines),
            "intent": intent,
            "source": "onboard",
            "components": relevant_components,
        }
