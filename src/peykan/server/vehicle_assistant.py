"""On-vehicle natural-language assistant.

Answers run fully on the edge: live CAN snapshot + retained trip memory,
optionally refined by a local Ollama SLM. If no model is installed the
same retrieved facts are turned into a readable answer by a rule engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("PEYKAN_OLLAMA_URL", "http://127.0.0.1:11434")
PREFERRED_MODELS = (
    "llama3.2:3b",
    "llama3.2",
    "qwen2.5:3b",
    "phi3:mini",
    "phi3",
    "gemma2:2b",
    "mistral",
)

SYSTEM_PROMPT = """You are VehicleGPT, an on-device time-machine assistant inside a Software Defined Vehicle.
You run on the edge with no cloud. Answer any vehicle question from live telemetry,
14-day trip memory, ADAS events and warnings. Be concise, use short bullet lists,
include numbers with units, and never invent signal values. If something is
not in the data, say so. Point at the subsystem (engine, ABS, battery, cabin, ADAS)."""

PERSONAS = {
    "driver": "Speak plainly for the driver. Lead with what it means and what to do.",
    "service": "Speak as a service engineer. Cite signal names, units and likely root cause.",
    "fleet": "Speak for fleet ops. Stress cost, utilisation, city vs highway and range.",
    "validation": "Speak for HIL/validation. Cite evidence windows, pass/fail and causal order.",
}


def _num(signals: dict, *names: str) -> Optional[float]:
    for name in names:
        info = signals.get(name)
        if not info:
            continue
        value = info.get("value") if isinstance(info, dict) else info
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _fmt(signals: dict, name: str, unit: str = "") -> str:
    info = signals.get(name)
    if not info:
        return "N/A"
    value = info.get("value")
    unit = unit or info.get("unit") or ""
    return f"{value} {unit}".strip()


@lru_cache(maxsize=1)
def trip_memory() -> Dict[str, Any]:
    """Deterministic 14-day onboard memory used for trend / history questions."""
    today = date.today()
    daily = []
    # Range falls from ~428 km to ~361 km; efficiency worsens mid-window.
    for i in range(14, 0, -1):
        day = today - timedelta(days=i)
        city_share = 0.35 + (0.04 * (14 - i) / 13) + (0.18 if i <= 7 else 0)
        hwy_share = max(0.15, 1 - city_share)
        efficiency = round(6.1 + (8.9 - 6.1) * city_share * 0.9, 2)
        if i == 7:
            efficiency = 9.4  # largest single-day drop
        range_km = round(428 - (14 - i) * 4.2 - (12 if i <= 7 else 0), 1)
        daily.append(
            {
                "date": day.isoformat(),
                "estimated_range_km": range_km,
                "fuel_l_per_100km": efficiency,
                "city_share": round(city_share, 2),
                "highway_share": round(hwy_share, 2),
                "ac_on_hours": round(0.4 + (0.9 if i <= 8 else 0), 1),
                "avg_speed_kph": round(72 - city_share * 38, 1),
                "tire_pressure_bar": round(2.4 - (0.05 if i <= 6 else 0), 2),
            }
        )

    yesterday = (today - timedelta(days=1)).isoformat()
    overheat_day = (today - timedelta(days=3)).isoformat()
    adas = [
        {
            "time": f"{yesterday} 08:14",
            "system": "Lane Keep Assist",
            "event": "disengagement",
            "reason": "Heavy rain and faded lane markings on arterial road",
            "speed_kph": 48,
        },
        {
            "time": f"{yesterday} 08:16",
            "system": "Lane Keep Assist",
            "event": "disengagement",
            "reason": "Repeated after driver override",
            "speed_kph": 46,
        },
        {
            "time": f"{yesterday} 17:42",
            "system": "Adaptive Cruise Control",
            "event": "disengagement",
            "reason": "Cut-in vehicle inside minimum following distance",
            "speed_kph": 91,
        },
        {
            "time": f"{yesterday} 21:05",
            "system": "Automatic Emergency Braking",
            "event": "warning",
            "reason": "Construction barrier / false-positive at night",
            "speed_kph": 32,
        },
        {
            "time": f"{(today - timedelta(days=5)).isoformat()} 12:03",
            "system": "Lane Keep Assist",
            "event": "disengagement",
            "reason": "Work zone, lanes not detected",
            "speed_kph": 40,
        },
    ]
    warnings = [
        {
            "time": f"{overheat_day} 14:22",
            "code": "BATT_OVERTEMP",
            "title": "Traction battery overheating warning",
            "peak_c": 62,
            "preceding": [
                {"t": "-48 min", "fact": "DC fast charge started (150 kW), SOC 18% → 71%"},
                {"t": "-22 min", "fact": "Cabin HVAC on MAX A/C, ambient 38 °C"},
                {"t": "-11 min", "fact": "Charge tapered as pack hit 54 °C"},
                {"t": "-4 min", "fact": "Stop-and-go traffic after unplug, coolant pump at 92%"},
                {"t": "0", "fact": "BMS raised overtemperature warning at 62 °C"},
            ],
        }
    ]
    energy = {
        "city_l_per_100km": 9.2,
        "highway_l_per_100km": 5.8,
        "city_kwh_per_100km": 18.4,
        "highway_kwh_per_100km": 14.1,
        "city_avg_speed_kph": 28,
        "highway_avg_speed_kph": 96,
        "city_idle_pct": 21,
        "highway_idle_pct": 3,
    }
    return {
        "daily": daily,
        "adas_events": adas,
        "warnings": warnings,
        "energy": energy,
        "notes": [
            "Range drop over 14 days coincides with more city driving, A/C hours, and a 0.2 bar tire-pressure sag.",
            "Largest fuel-efficiency drop is 7 days ago (6.4 → 9.4 L/100 km) on a cold, congested commute.",
        ],
    }


def _live_brief(snapshot: dict) -> str:
    signals = snapshot.get("signals") or {}
    lines = [
        f"frames={snapshot.get('frame_count', 0)} uptime_s={snapshot.get('uptime_s', 0)}",
    ]
    for name in (
        "ENGINE_SPEED",
        "ENGINE_TEMP",
        "THROTTLE_POSITION",
        "ENGINE_LOAD",
        "FUEL_LEVEL",
        "BATTERY_VOLTAGE",
        "WHEEL_SPEED_FL",
        "WHEEL_SPEED_FR",
        "WHEEL_SPEED_RL",
        "WHEEL_SPEED_RR",
        "WHEEL_BASED_VEHICLE_SPEED",
        "CRASH_DETECTED",
        "SYSTEM_STATUS",
        "VEHICLE_LOCKED",
    ):
        if name in signals:
            info = signals[name]
            lines.append(f"{name}={info.get('value')} {info.get('unit') or ''}".strip())
    events = snapshot.get("events") or []
    if events:
        lines.append("recent_events:")
        for ev in events[-8:]:
            lines.append(f"  - {ev.get('title')}: {ev.get('detail')}")
    return "\n".join(lines)


def _ollama_reachable() -> bool:
    try:
        host = "127.0.0.1"
        port = 11434
        if "://" in OLLAMA_URL:
            rest = OLLAMA_URL.split("://", 1)[1]
            hostport = rest.split("/", 1)[0]
            if ":" in hostport:
                host, port_s = hostport.rsplit(":", 1)
                port = int(port_s)
            else:
                host = hostport
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _ollama_model() -> Optional[str]:
    if os.environ.get("PEYKAN_NO_OLLAMA"):
        return None
    if not _ollama_reachable():
        return None
    forced = os.environ.get("PEYKAN_OLLAMA_MODEL")
    if forced:
        return forced
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in payload.get("models") or []]
        for preferred in PREFERRED_MODELS:
            for name in names:
                if name == preferred or name.startswith(preferred):
                    return name
        return names[0] if names else None
    except Exception:
        return None


def _ollama_answer(query: str, snapshot: dict) -> Optional[str]:
    model = _ollama_model()
    if not model:
        return None
    memory = trip_memory()
    user = (
        f"LIVE TELEMETRY\n{_live_brief(snapshot)}\n\n"
        f"TRIP MEMORY (14 days)\n{json.dumps(memory, indent=2)}\n\n"
        f"USER QUESTION\n{query}"
    )
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("message") or {}).get("content") or ""
        return text.strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.info("Ollama unavailable (%s); using onboard analyzer", exc)
        return None


def _health_issues(signals: dict) -> List[str]:
    issues: List[str] = []
    temp = _num(signals, "ENGINE_TEMP")
    if temp is not None and temp > 85:
        issues.append(f"Engine coolant high at {temp} °C")
    fuel = _num(signals, "FUEL_LEVEL")
    if fuel is not None and fuel < 15:
        issues.append(f"Fuel low at {fuel}%")
    volt = _num(signals, "BATTERY_VOLTAGE")
    if volt is not None and volt < 12:
        issues.append(f"Battery voltage low at {volt} V")
    crash = _num(signals, "CRASH_DETECTED")
    if crash and crash >= 1:
        issues.append("Crash detected flag is set")
    speeds = [
        _num(signals, "WHEEL_SPEED_FL"),
        _num(signals, "WHEEL_SPEED_FR"),
        _num(signals, "WHEEL_SPEED_RL"),
        _num(signals, "WHEEL_SPEED_RR"),
    ]
    present = [s for s in speeds if s is not None]
    if len(present) >= 3 and max(present) - min(present) > 8:
        issues.append("Wheel-speed mismatch — possible ABS sensor issue")
    return issues


def _rule_answer(query: str, snapshot: dict) -> str:
    q = query.lower()
    signals = snapshot.get("signals") or {}
    memory = trip_memory()
    daily: List[dict] = memory["daily"]
    energy = memory["energy"]
    live_tail = []
    speed = _num(signals, "WHEEL_BASED_VEHICLE_SPEED", "WHEEL_SPEED_FL")
    if speed is not None:
        live_tail.append(f"Live speed right now: **{speed} kph**")
    temp = _num(signals, "ENGINE_TEMP")
    if temp is not None:
        live_tail.append(f"Live coolant: **{temp} °C**")

    def with_live(text: str) -> str:
        if live_tail:
            return text + "\n\n" + "\n".join(f"- {line}" for line in live_tail)
        return text

    if any(w in q for w in ["range", "two week", "2 week", "last two", "driving range"]):
        first, last = daily[0], daily[-1]
        drop = round(first["estimated_range_km"] - last["estimated_range_km"], 1)
        return with_live(
            "**Driving range over the last 14 days**\n"
            f"- {first['date']}: **{first['estimated_range_km']} km** remaining\n"
            f"- {last['date']}: **{last['estimated_range_km']} km** remaining\n"
            f"- Net drop: **{drop} km** ({round(100 * drop / first['estimated_range_km'], 1)}%)\n\n"
            "**Why it fell**\n"
            "- City-share of driving rose (more stop-go, lower average speed)\n"
            f"- A/C on-time increased to **{last['ac_on_hours']} h/day** in the last week\n"
            f"- Tire pressure sagged to **{last['tire_pressure_bar']} bar** (adds rolling resistance)\n"
            "- Cold-start urban commutes last week used more fuel per kilometre"
        )

    if any(w in q for w in ["efficiency", "fuel economy", "mpg", "l/100", "largest drop"]):
        worst = max(daily, key=lambda d: d["fuel_l_per_100km"])
        prev_idx = max(0, daily.index(worst) - 1)
        prev = daily[prev_idx]
        return with_live(
            "**Fuel efficiency**\n"
            f"- Largest drop: **{prev['date']} → {worst['date']}**\n"
            f"- {prev['fuel_l_per_100km']} → **{worst['fuel_l_per_100km']} L/100 km**\n"
            f"- That day was **{int(worst['city_share'] * 100)}% city**, avg speed "
            f"**{worst['avg_speed_kph']} kph**, with a cold congested commute\n"
            f"- 14-day span: {daily[0]['fuel_l_per_100km']} → {daily[-1]['fuel_l_per_100km']} L/100 km"
        )

    if "adas" in q or "disengage" in q or "lane keep" in q or "yesterday" in q:
        events = memory["adas_events"]
        if "yesterday" in q:
            yday = (date.today() - timedelta(days=1)).isoformat()
            events = [e for e in events if e["time"].startswith(yday)]
        lines = [f"**ADAS events ({len(events)})**"]
        for e in events:
            lines.append(
                f"- {e['time']} — **{e['system']} {e['event']}** at {e['speed_kph']} kph. {e['reason']}"
            )
        if not events:
            lines.append("- No matching ADAS events in onboard memory.")
        return with_live("\n".join(lines))

    if any(w in q for w in ["overheat", "overheating", "battery warning", "what happened before"]):
        warning = memory["warnings"][0]
        lines = [
            f"**{warning['title']}** at {warning['time']} (peak **{warning['peak_c']} °C**)",
            "What happened immediately before:",
        ]
        for step in warning["preceding"]:
            lines.append(f"- {step['t']}: {step['fact']}")
        lines.append(
            "- After the warning the BMS limited charge/discharge until pack cooled below 45 °C."
        )
        return with_live("\n".join(lines))

    if any(w in q for w in ["city", "highway", "energy consumption", "compare"]):
        return with_live(
            "**Energy consumption: city vs highway**\n"
            f"- City: **{energy['city_l_per_100km']} L/100 km** "
            f"({energy['city_kwh_per_100km']} kWh/100 km), avg {energy['city_avg_speed_kph']} kph, "
            f"{energy['city_idle_pct']}% idle\n"
            f"- Highway: **{energy['highway_l_per_100km']} L/100 km** "
            f"({energy['highway_kwh_per_100km']} kWh/100 km), avg {energy['highway_avg_speed_kph']} kph, "
            f"{energy['highway_idle_pct']}% idle\n"
            "- Highway is ~37% more efficient on fuel and ~23% better on electrical energy, "
            "mainly from less idle and steadier load."
        )

    if any(w in q for w in ["door", "lock", "cabin", "seatbelt", "light", "wiper", "interior", "ambient"]):
        doors = []
        for corner in ("FL", "FR", "RL", "RR"):
            open_v = _num(signals, f"DOOR_OPEN_{corner}")
            doors.append(f"{corner} {'OPEN' if open_v == 1 else 'closed'}")
        locked = _num(signals, "VEHICLE_LOCKED")
        belt_d = _num(signals, "SEATBELT_DRIVER")
        belt_p = _num(signals, "SEATBELT_PASSENGER")
        lamps = _num(signals, "HEADLIGHTS_ON")
        return (
            "**Cabin & body**\n"
            f"- Doors: {', '.join(doors)}\n"
            f"- Locks: **{'locked' if locked == 1 else 'unlocked'}**\n"
            f"- Seatbelts: driver {'fastened' if belt_d == 1 else 'unfastened'}, "
            f"passenger {'fastened' if belt_p == 1 else 'unfastened'}\n"
            f"- Headlights: **{'ON' if lamps == 1 else 'off'}** · ambient {_fmt(signals, 'AMBIENT_LIGHT', 'lux')}\n"
            f"- Interior: **{_fmt(signals, 'INTERIOR_TEMP', '°C')}** · wipers {_fmt(signals, 'WIPER_STATUS')}"
        )

    if any(w in q for w in ["tire", "tyre", "pressure"]):
        last = daily[-1]
        return with_live(
            "**Tires**\n"
            f"- Last logged pressure: **{last['tire_pressure_bar']} bar** "
            "(sag of ~0.2 bar over two weeks increases rolling resistance and cuts range)."
        )

    if any(w in q for w in ["vin", "identify", "what car", "which vehicle"]):
        return (
            "**Vehicle identity**\n"
            "- Edge node: **VehicleGPT** on-device assistant\n"
            "- Bus: 11-bit DBC + J1939 heavy-duty PGNs on the same virtual backbone\n"
            "- VIN is not a periodic CAN signal; ask a service session to read it over UDS."
        )

    if any(w in q for w in ["j1939", "pgn", "heavy", "torque", "dm1"]):
        lines = ["**J1939 live**"]
        for name, info in sorted(signals.items()):
            if str(info.get("message", "")).startswith("J1939"):
                lines.append(f"- {name}: **{info.get('value')} {info.get('unit') or ''}**".strip())
        if len(lines) == 1:
            lines.append("- No J1939 frames decoded yet.")
        return "\n".join(lines[:16])

    if any(w in q for w in ["speed", "fast", "rpm", "engine", "throttle", "load", "powertrain"]):
        return (
            "**Live powertrain**\n"
            f"- Speed: **{_fmt(signals, 'WHEEL_SPEED_FL', 'kph')}**\n"
            f"- Engine: **{_fmt(signals, 'ENGINE_SPEED', 'rpm')}**\n"
            f"- Throttle: **{_fmt(signals, 'THROTTLE_POSITION', '%')}**\n"
            f"- Load: **{_fmt(signals, 'ENGINE_LOAD', '%')}**\n"
            f"- Coolant: **{_fmt(signals, 'ENGINE_TEMP', '°C')}**"
        )

    if any(w in q for w in ["temp", "coolant", "hot", "thermal"]):
        t = _num(signals, "ENGINE_TEMP")
        status = "unknown"
        if t is not None:
            status = "critical" if t > 100 else "elevated" if t > 85 else "normal"
        return f"**Thermal**\n- Coolant: **{_fmt(signals, 'ENGINE_TEMP', '°C')}** ({status})"

    if any(w in q for w in ["battery", "voltage", "charge"]):
        v = _num(signals, "BATTERY_VOLTAGE")
        status = "unknown"
        if v is not None:
            status = "low" if v < 12 else "marginal" if v < 13 else "healthy"
        return f"**Electrical**\n- 12 V battery: **{_fmt(signals, 'BATTERY_VOLTAGE', 'V')}** ({status})"

    if any(w in q for w in ["fuel", "tank", "petrol", "gas"]):
        return f"**Fuel**\n- Level: **{_fmt(signals, 'FUEL_LEVEL', '%')}**"

    if any(w in q for w in ["abs", "wheel", "brake"]):
        return (
            "**ABS / wheel speeds**\n"
            f"- FL {_fmt(signals, 'WHEEL_SPEED_FL', 'kph')}\n"
            f"- FR {_fmt(signals, 'WHEEL_SPEED_FR', 'kph')}\n"
            f"- RL {_fmt(signals, 'WHEEL_SPEED_RL', 'kph')}\n"
            f"- RR {_fmt(signals, 'WHEEL_SPEED_RR', 'kph')}"
        )

    if any(w in q for w in ["crash", "airbag", "collision"]):
        crash = _num(signals, "CRASH_DETECTED")
        flag = "YES" if crash and crash >= 1 else "No"
        return (
            "**Restraint systems**\n"
            f"- Crash detected: **{flag}**\n"
            f"- Airbag ECU: **{_fmt(signals, 'SYSTEM_STATUS')}**"
        )

    if any(w in q for w in ["health", "fault", "wrong", "issue", "problem", "diagnostic", "check"]):
        issues = _health_issues(signals)
        if issues:
            return "**Attention needed**\n" + "\n".join(f"- {i}" for i in issues)
        if signals:
            return (
                "**Health check: no critical live faults.**\n"
                f"- Speed {_fmt(signals, 'WHEEL_SPEED_FL', 'kph')}, "
                f"RPM {_fmt(signals, 'ENGINE_SPEED', 'rpm')}, "
                f"coolant {_fmt(signals, 'ENGINE_TEMP', '°C')}, "
                f"fuel {_fmt(signals, 'FUEL_LEVEL', '%')}"
            )
        return "No live telemetry yet. Start the vehicle simulator so signals begin streaming."

    if any(w in q for w in ["overview", "summary", "snapshot", "right now", "status", "what's happening"]):
        n = len(signals)
        lines = [
            f"**Live snapshot** ({n} signals, {snapshot.get('frame_count', 0)} frames)",
            f"- Speed: **{_fmt(signals, 'WHEEL_SPEED_FL', 'kph')}**",
            f"- Engine: **{_fmt(signals, 'ENGINE_SPEED', 'rpm')}** @ {_fmt(signals, 'ENGINE_TEMP', '°C')}",
            f"- Fuel: **{_fmt(signals, 'FUEL_LEVEL', '%')}** · Battery **{_fmt(signals, 'BATTERY_VOLTAGE', 'V')}**",
        ]
        return "\n".join(lines)

    hits = _search_signals(q, signals)
    if hits:
        lines = ["**Matching live signals**"]
        lines.extend(hits[:10])
        return "\n".join(lines)

    n = len(signals)
    return (
        f"I have **{n} live signals** and 14 days of flight-recorder memory.\n\n"
        "I can talk about speed, RPM, coolant, fuel, battery, ABS wheels, doors, "
        "locks, lights, seatbelts, ADAS, range, efficiency, city vs highway, "
        "and what happened before a warning.\n"
        "Try: *What's my coolant temperature?* or *Are the doors locked?*"
    )


def _search_signals(q: str, signals: dict) -> List[str]:
    tokens = [t for t in re.split(r"\W+", q) if len(t) > 2 and t not in {"the", "and", "for", "what", "show"}]
    if not tokens:
        return []
    lines = []
    for name, info in sorted(signals.items()):
        blob = f"{name} {info.get('message') or ''}".lower().replace("_", " ")
        if any(t in blob for t in tokens):
            lines.append(f"- {name}: **{info.get('value')} {info.get('unit') or ''}**".strip())
    return lines


def _evidence(query: str, answer: str, snapshot: dict) -> Dict[str, Any]:
    blob = f"{query} {answer}".lower()
    nodes: List[str] = []
    if any(w in blob for w in ["engine", "rpm", "coolant", "throttle", "powertrain"]):
        nodes.append("engine")
    if any(w in blob for w in ["abs", "wheel", "brake"]):
        nodes.append("abs")
    if any(w in blob for w in ["battery", "volt", "overtemp", "charge"]):
        nodes.append("battery")
    if any(w in blob for w in ["airbag", "crash", "seatbelt"]):
        nodes.append("airbag")
    if any(w in blob for w in ["door", "lock", "cabin", "light", "wiper"]):
        nodes.append("body")
    if any(w in blob for w in ["adas", "lane", "disengage"]):
        nodes.append("adas")
    if any(w in blob for w in ["fuel", "range", "efficiency"]):
        nodes.append("fuel")
    if not nodes:
        nodes = ["engine"]

    memory = trip_memory()
    daily = memory["daily"]
    markers: List[str] = []
    chain: List[Dict[str, str]] = []
    if any(w in blob for w in ["range", "two week"]):
        markers = [daily[0]["date"], daily[-1]["date"]]
        chain = [
            {"t": daily[0]["date"], "label": "Range baseline"},
            {"t": daily[7]["date"] if len(daily) > 7 else daily[-1]["date"], "label": "City share rises"},
            {"t": daily[-1]["date"], "label": "Range now"},
        ]
    elif any(w in blob for w in ["efficiency", "largest drop"]):
        worst = max(daily, key=lambda d: d["fuel_l_per_100km"])
        markers = [worst["date"]]
        chain = [
            {"t": worst["date"], "label": "Efficiency cliff"},
            {"t": "commute", "label": "Cold + congestion"},
        ]
    elif "adas" in blob or "yesterday" in blob:
        yday = (date.today() - timedelta(days=1)).isoformat()
        markers = [yday]
        chain = [{"t": e["time"][-5:], "label": f"{e['system']} {e['event']}"} for e in memory["adas_events"] if e["time"].startswith(yday)]
    elif "overheat" in blob or "before" in blob:
        w = memory["warnings"][0]
        markers = [w["time"][:10]]
        chain = [{"t": s["t"], "label": s["fact"][:42]} for s in w["preceding"]]
    elif any(w in blob for w in ["city", "highway"]):
        markers = [daily[-1]["date"]]
        chain = [
            {"t": "city", "label": "Stop-go / idle"},
            {"t": "highway", "label": "Steady load"},
        ]
    return {"nodes": nodes, "markers": markers, "chain": chain}


async def answer_query(
    query: str,
    snapshot: dict,
    persona: str = "driver",
    data_access: Any = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    query = (query or "").strip()
    persona = persona if persona in PERSONAS else "driver"
    if not query:
        return {
            "answer": "Please enter a question about the vehicle.",
            "source": "onboard",
            "persona": persona,
            "focus": {"nodes": [], "markers": [], "chain": []},
            "latency_ms": 0,
        }

    if data_access is not None:
        historical = data_access.query_historical_question(query)
        if historical:
            focus = _evidence(query, historical["answer"], snapshot)
            return {
                "answer": historical["answer"],
                "source": "onboard",
                "persona": persona,
                "focus": focus,
                "tier": historical.get("tier"),
                "attribution": historical.get("attribution"),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

    import asyncio

    ollama_text = await asyncio.to_thread(_ollama_answer, query, snapshot)
    answer = ollama_text or _rule_answer(query, snapshot)
    source = "ollama" if ollama_text else "onboard"
    if persona != "driver":
        answer = f"**Persona: {persona.replace('_', ' ')}**\n" + answer
    focus = _evidence(query, answer, snapshot)
    return {
        "answer": answer,
        "source": source,
        "persona": persona,
        "focus": focus,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
