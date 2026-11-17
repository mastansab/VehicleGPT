import asyncio
import os

os.environ["PEYKAN_NO_OLLAMA"] = "1"

from peykan.server.vehicle_assistant import _rule_answer, answer_query, trip_memory


def test_trip_memory_has_fourteen_days_and_adas():
    memory = trip_memory()
    assert len(memory["daily"]) == 14
    assert memory["adas_events"]
    assert memory["warnings"]


def test_challenge_queries_return_grounded_answers():
    snapshot = {"signals": {}, "frame_count": 0, "uptime_s": 0}
    cases = {
        "Why has my driving range reduced over the last two weeks?": "range",
        "When did fuel efficiency show the largest drop?": "l/100",
        "Show all ADAS disengagement events from yesterday.": "lane keep",
        "What happened before the battery overheating warning?": "overtemp",
        "Compare energy consumption between city and highway driving.": "highway",
    }
    for query, needle in cases.items():
        answer = _rule_answer(query, snapshot).lower()
        assert needle in answer, query


def test_answer_query_empty_and_live_health():
    empty = asyncio.run(answer_query("  ", {}))
    assert empty["source"] == "onboard"
    snapshot = {
        "signals": {
            "ENGINE_TEMP": {"value": 70, "unit": "degC"},
            "FUEL_LEVEL": {"value": 55, "unit": "%"},
            "BATTERY_VOLTAGE": {"value": 14.1, "unit": "V"},
            "WHEEL_SPEED_FL": {"value": 42, "unit": "kph"},
            "ENGINE_SPEED": {"value": 1800, "unit": "rpm"},
        },
        "frame_count": 12,
        "uptime_s": 3,
    }
    result = asyncio.run(answer_query("Run a health check on the live signals.", snapshot))
    assert "health check" in result["answer"].lower() or "no critical" in result["answer"].lower()
    assert result.get("focus", {}).get("nodes")


def test_cabin_and_unmatched_signal_search():
    snapshot = {
        "signals": {
            "DOOR_OPEN_FL": {"value": 0, "unit": ""},
            "DOOR_OPEN_FR": {"value": 1, "unit": ""},
            "DOOR_OPEN_RL": {"value": 0, "unit": ""},
            "DOOR_OPEN_RR": {"value": 0, "unit": ""},
            "VEHICLE_LOCKED": {"value": 1, "unit": ""},
            "HEADLIGHTS_ON": {"value": 1, "unit": ""},
            "SEATBELT_DRIVER": {"value": 1, "unit": ""},
            "SEATBELT_PASSENGER": {"value": 0, "unit": ""},
            "INTERIOR_TEMP": {"value": 21.5, "unit": "degC"},
            "AMBIENT_LIGHT": {"value": 40, "unit": "lux"},
            "WIPER_STATUS": {"value": 0, "unit": ""},
            "ENGINE_LOAD": {"value": 32, "unit": "%"},
        },
        "frame_count": 4,
        "uptime_s": 1,
    }
    doors = _rule_answer("Are the doors locked and the lights on?", snapshot).lower()
    assert "locked" in doors and "headlights" in doors
    load = _rule_answer("What is engine load right now?", snapshot).lower()
    assert "load" in load
