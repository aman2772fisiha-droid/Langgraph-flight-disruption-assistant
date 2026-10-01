from typing import Any, Dict, List
from langchain_core.tools import tool

# Static mock database
BOOKINGS_DB: Dict[str, Dict[str, Any]] = {
    "XK9L2P": {
        "booking_ref": "XK9L2P",
        "passenger_name": "Sarah Connor",
        "origin": "LHR",
        "destination": "DXB",
        "cabin_class": "economy",
        "fare_type": "economy_standard",
        "status": "cancelled",
        "original_departure": "2026-10-01T08:00:00",
        "delay_hours": 0.0,
    },
    "DL600X": {
        "booking_ref": "DL600X",
        "passenger_name": "James Howlett",
        "origin": "JFK",
        "destination": "FRA",
        "cabin_class": "economy",
        "fare_type": "economy_saver",
        "status": "delayed",
        "original_departure": "2026-10-01T10:00:00",
        "delay_hours": 6.5,
    },
    "UPGRADE1": {
        "booking_ref": "UPGRADE1",
        "passenger_name": "Bruce Wayne",
        "origin": "ORD",
        "destination": "NRT",
        "cabin_class": "economy",
        "fare_type": "economy_standard",
        "status": "cancelled",
        "original_departure": "2026-10-01T06:00:00",
        "delay_hours": 0.0,
    },
    "COMPLAIN9": {
        "booking_ref": "COMPLAIN9",
        "passenger_name": "Arthur Dent",
        "origin": "LGW",
        "destination": "MAD",
        "cabin_class": "economy",
        "fare_type": "economy_flex",
        "status": "cancelled",
        "original_departure": "2026-10-01T07:00:00",
        "delay_hours": 0.0,
    }
}

FARE_RULES_DB: Dict[str, Dict[str, Any]] = {
    "economy_standard": {"refundable": True, "fare_amount": 750.0, "taxes": 120.0},
    "economy_saver": {"refundable": False, "fare_amount": 450.0, "taxes": 85.0},
    "economy_flex": {"refundable": True, "fare_amount": 980.0, "taxes": 140.0},
}

@tool
def get_booking(booking_ref: str) -> Dict[str, Any]:
    """Retrieve passenger booking, cabin class, fare details, and disruption status."""
    return BOOKINGS_DB.get(booking_ref, {
        "error": f"Booking reference '{booking_ref}' not found."
    })

@tool
def search_flights(origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
    """Search available rebooking flights given origin, destination, and flight date."""
    # Deterministic branch for Policy Violation testing
    if origin == "ORD" and destination == "NRT":
        return [
            {
                "flight_number": "NH-011",
                "cabin_class": "business",  # Policy violation: upgrade not allowed
                "departure_time": "2026-10-01T14:00:00",
                "arrival_time": "2026-10-02T18:00:00",
                "connection_time_mins": 90,
                "hours_from_disruption": 8,
            }
        ]
    
    # Standard valid flights
    return [
        {
            "flight_number": "EK-008",
            "cabin_class": "economy",
            "departure_time": "2026-10-01T14:00:00",
            "arrival_time": "2026-10-02T00:30:00",
            "connection_time_mins": 75,
            "hours_from_disruption": 6,
        },
        {
            "flight_number": "BA-105",
            "cabin_class": "economy",
            "departure_time": "2026-10-01T19:00:00",
            "arrival_time": "2026-10-02T06:00:00",
            "connection_time_mins": 90,
            "hours_from_disruption": 11,
        }
    ]

@tool
def get_fare_rules(fare_type: str) -> Dict[str, Any]:
    """Retrieve refundability terms and refundable tax totals for a fare type."""
    return FARE_RULES_DB.get(fare_type, {"refundable": False, "fare_amount": 0.0, "taxes": 0.0})