
from __future__ import annotations

from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from config import STADIUM_COORDS

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

def _nearest_hour_index(times, kickoff):
    ts = pd.to_datetime(times, utc=True, errors="coerce")
    if len(ts) == 0:
        return None
    kickoff = pd.to_datetime(kickoff, utc=True)
    delta = (ts - kickoff).to_series().abs()
    return int(delta.values.argmin())

def get_game_weather(home_team: str, kickoff_iso: str, roof=None, neutral=False, timeout=15):
    roof_s = str(roof or "").lower()
    if roof_s in {"dome", "closed"}:
        return {
            "weather_available": True,
            "weather_note": "Indoor/closed roof",
            "temperature_f": 70.0,
            "wind_mph": 0.0,
            "wind_gust_mph": 0.0,
            "precip_in": 0.0,
        }
    if neutral or home_team not in STADIUM_COORDS:
        return {
            "weather_available": False,
            "weather_note": "Weather skipped: neutral/unknown venue",
            "temperature_f": np.nan, "wind_mph": np.nan,
            "wind_gust_mph": np.nan, "precip_in": np.nan,
        }

    lat, lon = STADIUM_COORDS[home_team]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
        "forecast_days": 10,
    }
    try:
        r = requests.get(OPEN_METEO, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        h = payload.get("hourly", {})
        i = _nearest_hour_index(h.get("time", []), kickoff_iso)
        if i is None:
            raise ValueError("No hourly forecast")
        return {
            "weather_available": True,
            "weather_note": "Outdoor forecast",
            "temperature_f": float(h["temperature_2m"][i]),
            "wind_mph": float(h["wind_speed_10m"][i]),
            "wind_gust_mph": float(h["wind_gusts_10m"][i]),
            "precip_in": float(h["precipitation"][i]),
        }
    except Exception as e:
        return {
            "weather_available": False,
            "weather_note": f"Weather unavailable: {e}",
            "temperature_f": np.nan, "wind_mph": np.nan,
            "wind_gust_mph": np.nan, "precip_in": np.nan,
        }
