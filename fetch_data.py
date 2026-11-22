"""
Download real daily weather data for Sri Lankan cities (2015-2024)
from the Open-Meteo historical archive (ERA5 reanalysis, free, no API key).

Output: data/sl_weather_daily.csv
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

CITIES = {
    # city: (lat, lon, climate zone)
    "Colombo": (6.93, 79.85, "Wet"),
    "Kandy": (7.29, 80.63, "Wet"),
    "Galle": (6.05, 80.22, "Wet"),
    "Ratnapura": (6.68, 80.40, "Wet"),
    "Nuwara Eliya": (6.97, 80.78, "Wet"),
    "Jaffna": (9.66, 80.02, "Dry"),
    "Anuradhapura": (8.31, 80.40, "Dry"),
    "Trincomalee": (8.59, 81.21, "Dry"),
    "Batticaloa": (7.73, 81.69, "Dry"),
    "Hambantota": (6.12, 81.12, "Dry"),
    "Kurunegala": (7.49, 80.36, "Intermediate"),
    "Badulla": (6.99, 81.06, "Intermediate"),
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
]

START, END = "2015-01-01", "2024-12-31"


def fetch_city(name, lat, lon):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={START}&end_date={END}"
        f"&daily={','.join(DAILY_VARS)}&timezone=Asia%2FColombo"
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = 30 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s ...")
            time.sleep(wait)
    else:
        raise RuntimeError("Open-Meteo kept rate-limiting; try again later.")
    df = pd.DataFrame(payload["daily"])
    df["city"] = name
    return df


def main():
    out = Path(__file__).parent / "data"
    cache = out / "cities"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (lat, lon, zone) in CITIES.items():
        path = cache / f"{name.replace(' ', '_')}.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            print(f"Fetching {name} ...")
            df = fetch_city(name, lat, lon)
            df["climate_zone"] = zone
            df.to_csv(path, index=False)
            time.sleep(5)  # be polite to the free API
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    data.to_csv(out / "sl_weather_daily.csv", index=False)
    print(f"Saved {len(data):,} rows to data/sl_weather_daily.csv")


if __name__ == "__main__":
    main()
