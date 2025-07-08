import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly
import os

LAT = 40.7128
LON = -74.0060
ALTITUDE = 10
location = Point(LAT, LON, ALTITUDE)

OUTPUT_FILE = "nyc_weather_2024_hourly_meteostat.csv"

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31, 23, 59, 59)
print(f"Fetching historical weather using meteostat from {start_date} to {end_date}...")

try:
    data = Hourly(location, start_date, end_date)
    df_weather = data.fetch()

    if df_weather.empty:
        print("ERROR: Meteostat fetch returned no data. Check location or date range.")
        exit()

    print(f"Fetched {len(df_weather)} hourly records.")
    df_weather = df_weather.rename(
        columns={
            "temp": "temperature",
            "prcp": "precipitation",
            "snow": "snow_depth",
            "rhum": "humidity",
            "wspd": "wind_speed",
        }
    )
    cols_to_keep = [
        "temperature",
        "precipitation",
        "snow_depth",
        "humidity",
        "wind_speed",
    ]
    df_weather = df_weather[cols_to_keep]
    df_weather = df_weather.interpolate(method="time").ffill().bfill()
    df_weather.to_csv(OUTPUT_FILE)
    print(f"Historical weather data saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"ERROR during meteostat fetch or processing: {e}")
    import traceback

    traceback.print_exc()
