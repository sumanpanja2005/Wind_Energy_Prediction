import pandas as pd
import re
import sys

# Map canonical names to possible source column variants
COLUMN_ALIASES = {
    "timestamp": [
        "Date/Time", "date/time", "date time", "datetime", "timestamp"
    ],
    "actual_power_output_kw": [
        "LV ActivePower (kW)", "lv activepower (kw)", "active power (kw)",
        "actual_power_output_kw", "activepower_kw"
    ],
    "wind_speed_ms": [
        "Wind Speed (m/s)", "wind speed (m/s)", "wind speed", "windspeed",
        "wind_speed", "wind_speed_ms"
    ],
    "theoretical_power_curve_kwh": [
        "Theoretical_Power_Curve (KWh)", "theoretical_power_curve (kwh)",
        "theoretical power curve kwh", "theoretical_power_curve_kwh"
    ],
    "wind_direction_deg": [
        "Wind Direction (?)", "wind direction (deg)", "wind direction (°)",
        "wind direction", "wind_direction", "wind_direction_deg"
    ],
}

def _norm(name: str) -> str:
    # lower, remove non-alphanumerics
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {_norm(c): c for c in df.columns}
    rename_map = {}
    for target, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            key = _norm(cand)
            if key in normalized:
                rename_map[normalized[key]] = target
                break
    return df.rename(columns=rename_map)

def preprocess_wind_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df = _rename_to_canonical(df)

    required = [
        "timestamp",
        "wind_speed_ms",
        "wind_direction_deg",
        "actual_power_output_kw",
        "theoretical_power_curve_kwh",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Parse timestamp if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts = pd.to_datetime(df["timestamp"], format="%d %m %Y %H:%M", errors="coerce")
        # Fallback to inference if too many NaT
        if ts.isna().mean() > 0.5:
            ts = pd.to_datetime(df["timestamp"], dayfirst=True, infer_datetime_format=True, errors="coerce")
        df["timestamp"] = ts

    # Keep only required columns in order
    df = df[required]
    return df

if __name__ == "__main__":
    in_path = r"C:\Users\suman\OneDrive\download\OneDrive\Desktop\Wind Energy Prediction\processed_wind_data.csv"
    out_path = "reprocessed_wind_data.csv"
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    try:
        processed_df = preprocess_wind_data(in_path)
        processed_df.to_csv(out_path, index=False)
        print(f"Reprocessed wind data saved to {out_path}")
        print("First 5 rows of the reprocessed dataframe:")
        print(processed_df.head())
    except Exception as e:
        print(f"Error: {e}")
