import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates new time-based and rolling/lag features for the wind energy dataset.
    Expects 'timestamp' and 'actual_power_output_kw' columns to be present.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0) # Saturday (5) and Sunday (6)

    # Rolling averages of 'actual_power_output_kw'
    df['rolling_avg_3'] = df['actual_power_output_kw'].rolling(window=3).mean()
    df['rolling_avg_6'] = df['actual_power_output_kw'].rolling(window=6).mean()
    df['rolling_avg_24'] = df['actual_power_output_kw'].rolling(window=24).mean()

    # Lag features of 'actual_power_output_kw'
    df['lag_1'] = df['actual_power_output_kw'].shift(1)
    df['lag_3'] = df['actual_power_output_kw'].shift(3)
    df['lag_24'] = df['actual_power_output_kw'].shift(24)

    # Fill NaN values created by rolling/lag features (e.g., with 0 or a suitable method)
    df = df.fillna(0) # Simple fill for demonstration; consider more advanced methods

    return df

if __name__ == "__main__":
    input_filepath = 'data.csv'
    output_filepath = 'processed_data.csv'

    try:
        df = pd.read_csv(input_filepath)
        processed_df = create_features(df)
        processed_df.to_csv(output_filepath, index=False)
        print(f"Features engineered successfully and saved to {output_filepath}")
        print("First 5 rows of processed data:")
        print(processed_df.head())
    except FileNotFoundError:
        print(f"Error: {input_filepath} not found. Please ensure the file exists.")
    except KeyError as e:
        print(f"Error: Missing expected column in input data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

