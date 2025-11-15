import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def train_and_evaluate_models(filepath):
    """
    Loads processed wind data, splits it into train/test sets,
    trains XGBoost and LSTM models, evaluates them, and plots results.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file exists.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Features and Target
    # Exclude 'timestamp' and the target itself
    features = [col for col in df.columns if col not in ['timestamp', 'actual_power_output_kw', 'theoretical_power_curve_kwh', 'wind_direction_deg']]
    target = 'actual_power_output_kw'

    X = df[features]
    y = df[target]

    # Split data (using time-series split to avoid data leakage)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # --- XGBoost Model ---
    print("\n--- Training XGBoost Regressor ---")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate XGBoost
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    xgb_r2 = r2_score(y_test, xgb_predictions)
    print(f"XGBoost MAE: {xgb_mae:.2f}")
    print(f"XGBoost RMSE: {xgb_rmse:.2f}")
    print(f"XGBoost R2: {xgb_r2:.2f}")

    # Plot XGBoost Predictions
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, xgb_predictions, label='XGBoost Predicted', alpha=0.7)
    plt.title('XGBoost: Actual vs Predicted Wind Power Output')
    plt.xlabel('Time Index')
    plt.ylabel('Actual Power Output (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('xgboost_actual_vs_predicted.png')
    plt.show()

    # --- LSTM Model ---
    print("\n--- Training LSTM Model ---")

    # Normalize features for LSTM
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Reshape for LSTM [samples, time_steps, features]
    # Using 1 time step for simplicity for now, can be adjusted
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    y_lstm = y_scaled

    X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
    y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    # Reduced epochs and added verbose output for clearer progress during potentially long training
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1, shuffle=False)

    lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
    lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled)

    # Evaluate LSTM
    lstm_mae = mean_absolute_error(y_test, lstm_predictions)
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    lstm_r2 = r2_score(y_test, lstm_predictions)
    print(f"LSTM MAE: {lstm_mae:.2f}")
    print(f"LSTM RMSE: {lstm_rmse:.2f}")
    print(f"LSTM R2: {lstm_r2:.2f}")

    # Plot LSTM Predictions
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, lstm_predictions, label='LSTM Predicted', alpha=0.7)
    plt.title('LSTM: Actual vs Predicted Wind Power Output')
    plt.xlabel('Time Index')
    plt.ylabel('Actual Power Output (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_actual_vs_predicted.png')
    plt.show()

    # --- Comparison Table ---
    print("\n--- Model Comparison ---")
    metrics_data = {
        'Model': ['XGBoost', 'LSTM'],
        'MAE': [xgb_mae, lstm_mae],
        'RMSE': [xgb_rmse, lstm_rmse],
        'R2': [xgb_r2, lstm_r2]
    }
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.round(2))

if __name__ == "__main__":
    train_and_evaluate_models('processed_data.csv')

