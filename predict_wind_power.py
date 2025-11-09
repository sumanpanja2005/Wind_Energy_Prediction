import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt # Import for plotting

def predict_wind_power(filepath):
    """
    Loads processed wind data, trains a linear regression model to predict
    actual power output, and evaluates the model.

    Args:
        filepath (str): The path to the processed wind data CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please make sure the file is in the correct directory.")
        return

    # Ensure timestamp is datetime type for potential time-series features later
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Define features (X) and target (y)
    features = ['wind_speed_ms', 'theoretical_power_curve_kwh']
    target = 'actual_power_output_kw'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Display some sample predictions
    print("\n--- Sample Predictions (Actual vs. Predicted) ---")
    sample_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)
    print(sample_df)

    # Plotting actual vs predicted values for a visual check
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Power Output (kW)')
    plt.ylabel('Predicted Power Output (kW)')
    plt.title('Actual vs. Predicted Wind Power Output')
    plt.grid(True)
    plt.savefig('actual_vs_predicted_wind_power.png')
    print("\nPlot of Actual vs. Predicted Wind Power Output saved as 'actual_vs_predicted_wind_power.png')")


if __name__ == "__main__":
    predict_wind_power("processed_wind_data.csv")
