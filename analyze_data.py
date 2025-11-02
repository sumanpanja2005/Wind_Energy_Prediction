import pandas as pd

def analyze_data(filepath):
    df = pd.read_csv(filepath)
    print("\n--- Data Info ---")
    df.info()
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n--- Unique Values for Key Columns (First 10) ---")
    for column in df.columns:
        if df[column].nunique() < 20 and df[column].dtype == 'object':  # Arbitrary limit for 'key' columns
            print(f"  {column}: {df[column].unique()[:10]}")

if __name__ == "__main__":
    analyze_data("nrel_alt_fuel_stations.csv")
