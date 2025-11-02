import requests
import pandas as pd

def get_nrel_data(api_key, limit=1):
    url = f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?limit={limit}&api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

if __name__ == "__main__":
    api_key = "1iJO9FMt0Fshh6eVu3a0HNFb4WEszzC49l7IW7Mz"  # Replace with your actual API key
    limit = 100  # You can adjust the limit to retrieve more data
    
    print(f"Fetching {limit} records from NREL API...")
    nrel_data = get_nrel_data(api_key, limit)
    
    if nrel_data and 'fuel_stations' in nrel_data:
        df = pd.DataFrame(nrel_data['fuel_stations'])
        df.to_csv("nrel_alt_fuel_stations.csv", index=False)
        print(f"Successfully retrieved {len(df)} records and saved to nrel_alt_fuel_stations.csv")
        print("First 5 rows of the dataframe:")
        print(df.head())
    else:
        print("Failed to retrieve data or no fuel stations found.")
        print(nrel_data)
