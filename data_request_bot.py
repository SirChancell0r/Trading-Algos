import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Define symbol, timeframe, and the total limit you want to fetch
symbol = 'AAVE'
timeframe = '15m'
total_limit = 5000  # Total records you want to fetch

# Maximum records per call allowed by the API
max_call_limit = 5000

# Calculate the number of iterations needed
iterations = -(-total_limit // max_call_limit)  # Ceiling division to ensure we cover all records

# Initialize an empty DataFrame to append fetched data
all_data = pd.DataFrame()

def get_ohlcv2(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        return snapshot_data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")
        return None


def process_data_to_df(snapshot_data):
    if snapshot_data:
        # Assuming the response contains a list of candles
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = snapshot['o']
            high_price = snapshot['h']
            low_price = snapshot['l']
            close_price = snapshot['c']
            volume = snapshot['v']
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)

        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data
    


# Loop to fetch and append data
for i in range(iterations):
    print(f'Fetching data for iteration {i + 1}/{iterations}')
    # Calculate the limit for this iteration
    iteration_limit = min(max_call_limit, total_limit - (i * max_call_limit))
    
    # Fetch the OHLCV data
    snapshot_data = get_ohlcv2(symbol, timeframe, iteration_limit)
    df = process_data_to_df(snapshot_data)
    
    # Append the fetched data to the all_data DataFrame
    all_data = pd.concat([all_data, df], ignore_index=True)

# Construct the file path using the symbol, timeframe, and total_limit
directory_path = '/root/AlgoCode/Hyper Liquid Bots/Data Pulls/Data'
file_name = f'{symbol}_{timeframe}_{total_limit}.csv'
file_path = os.path.join(directory_path, file_name)

# Save the concatenated DataFrame to CSV
all_data.to_csv(file_path, index=False)

print(all_data)

print(f'Data saved to {file_path}')
