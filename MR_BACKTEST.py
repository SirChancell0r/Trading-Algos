import os
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.test import SMA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class SMABuySellStrategy(Strategy):
    sma_period = 14 # Default SMA period, will be optimized
    buy_pct = 1.0 # Default buy percentage, will be optimized
    sell_pct = 1.0 # Default sell percentage, will be optimized

    def init(self):
        # Calculate the SMA using the Close price and the SMA period
        self.sma = self.I(SMA, self.data.Close, self.sma_period)

    def next(self):
        # Calculate the buying and selling thresholds
        buy_threshold = self.sma[-1] * (1 - self.buy_pct / 100)
        sell_threshold = self.sma[-1] * (1 + self.sell_pct / 100)

        # If the Close price is below the buying threshold, buy
        if len(self.data.Close) > 0 and self.data.Close[-1] <= buy_threshold:
            self.buy()

        # If the Close price is above the selling threshold, sell
        elif len(self.data.Close) > 0 and self.data.Close[-1] >= sell_threshold:
            self.position.close()

# Load the data
data_path = r'/root/AlgoCode/Hyper Liquid Bots/Data Pulls/Data/dgbusdt_15min_2000.csv'
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Extract coin name and timeframe from the file name
filename = os.path.basename(data_path)
filename_no_ext = os.path.splitext(filename)[0]
parts = filename_no_ext.split('_')
coinname = parts[0]           # 'BTC'
timeframe = parts[1]          # '5M'

# Correct the column names to match the DataFrame
data = data[['open', 'high', 'low', 'close', 'volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Sort the data index in ascending order
data = data.sort_index()

# Create and configure the backtest
bt = Backtest(data, SMABuySellStrategy, cash=100000, commission=.0002)

# Optimization with Heatmap
opt_stats, heatmap = bt.optimize(
    sma_period=range(10, 50, 2), # Testing different SMA periods
    buy_pct=range(1, 21, 1), # Testing different buy percentages
    sell_pct=range(1, 21, 1), # Testing different sell percentages
    maximize='Equity Final [$]', # Maximizing the final equity
    constraint=lambda param: param.sma_period > 0 and param.buy_pct > 0 and param.sell_pct > 0, # Ensuring all parameters are positive
    return_heatmap=True # Returning the heatmap
)

# Print the optimization results
print(opt_stats)

# Assuming opt_stats is the result of the optimization
opt_params = opt_stats._strategy  # Get the parameters of the best strategy

# Print the parameters
print("Optimal SMA Period:", opt_params.sma_period)
print("Optimal Buy Percentage:", opt_params.buy_pct)
print("Optimal Sell Percentage:", opt_params.sell_pct)

# Convert the heatmap to a DataFrame
heatmap_df = heatmap.unstack(level='buy_pct').T

# Ensure the directory exists
os.makedirs('/root/AlgoCode/Hyper Liquid Bots/Data Pulls/HeatMaps', exist_ok=True)

# Plot the heatmap for the optimized parameters
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='viridis')
plt.title("Optimization Heatmap")
plt.xlabel("Sell Percentage (%)")
plt.ylabel("Buy Percentage (%)")

# Save the heatmap image to the specified folder
output_folder = r"/root/AlgoCode/Hyper Liquid Bots/Data Pulls/HeatMaps"
filename = f"CCI_Backtest_{coinname}_{timeframe}.png"
heatmap_filepath = os.path.join(output_folder, filename)
plt.savefig(heatmap_filepath)
plt.show()


# Run the backtest with the best parameters
results = bt.run(sma_period=opt_stats.sma_period, buy_pct=opt_stats.buy_pct, sell_pct=opt_stats.sell_pct)
print(results)


# Plot the performance
bt.plot()



