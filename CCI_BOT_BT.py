# Import necessary libraries
from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # For file path operations

class MACD_CCI_Strategy(Strategy):
    """
    MACD and CCI based trading strategy with decimal TP and SL percentages.
    """
    tp_percent = 3.0  # Default TP percentage
    sl_percent = 1.0  # Default SL percentage

    def init(self):
        # Initialize the indicators
        macd, macd_signal, _ = talib.MACD(
            self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.macd = self.I(lambda: macd)
        self.macd_signal = self.I(lambda: macd_signal)
        self.cci = self.I(
            talib.CCI, self.data.High, self.data.Low, self.data.Close, timeperiod=10)
        
    def next(self):
        current_close = self.data.Close[-1]
        entry_price = current_close
        cci_delta = self.cci[-1] - self.cci[-2]

        # Check if no position is currently open
        if not self.position:
            # Condition to Open Short Position
            if (
                self.cci[-2] <= -100            # CCI is decreasing
            ):
                stop_loss = entry_price * (1 + self.sl_percent / 100)
                take_profit = entry_price * (1 - self.tp_percent / 100)
                self.sell(sl=stop_loss, tp=take_profit)

# Specify the data file path
data_filepath = r'/root/AlgoCode/Hyper Liquid Bots/Data Pulls/Data/POPCAT_1m_5000.csv'

# Extract coin name and timeframe from the file name
filename = os.path.basename(data_filepath)
filename_no_ext = os.path.splitext(filename)[0]
parts = filename_no_ext.split('_')
coinname = parts[0]           # 'BTC'
timeframe = parts[1]          # '5M'

# Load your historical data into a pandas DataFrame
data = pd.read_csv(data_filepath)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# Ensure the data contains the necessary columns
data = data[['open', 'high', 'low', 'close', 'volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Initialize the backtest
bt = Backtest(data, MACD_CCI_Strategy, commission=.0002, cash=100000, exclusive_orders=True)

# Convert numpy arrays to lists for parameter ranges
tp_range = list(np.arange(0.5, 10.1, 0.5))  # From 0.2% to 10% in steps of 0.3%
sl_range = list(np.arange(0.5, 10.1, 0.5))

# Optimize the strategy parameters with decimal steps
opt_stats, heatmap = bt.optimize(
    tp_percent=tp_range,
    sl_percent=sl_range,
    maximize='Equity Final [$]',  
    constraint=lambda param: param.tp_percent > 0 and param.sl_percent > 0,
    return_heatmap=True
)

print(opt_stats)

# Extract optimal parameters
opt_params = opt_stats._strategy

print("Optimal Take Profit:", opt_params.tp_percent)
print("Optimal Stop Loss:", opt_params.sl_percent)

# Plot the optimization heatmap
heatmap_df = heatmap.unstack(level='tp_percent').T
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=False, fmt='.2f', cmap='viridis')
plt.title("Optimization Heatmap")
plt.ylabel("Take Profit Percentage (%)")
plt.xlabel("Stop Loss Percentage (%)")

# Save the heatmap image to the specified folder
output_folder = r"/root/AlgoCode/Hyper Liquid Bots/Data Pulls/HeatMaps"
filename = f"CCI_Backtest_{coinname}_{timeframe}.png"
heatmap_filepath = os.path.join(output_folder, filename)
plt.savefig(heatmap_filepath)
plt.show()

# Run the backtest with optimal parameters
results = bt.run(tp_percent=opt_params.tp_percent, sl_percent=opt_params.sl_percent)

