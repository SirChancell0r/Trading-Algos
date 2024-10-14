# Standard library imports
import json
import time
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
import requests
import schedule
import eth_account
from eth_account.signers.local import LocalAccount

# Local imports (assuming you have these modules)
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# Replace this with your actual private key
from dontshare import private_key_CCI_1h

order_usd_size = 25
timeframe = '1h'

# Define symbol-specific parameters
symbols = ['SUI', 'WLD', 'CFX', 'DYDX', 'MANTA', 'BIGTIME', 'kLUNC', 'PIXEL', 'POPCAT', 'MEW', 'kDOGS', 'BOME']  # Add more symbols as needed
symbols_data = {
    'SUI': {'cci_period': 8, 'tp_pct': 5.5, 'sl_pct': 1, 'leverage': 20},
    'WLD': {'cci_period': 5, 'tp_pct': 9, 'sl_pct': 3, 'leverage': 20},
    'CFX': {'cci_period': 10, 'tp_pct': 5, 'sl_pct': 8.5, 'leverage': 20},
    'DYDX': {'cci_period': 9, 'tp_pct': 5.5, 'sl_pct': 2.0, 'leverage': 20},
    'MANTA': {'cci_period': 9, 'tp_pct': 7, 'sl_pct': 2.5, 'leverage': 20},
    'BIGTIME': {'cci_period': 5, 'tp_pct': 8, 'sl_pct': 2, 'leverage': 20},
    'kLUNC': {'cci_period': 8, 'tp_pct': 3, 'sl_pct': 8, 'leverage': 20},
    'PIXEL': {'cci_period': 5, 'tp_pct': 9, 'sl_pct': 0.5, 'leverage': 20},
    'POPCAT': {'cci_period': 6, 'tp_pct': 9, 'sl_pct': 0.5, 'leverage': 20},
    'MEW': {'cci_period': 6, 'tp_pct': 8.5, 'sl_pct': 1.5, 'leverage': 20},
    'kDOGS': {'cci_period': 12, 'tp_pct': 8, 'sl_pct': 3, 'leverage': 20},
    'BOME': {'cci_period': 12, 'tp_pct': 10, 'sl_pct': 0.5, 'leverage': 20},
    # Add other symbols with their respective parameters
}

# Initialize a dictionary to track the last signal candle timestamp for each symbol
last_signal_time = {}

def calculate_cci(data, n):
    TP = (data['high'] + data['low'] + data['close']) / 3
    MA = TP.rolling(n).mean()
    MD = TP.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    CCI = (TP - MA) / (0.015 * MD)
    return CCI

def cci_strategy(symbol, data, cci_period, tp_pct, sl_pct):
    # Exclude the last row (current incomplete candle)
    data = data.iloc[:-1].copy()

    # Ensure there is enough data to calculate CCI
    if len(data) < cci_period + 1:
        print(f"Not enough data to calculate CCI for period {cci_period} for {symbol}")
        return "HOLD", None, None, None, None, None

    # Calculate CCI on data without the current candle
    data['cci'] = calculate_cci(data, cci_period)

    # Get the last two CCI values
    cci_previous = data['cci'].iloc[-2]
    cci_current = data['cci'].iloc[-1]

    # Get the timestamp of the signal candle (current candle in this context)
    signal_time = data.index[-1]

    # Strategy: Enter short when CCI decreases after being above +100
    if cci_previous > 100 and cci_current < cci_previous:
        action = "SELL"  # Enter a short position
    else:
        action = "HOLD"

    # Get the current price (from the last closed candle)
    current_price = data['close'].iloc[-1]

    # Calculate take profit and stop loss prices
    tp_price = current_price * (1 - tp_pct / 100)
    sl_price = current_price * (1 + sl_pct / 100)

    return action, current_price, tp_price, sl_price, cci_current, signal_time

def get_ohlcv2(symbol, interval, lookback_days):
    end_time = datetime.utcnow()
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
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.utcfromtimestamp(snapshot['t'] / 1000)
            open_price = float(snapshot['o'])
            high_price = float(snapshot['h'])
            low_price = float(snapshot['l'])
            close_price = float(snapshot['c'])
            volume = float(snapshot['v'])
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)
        df.set_index('timestamp', inplace=True)
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data

def ask_bid(symbol):
    """Gets the ask and bid for the specified symbol."""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {'type': 'l2Book', 'coin': symbol}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']

    # Get ask and bid
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

def get_sz_px_decimals(symbol):
    """Returns size decimals and price decimals for the symbol."""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print('Symbol not found')
            sz_decimals = 0
    else:
        print('Error:', response.status_code)
        sz_decimals = 0

    ask = ask_bid(symbol)[0]

    # Compute the number of decimal points in the ask price
    ask_str = str(ask)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    print(f'{symbol} price: {ask}, sz decimals: {sz_decimals}, px decimals: {px_decimals}')

    return sz_decimals, px_decimals

def adjust_leverage_usd_size(symbol, usd_size, leverage, account, exchange, info):
    """Calculates size based on a specific USD dollar amount and updates leverage per symbol."""
    print(f'Adjusting leverage for {symbol}: {leverage}x')

    # Get account value
    user_state = info.user_state(account.address)
    acct_value = float(user_state["marginSummary"]["accountValue"])
    print(f"Account Value: {acct_value}")

    # Update leverage for the symbol
    exchange.update_leverage(leverage, symbol)

    # Get current price
    price = ask_bid(symbol)[0]

    # Calculate size
    size = (usd_size / price) * leverage
    rounding = get_sz_px_decimals(symbol)[0]
    size = round(size, rounding)
    print(f'Size of {symbol} to trade: {size}')

    return size

def get_position(symbol, account, info):
    """Gets the current position info for the symbol."""
    user_state = info.user_state(account.address)

    print(f'Current account value: {user_state["marginSummary"]["accountValue"]}')

    in_pos = False
    size = 0
    entry_px = 0
    pnl_perc = 0
    long = None

    for position in user_state["assetPositions"]:
        if position["position"]["coin"] == symbol and float(position["position"]["szi"]) != 0:
            in_pos = True
            size = float(position["position"]["szi"])
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            long = size > 0
            print(f'Position found for {symbol}: Size={size}, Entry Price={entry_px}, PnL%={pnl_perc}')
            break

    return in_pos, size, entry_px, pnl_perc, long

def cancel_symbol_orders(account, symbol, exchange, info):
    """Cancels all open orders for the specified symbol."""
    open_orders = info.open_orders(account.address)

    for open_order in open_orders:
        if open_order['coin'] == symbol:
            print(f'Cancelling order {open_order}')
            exchange.cancel(open_order['coin'], open_order['oid'])

def open_order_deluxe(symbol_info, size, account, exchange, info):
    """Places a limit order and sets stop loss and take profit orders for short positions."""
    symbol = symbol_info["Symbol"]
    entry_price = symbol_info["Entry Price"]
    sl = symbol_info["Stop Loss"]
    tp = symbol_info["Take Profit"]

    _, rounding = get_sz_px_decimals(symbol)
    entry_price = round(entry_price, rounding)
    sl = round(sl, rounding)
    tp = round(tp, rounding)

    print(f'Symbol: {symbol}, Entry Price: {entry_price}, Stop Loss: {sl}, Take Profit: {tp}')

    # Determine the order side (sell to open a short position)
    is_buy = False  # For short positions, we sell first

    cancel_symbol_orders(account, symbol, exchange, info)

    # Place the entry order
    order_result = exchange.order(
        symbol,
        is_buy,
        size,
        entry_price,
        {"limit": {"tif": "Gtc"}}
    )
    print(f"Limit order result for {symbol}: {order_result}")

    # Place the stop loss order (buy to close)
    stop_order_type = {"trigger": {"triggerPx": sl, "isMarket": True, "tpsl": "sl"}}
    stop_result = exchange.order(
        symbol,
        not is_buy,  # Opposite side to close the position
        size,
        sl,
        stop_order_type,
        reduce_only=True
    )
    print(f"Stop loss order result for {symbol}: {stop_result}")

    # Place the take profit order (buy to close)
    tp_order_type = {"trigger": {"triggerPx": tp, "isMarket": True, "tpsl": "tp"}}
    tp_result = exchange.order(
        symbol,
        not is_buy,
        size,
        tp,
        tp_order_type,
        reduce_only=True
    )
    print(f"Take profit order result for {symbol}: {tp_result}")

def main():
    account1 = eth_account.Account.from_key(private_key_CCI_1h)

    # Initialize Exchange
    exchange = Exchange(account1, constants.MAINNET_API_URL)
    
    # Initialize Info
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    for symbol in symbols:
        # Fetch the OHLCV data
        snapshot_data = get_ohlcv2(symbol, timeframe, 2)
        ohlcv_data = process_data_to_df(snapshot_data)

        if not ohlcv_data.empty:
            # Fetch symbol-specific parameters
            cci_period = symbols_data[symbol]['cci_period']
            tp_pct = symbols_data[symbol]['tp_pct']
            sl_pct = symbols_data[symbol]['sl_pct']
            symbol_leverage = symbols_data[symbol]['leverage']

            # Run the CCI strategy
            action, current_price, tp_price, sl_price, cci_current, signal_time = cci_strategy(
                symbol, ohlcv_data, cci_period, tp_pct, sl_pct
            )

            print(f"{symbol} - Action: {action}, Current Price: {current_price}, TP: {tp_price}, SL: {sl_price}, CCI: {cci_current}")

            if action == "SELL":
                # Check if the signal is from a new candle
                last_signal = last_signal_time.get(symbol)
                if last_signal == signal_time:
                    print(f"Signal for {symbol} has already been acted upon. Skipping.")
                    continue  # Skip to the next symbol

                # Update the last signal time
                last_signal_time[symbol] = signal_time

                print(f"Executing SHORT order for {symbol} at {current_price}")
                size = adjust_leverage_usd_size(
                    symbol, order_usd_size, symbol_leverage, account1, exchange, info
                )

                # Check if already in position
                in_pos, pos_size, entry_px, pnl_perc, long = get_position(
                    symbol, account1, info
                )

                if not in_pos:
                    print(f'Not in position for {symbol}')
                    # Create a dictionary to hold the symbol information before opening the order
                    symbol_info = {
                        "Symbol": symbol,
                        "Entry Price": current_price,
                        "Stop Loss": sl_price,
                        "Take Profit": tp_price,
                    }

                    open_order_deluxe(symbol_info, size, account1, exchange, info)
                    print(f"Short order opened for {symbol}")
                else:
                    print(f'Already in a position for {symbol}')
            else:
                print(f"HOLD for {symbol}")
        else:
            print(f"No data available for {symbol}")

print('Running CCI-based shorting bot...')
main()
schedule.every(1).minutes.do(main)

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"Encountered an error: {e}")
        time.sleep(10)
