# Standard library imports
import json
import time
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
import requests
import schedule
from eth_account.signers.local import LocalAccount
import eth_account
from dontshare import private_key_1h

# Local imports
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

order_usd_size = 75
leverage = 3
timeframe = '1h'

#Define symbol-specific parameters
symbols = ['BTC', 'ETH', 'SOL', 'SUI', 'TIA', 'kPEPE', 'FTM', 'WIF', 'SEI', 'AAVE', 'WLD', 'STX', 'POPCAT']
symbols_data = {
    'BTC': {'sma_period': 18, 'buy_range': 3, 'sell_range': 5,},
    'ETH': {'sma_period': 36, 'buy_range': 5, 'sell_range': 3,},
    'SOL': {'sma_period': 46, 'buy_range': 7, 'sell_range': 7,}, 
    'SUI': {'sma_period': 48, 'buy_range': 11, 'sell_range': 9,},
    'TIA': {'sma_period': 48, 'buy_range': 15, 'sell_range': 9,},
    'kPEPE': {'sma_period': 18, 'buy_range': 11, 'sell_range': 15,},
    'FTM': {'sma_period': 28, 'buy_range': 1, 'sell_range': 5,},
    'WIF': {'sma_period': 10, 'buy_range': 1, 'sell_range': 15,},
    'SEI': {'sma_period': 48, 'buy_range': 17, 'sell_range': 17,},
    'AAVE': {'sma_period': 32, 'buy_range': 19, 'sell_range': 13,},
    'WLD': {'sma_period': 48, 'buy_range': 17, 'sell_range': 11,},
    'STX': {'sma_period': 30, 'buy_range': 11, 'sell_range': 9,},
    'POPCAT': {'sma_period': 34, 'buy_range': 17, 'sell_range': 19,},
}

# Function to get the buy/sell range for a symbol
def get_ranges(symbol):
    return symbols_data.get(symbol, {'buy_range': (0, 0), 'sell_range': (0, 0)})

#Fetch symbols from the API
def get_symbols():
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print('Error:', response.status_code)
        return []
    
    data = response.json()
    symbols = [symbol['name'] for symbol in data['universe']]
    print("Fetched Symbols: ", symbols)
    return symbols
        
def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

# Mean reversion startegy function
def mean_reversion_strategy(symbol, data, sma_period, buy_range, sell_range):
    # Calculate SMA
    data['sma'] = calculate_sma(data, sma_period)
    #print(data) # For debugging; may remove this later

    #Ensure there is enough data to calculate SMA
    if len(data) < sma_period:
        print(f"Not enough data to calculate SMA for period {sma_period} for {symbol}")
        return "HOLD", None, None, None
    
    # Get the last valid SMA value (non-NaN)
    last_valid_sma = data['sma'].dropna().iloc[-1]

    # Calculate buying and selling thresholds
    buy_threshold = last_valid_sma * (1 - buy_range / 100)
    sell_threshold = last_valid_sma * (1 + sell_range / 100)

    # Get the last valid close price (non-NaN)
    current_price = data['close'].iloc[-1]

    # convert the numpy numbers of current_price, buy_threshold and sell_threshold to python numbers
    current_price = float(current_price)
    buy_threshold = float(buy_threshold)
    sell_threshold = float(sell_threshold)

    # Strategy: Buy if current price is below buy threshold, sell if current price is above sell threshold
    if current_price <= buy_threshold:
        action = "BUY"
    elif current_price >= sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"

    return action, current_price, buy_threshold, sell_threshold

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

        # Calculate support and resistance, excluding the last two rows for the calculation
        if len(df) > 2: # Check if DataFrame has more than 2 rows to avoid errors
            df['support'] = df[:-2]['close'].min()
            df['resis'] = df[:-2]['close'].max()
        else: # If DataFrame has 2 or fewer rows, use the available 'close' prices for calculation
            df['support'] = df['close'].min()
            df['resis'] = df['close'].max()

        return df
    else: 
        return pd.DataFrame() # REturn empty DataFrame if no data
    
def ask_bid(symbol):
    '''this gets the ask and bid for any symbol passed in'''

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book', 
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']

    # get ask bid 
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

def get_sz_px_decimals(symbol):

    ''' this returns size devimals and price decimals '''

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # Success
        data = response.json()
        #print(data)
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']

        else:
            print('symbol not found')

    else:
        # Error
        print('Error:', response.status_code)

    
    ask = ask_bid(symbol)[0]

    # Compute the number of decimal points in the ask price
    ask_str = str(ask)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0 

    print(f'{symbol} this is the price: {ask} sz decimal(s) {sz_decimals}, px decimal(s) {px_decimals}')

    return sz_decimals, px_decimals
    
def adjust_leverage_usd_size(symbol, usd_size, leverage, account):

        '''
        this calculates size based off a specific USD dollar amount
        '''

        print('leverage:', leverage)

        #account: LocalAccount = eth_account.Account.from_key(key)
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get the user state and print out leverage information for ETH
        user_state = info.user_state(account.address)
        acct_value = user_state["marginSummary"]["accountValue"]
        acct_value = float(acct_value)

        print(exchange.update_leverage(leverage, symbol))

        price = ask_bid(symbol)[0]

        # size == balance / price * leverage
        # INJ 6.95 ... at 10x lev...10 INJ == $cost 6.95
        size = (usd_size / price) * leverage
        size = float(size)
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        print(f'this is the size of crypto we will be using {size}')

        user_state = info.user_state(account.address)

        return leverage, size

def get_position(symbol, account):

    '''
    gets the current position info, like size etc. 
    '''

    # account = LocalAccount = eth_account.Account.from_key(key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    
    positions = []
    print(f'this is the symbol {symbol}')
    print(user_state["assetPositions"])

    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True 
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"])*100
            print(f'this is the pnl perc {pnl_perc}')
            break 
    else:
        in_pos = False 
        size = 0 
        pos_sym = None 
        entry_px = 0 
        pnl_perc = 0

    if size > 0:
        long = True 
    elif size < 0:
        long = False 
    else:
        long = None 

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

def cancel_symbol_orders(account, symbol):
    """
    Cancels all open orders for the specified symbol.

    Parameters:
    - account: The trading account
    - symbol: The symbol (coin) for which to cancel open orders
    """
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    open_orders = info.open_orders(account.address)
    print(open_orders)

    print('Above are the open orders... need the cancel any...')
    for open_order in open_orders:
        if open_order['coin'] == symbol:
            print(f'Cancelling order {open_order}')
            exchange.cancel(open_order['coin'], open_order['oid'])

def open_order_deluxe(symbol_info, size, account):
    """
    Places a limit order and sets stop loss and take profit orders.

    Parameters:
    - symbol_info: A row from a DataFrame containing symbol, entry price, stop loss, and take profit
    - tp: The take profit price
    - sl: The stop loss price
    """
    # config = utils.get_config()
    # account = eth_account.Account.from_key(config["secret_key"])

    print(f'opening order for {symbol_info["Symbol"]} size {size}')


    exchange = Exchange(account, constants.MAINNET_API_URL)


    symbol = symbol_info["Symbol"]
    entry_price = symbol_info["Entry Price"]

    sl = symbol_info["Stop Loss"]
    tp = symbol_info["Take Profit"]

    _, rounding = get_sz_px_decimals(symbol)
    # round tp and sl

    if symbol == 'BTC':
        tp = int(tp)
        sl = int(sl)
    else:
        tp = round(tp, rounding)
        sl = round(sl, rounding)

    print(f'symbol: {symbol}, entry price: {entry_price}, stop loss: {sl}, take profit: {tp}')

    # Determine the order side (buy or sell)
    is_buy = True

    cancel_symbol_orders(account, symbol)

    print(f' entry price: {entry_price} type{type(entry_price)}, stop loss: {sl} type{type(tp)}, take profit: {tp} type{type(tp)}')

    order_result = exchange.order(
        symbol,
        is_buy,
        size, # Assuming a fixed quantity; adjust as needed
        entry_price,
        {"limit": {"tif": "Gtc"}}
    )
    print(f"Limit order result for {symbol}: {order_result}")

    # Place the stop loss order
    stop_order_type = {"trigger": {"triggerPx": sl, "isMarket": True, "tpsl": "sl"}}
    stop_result = exchange.order(
        symbol,
        not is_buy,
        size, # Assuming a fixed quantity; adjust as needed
        sl,
        stop_order_type,
        reduce_only=True
    )
    print(f"Stop loss order result for {symbol}: {stop_result}")

    # Place the take profit order
    tp_order_type = {"trigger": {"triggerPx": tp, "isMarket": True, "tpsl": "tp"}}
    tp_result = exchange.order(
        symbol,
        not is_buy,
        size, # Assuming a fixed quantity: adjust as needed
        tp,
        tp_order_type,
        reduce_only=True
    )
    print(f"Take profit order result for {symbol}: {tp_result}")


# Main function to scan for buy/sell opportunities
def main():
    # Get blanaces from acct 1, size, position, entry price, pnl, long/short
    account1 = LocalAccount = eth_account.Account.from_key(private_key_1h)

    for symbol in symbols:
        # Fetch the OHLCV data
        snapshot_data = get_ohlcv2(symbol, timeframe, 20)
        hourly_snapshots = process_data_to_df(snapshot_data)
        #print(hourly_snapshots)

        if not hourly_snapshots.empty:
            # Fetch symbol-specific parameters
            sma_period = symbols_data[symbol]['sma_period'] 
            buy_range = symbols_data[symbol]['buy_range']
            sell_range = symbols_data[symbol]['sell_range']

            # Run the mean reversion strategy
            action, current_price, buy_threshold, sell_threshold = mean_reversion_strategy(
                symbol, hourly_snapshots, sma_period, buy_range, sell_range
            )
            
            print(f"{symbol} - Action: {action}, Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold} Current Price: {current_price}")

            #action = 'BUY'
            if action == "BUY":
                print(f"Executing BUY order for {symbol} at {current_price}")
                lev, size = adjust_leverage_usd_size(symbol, order_usd_size, leverage, account1)

                # Check if we have a postiion on the symbol
                positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account1)
                
                if not im_in_pos:
                    print(f'not in position for {symbol}')
                    entry_price = current_price # Set entry price as the current price
                    # round entry price to 3
                    entry_price = round(entry_price, 3)
                    # float it
                    entry_price = float(entry_price)

                    # same for buy and sell thresholds
                    buy_threshold = round(buy_threshold, 3)
                    sell_threshold = round(sell_threshold, 3)
                    # float them
                    buy_threshold = float(buy_threshold)
                    sell_threshold = float(sell_threshold)

                    # Create a dictionary to hold the symbol information before opening the order
                    symbol_info = {
                        "Symbol": symbol,
                        "Entry Price": round(buy_threshold, 4),
                        "Stop Loss": round(buy_threshold * 0.3, 4), # Assuming buy threshold is the stop lsos for simplicity
                        "Take Profit": round(sell_threshold, 4), # Assuming sell threshold is the take profit for simplicity
                    }

                    open_order_deluxe(symbol_info, size, account1)
                    print(f"Order opened for {symbol}")

                else:
                    print(f'already in a position for {symbol}')
                
            elif action == "SELL":
                print(f"Flashing SELL for {symbol} but we should already have the orders in place")
                # Place your sell order logic here
            else:
                print(f"HOLD for {symbol}")

print('running algo...')
main() 
schedule.every(5).minutes.do(main)

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"Encountered an error: {e}")
        time.sleep(10)

        

        
