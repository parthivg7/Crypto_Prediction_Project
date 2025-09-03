import ccxt
import pandas as pd
import time

exchange_id = 'binance'
exchange = getattr(ccxt, exchange_id)()

symbol = 'BTC/USDT'  # Bitcoin to Tether
timeframe = '1h'  # 1-hour candles
limit = 1000  # Number of data points to fetch (limit of 1000 is common)

# Fetch historical data
all_ohlcv = []
since = exchange.parse8601('2022-01-01T00:00:00Z') # Starting date
end_time = int(time.time() * 1000)

print(f"Fetching historical data for {symbol} from {exchange_id}...")

while since < end_time:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    if len(ohlcv) == 0:
        break
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 1  # Get next batch from the last timestamp
    print(f"Fetched {len(all_ohlcv)} data points so far.")

df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Drop any duplicate timestamps
df = df[~df.index.duplicated(keep='first')]

# Handle potential missing data by forward filling
df.fillna(method='ffill', inplace=True)

df.to_csv(f'{symbol.replace("/", "")}_historical_data.csv')
print(f"Historical data for {symbol} saved successfully.")
print(df.tail())