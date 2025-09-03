import pandas as pd
import numpy as np
# Assuming you have already generated the CSV from step 2
df = pd.read_csv('BTCUSDT_historical_data.csv', index_col='timestamp', parse_dates=True)

# Calculate technical indicators
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']

# Create time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

df.dropna(inplace=True)

df.to_csv('BTCUSDT_features.csv')
print("Features created and saved.")
print(df.head())