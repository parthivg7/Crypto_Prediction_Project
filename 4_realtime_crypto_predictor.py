import websocket
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from threading import Thread
import time

# Your Finnhub API key
FINNHUB_API_KEY = "d2pkpihr01qnf9nlgsigd2pkpihr01qnf9nlgsj0"
# The Finnhub WebSocket endpoint for real-time trades
websocket_url = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"

# Load the trained model and scaler
model = load_model("crypto_predictor_model.h5")
scaler = joblib.load('crypto_scaler.pkl')

look_back = 60
current_window = []
first_data_point = True

def on_message(ws, message):
    global current_window, first_data_point
    try:
        data = json.loads(message)
        # Check if the message is a trade update
        if data['type'] == 'trade' and 'data' in data:
            latest_trade = data['data'][0]
            price = latest_trade['p']
            volume = latest_trade['v']
            timestamp = latest_trade['t']
            
            # Update the sliding window with the new data point
            # Note: This is a simplified example. In a real-world scenario, you would
            # calculate the other features (SMA, MACD, etc.) in real-time.
            if len(current_window) >= look_back:
                current_window.pop(0)
            
            # Append the new data to the window
            current_window.append([price, volume, 0, 0, 0, 0, 0])
            
            print(f"Received real-time crypto price from Finnhub: {price}")
            
            if len(current_window) == look_back:
                if first_data_point:
                    print("Sliding window is full. Starting predictions.")
                    first_data_point = False
                
                # Make the prediction
                input_data = np.array(current_window).reshape(1, look_back, -1)
                scaled_input = scaler.transform(input_data[0])
                scaled_input = scaled_input.reshape(1, look_back, -1)
                
                predicted_price_scaled = model.predict(scaled_input)
                
                temp_array = np.zeros((1, scaler.n_features_in_))
                temp_array[0, 0] = predicted_price_scaled
                predicted_price = scaler.inverse_transform(temp_array)[0, 0]
                
                print(f"Predicted next price: {predicted_price:.4f}")
    
    except Exception as e:
        print(f"An error occurred in on_message: {e}")

def on_error(ws, error):
    print(f"### Error ###: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### Connection closed ###")

def on_open(ws):
    def run(*args):
        # Subscribe to a cryptocurrency symbol on Finnhub
        ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')
        print("Subscribed to BTC/USDT on Finnhub. Waiting for data...")
    Thread(target=run).start()

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        websocket_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()