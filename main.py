from flask import Flask, render_template, jsonify
import requests
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Layer
import numpy as np
import joblib
import tensorflow as tf
import math
import json
import plotly.graph_objects as go
import plotly.utils
app = Flask(__name__)

# Define the custom Swish activation function
class Swish(Layer):
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)

# Load the historical data
historical_data = pd.read_csv('historical_data.csv', parse_dates=['Start'], index_col='Start')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the model with the custom Swish activation function
model = load_model('bitcoin_model.h5', custom_objects={'Swish': Swish})

def get_bitcoin_stats():
    url = 'https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT'
    response = requests.get(url)
    data = response.json()
    stats = {
        'price': float(data['lastPrice']),
        'low': float(data['lowPrice']),
        'high': float(data['highPrice']),
        'open': float(data['openPrice']),
        'close': float(data['prevClosePrice']),
        'volume': float(data['volume']),
        'market_cap': float(data['quoteVolume'])
    }
    return stats

def predict_next_day(scaled_data, window_size=60):
    dataX = [scaled_data[i:i + window_size] for i in range(len(scaled_data) - window_size)]
    dataX = np.array(dataX)
    predicted_price = model.predict(dataX[-1].reshape(1, window_size, 1))
    return float(scaler.inverse_transform(predicted_price)[0][0])
def calculate_angle(y1, y2, x1, x2):
    return np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
def create_moving_averages_plot(df):
    # Ensure 'End' column is in datetime format
    df['End'] = pd.to_datetime(df['End'], format='mixed', dayfirst=True)

    latest_dates = sorted(df['End'].dt.date.unique())[-7:]
    latest_data = df[df['End'].dt.date.isin(latest_dates)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=latest_data['End'], y=latest_data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=latest_data['End'], y=latest_data['SMA_5'], mode='lines', name='SMA 5'))
    fig.add_trace(go.Scatter(x=latest_data['End'], y=latest_data['SMA_7'], mode='lines', name='SMA 7'))
    fig.add_trace(go.Scatter(x=latest_data['End'], y=latest_data['SMA_20'], mode='lines', name='SMA 20'))

    for i in range(1, len(latest_data)):
        angle_5 = calculate_angle(latest_data['SMA_5'].iloc[i-1], latest_data['SMA_5'].iloc[i], i-1, i)
        angle_20 = calculate_angle(latest_data['SMA_20'].iloc[i-1], latest_data['SMA_20'].iloc[i], i-1, i)

        if angle_5 > 45 and angle_20 > 45:
            fig.add_annotation(x=latest_data['End'].iloc[i], y=latest_data['Close'].iloc[i],
                               text='Strong Buy', showarrow=True, arrowhead=1, ax=-10, ay=-30, bgcolor='green')

        if angle_5 < -45 and angle_20 < -45:
            fig.add_annotation(x=latest_data['End'].iloc[i], y=latest_data['Close'].iloc[i],
                               text='Strong Sell', showarrow=True, arrowhead=1, ax=-10, ay=30, bgcolor='red')


    fig.update_layout(
    title='Bitcoin Moving Averages Plot',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_dark',
    xaxis=dict(
        type='date',
        tickformat='%d/%m/%Y %H:%M:%S'
    ),
    paper_bgcolor='#1e1e1e',
    plot_bgcolor='#1e1e1e',
    font=dict(color='#e0e0e0')
)

    box_data = latest_data.copy()
    box_data['Hour'] = box_data['End'].dt.hour
    box_data['Day'] = box_data['End'].dt.date

    for hour in box_data['Hour'].unique():
        hourly_data = box_data[box_data['Hour'] == hour]
        fig.add_trace(go.Box(x=hourly_data['End'], y=hourly_data['Close'], name=f'Hour {hour}', boxmean=True, boxpoints='all', marker_color='lightblue'))

    for day in box_data['Day'].unique():
        daily_data = box_data[box_data['Day'] == day]
        fig.add_trace(go.Box(x=daily_data['End'], y=daily_data['Close'], name=str(day), boxmean=True, boxpoints='all', marker_color='lightgreen'))

    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

@app.route('/update_data', methods=['GET'])
def update_data():
    stats = get_bitcoin_stats()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_row = pd.DataFrame([{
        'Start': current_time,
        'End': current_time,
        'Open': stats['open'],
        'High': stats['high'],
        'Low': stats['low'],
        'Close': stats['price'],
        'Volume': stats['volume'],
        'Market Cap': stats['market_cap']
    }])

    new_row['Start'] = pd.to_datetime(new_row['Start'])
    new_row.set_index('Start', inplace=True)

    global historical_data
    historical_data = pd.concat([historical_data, new_row])
    historical_data.sort_index(inplace=True)


    # Save the updated historical data to the CSV file
    historical_data.to_csv('historical_data.csv')

    scaled_data = scaler.transform(historical_data[['Close']])

    try:
        predicted_price = predict_next_day(scaled_data)
    except ValueError as e:
        predicted_price = None
    # Ensure 'End' column is in datetime format
    historical_data['End'] = pd.to_datetime(historical_data['End'], format='mixed', dayfirst=True)

    # Calculate moving averages
    historical_data['SMA_5'] = historical_data['Close'].rolling(window=5).mean()
    historical_data['SMA_7'] = historical_data['Close'].rolling(window=7).mean()
    historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()

    # Create moving averages plot
    moving_averages_plot = create_moving_averages_plot(historical_data)

    # Replace NaN values with None
    historical_data_dict = historical_data.reset_index().to_dict(orient='records')
    for record in historical_data_dict:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    data = {
        'current_stats': stats,
        'predicted_price': predicted_price,
        'historical_data': historical_data_dict,
        'moving_averages_plot': moving_averages_plot
    }
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')

