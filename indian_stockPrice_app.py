
#IMPORT LIBRARIES

import pandas as pd
import numpy as np
from datetime import date
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import streamlit as st

# STOCK DATA FROM YAHOO FINANCE
import yfinance as yf

# =========================================================================================
#                               CONFIGURE PAGE | PAGE DETAILS
# =========================================================================================

# --- CONFIG PAGE AND PAGE LAYOUT ---
st.set_page_config(page_title='Indian Stock Price Prediction', layout='wide')

st.title("Indian Stock Price Prediction")  # Set page title
st.markdown('---')  # Add a page break

# --- DESCRIBE WEB APPLICATION ---
st.header('How to use the web app?')

bullet_points = '''
- FIRST INPUT:
    - Enter the ticker symbol for an Indian stock (use format: TICKER.NS, e.g., RELIANCE.NS).
- TIME PERIOD: 
    - Choose the prediction horizon for which you want the forecast.
'''
st.write(bullet_points)

# --- USER INPUT SECTION ---
st.header("Input Section")

# Enter the ticker symbol in the NSE format
ticker = st.text_input("Enter Indian stock ticker (NSE format, e.g., RELIANCE.NS)", value="RELIANCE.NS")

# Fetch stock data from Yahoo Finance with updated ticker format
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end=date.today().strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# --- PREDICTION MODEL SETUP ---
# Prepare data for prediction
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.dayofyear  # Use day of the year for seasonality

# Features and labels
X = np.array(data['Day']).reshape(-1, 1)
y = data['Close'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Support Vector Regression model
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X_train, y_train)

# Predict future prices
future_days = st.slider('Days to predict into the future:', 1, 30)
future_dates = [data['Date'].max() + pd.Timedelta(days=x) for x in range(1, future_days + 1)]
future_days_numbers = np.array([x.timetuple().tm_yday for x in future_dates]).reshape(-1, 1)

predictions = model.predict(future_days_numbers)

# Display predictions
st.subheader(f'Predicted Prices for {ticker} in INR')
predicted_data = pd.DataFrame({'Date': future_dates, 'Predicted Close Price (INR)': predictions})
st.write(predicted_data)

# Plotting results
st.line_chart(predicted_data.set_index('Date')['Predicted Close Price (INR)'])

st.markdown('---')
st.write("Note: This application is for educational purposes and uses historical data to predict stock trends. "
         "It does not guarantee future price accuracy.")
