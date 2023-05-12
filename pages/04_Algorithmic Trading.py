import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
import base64
from pathlib import Path
import sqlalchemy  
from funcs.financial_analysis import calculate_volatility, calculate_variance, calculate_beta, calculate_sharpe_ratio, calculate_returns, plot_returns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression



timeframe = st.session_state['timeframe']
end_date = st.session_state['end_date']
start_date = st.session_state['start_date']


# Set the variables for the Alpaca API and secret keys
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = 'https://paper-api.alpaca.markets'

# Create the Alpaca tradeapi.REST object
alpaca = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    base_url,
    api_version="v2")

tickers = st.session_state['tickers']

for ticker in tickers:
    df = alpaca.get_bars(
        ticker,
        timeframe,
        start = start_date,
        end = end_date
    ).df
    df.to_csv(f"data/individual_dfs/{ticker}_df.csv")

     


# Initialize the session state for the app
# This will allow us to store data between app runs

st.title(" :chart: Algorithmic Trading Analysis :chart:")
st.subheader(" Using a binary classification model, to predict trade signals based on short and long-term simple moving average (SMA)")
st.warning("For now, you may only generate trade signal predictions for individual assets-- future updates will allow for more advanced functionality")
st.markdown("---")
# Load the environment variables from the .env file
#by calling the load_dotenv function
load_dotenv()

# List the selected tickers, so that the user can choose one to analyze:
asset = st.multiselect("Select Asset", tickers)
# Import the portfolio dataset into a Pandas Dataframe
ohlcv_df = pd.read_csv(
    Path(f"data/individual_dfs/{asset}_df"), 
    index_col=[0], 
    infer_datetime_format=True, 
    parse_dates=True
)


# Review the DataFrame
ohlcv_df.head()

# Filter the date index and close columns
signals_df = ohlcv_df.loc[:, ["close"]]

# Use the pct_change function to generate  returns from close prices
signals_df["Actual Returns"] = signals_df["close"].pct_change()

# Drop all NaN values from the DataFrame
signals_df = signals_df.dropna()


st.markdown("""[Simple Moving Average (SMA)](https://www.investopedia.com/terms/s/sma.asp)
""")
short_window = st.slider("Select Short SMA",5, 20)
long_window = st.slider("Select Long SMA", 50, 200)

# Generate the fast and slow simple moving averages (4 and 100 days, respectively)
signals_df['SMA_Fast'] = signals_df['close'].rolling(window=short_window).mean()
signals_df['SMA_Slow'] = signals_df['close'].rolling(window=long_window).mean()

signals_df = signals_df.dropna()

# Initialize the new Signal column
signals_df['Signal'] = 0.0

