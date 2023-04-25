import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import os


# Define Streamlit app
st.title('Portfolio Returns Calculator')



# Load the environment variables from the .env file
#by calling the load_dotenv function
load_dotenv()


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


# Get all available assets from Alpaca API
assets = alpaca.list_assets()

# Create a list of asset symbols
symbols = [asset.symbol for asset in assets]

# Create a search bar for selecting assets
selected_symbols = st.multiselect(
    "Select assets",
    options=symbols,
    default=[]
)

# Filter assets based on search bar input
filtered_assets = [asset for asset in assets if asset.symbol in selected_symbols]

# Show filtered assets
for asset in filtered_assets:st.write(asset.name, asset.symbol)

# Create a list of symbols from filtered assets
tickers = [asset.symbol for asset in filtered_assets]
    
st.write(f" These will be the assets within your portfolio: {tickers}")

# Create select box for selecting timeframe
timeframe_options = ['1 year', '3 years', '5 years']
selected_timeframe = st.selectbox('Select timeframe', timeframe_options)


def get_ticker_data(ticker):
    # Set time range for data
    start_date = pd.Timestamp('2020-01-01', tz='America/New_York').isoformat()
    end_date = pd.Timestamp.now(tz='America/New_York').isoformat()

    # Get ticker data from Alpaca API
    barset = alpaca.get_bars(tickers, 'day', start=start_date, end=end_date)
    ticker_data = barset[tickers].df

    # Rename columns
    ticker_data.columns = [f"{ticker}_{col}" for col in ticker_data.columns]

    return ticker_data

# Define function to get portfolio data
@st.cache
def get_portfolio_data(tickers, timeframe):
    # Set timeframe
    if timeframe == '1 year':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=1)
    elif timeframe == '3 years':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=3)
    elif timeframe == '5 years':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=5)
    else:
        st.error('Invalid timeframe selected')
        return
    
    end_date = pd.Timestamp.now()

    # Get closing prices for each ticker
    df_portfolio = alpaca.get_bars(
        tickers,
        'day',
        start=start_date,
        end=end_date
    ).df

    # Reorganize the DataFrame
    df_portfolio = df_portfolio.drop('volume', axis=1, level=1)
    df_portfolio.columns = df_portfolio.columns.droplevel(1)
    
    # Return the portfolio DataFrame
    return df_portfolio





