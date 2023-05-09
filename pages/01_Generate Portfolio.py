# Import the required libraries and dependencies
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
from funcs.financial_analysis import get_benchmark_returns

# Define Streamlit app
st.title(':bank: Portfolio Generator')
st.warning(':warning: Please note that the app is still in development and most assets will not generate data yet!. :warning:')

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
assets =alpaca.list_assets()

# Create a list of asset symbols
symbols = [asset.symbol for asset in assets]
# Create a search bar for selecting assets
selected_symbols = st.multiselect("Select assets",options=symbols,default=[])

# Create select box for selecting timeframe
start_date_options = ['1 year', '3 years', '5 years']
selected_start_date = st.selectbox('Select the startdate (how many years back?)', start_date_options)
st.session_state['selected_start_date'] = selected_start_date

timeframe_options = ['Day', 'Week', 'Month']
selected_timeframe = st.selectbox('Select timeframe', timeframe_options)

st.session_state['selected_timeframe'] = selected_timeframe
    
@st.cache_data
def save_inputs(selected_symbols, selected_timeframe, selected_start_date):
    
    # Filter assets based on search bar input
    filtered_assets = [asset for asset in assets if asset.symbol in selected_symbols]
    st.session_state['filtered_assets'] = filtered_assets
    st.write('These are your selected assets, and their accompanied symbols:')
    for i, asset in enumerate(filtered_assets):
        st.write(f"{i+1}. {asset.name} ({asset.symbol})")

    # Create a list of symbols from filtered assets
    tickers = [asset.symbol for asset in filtered_assets]
    st.write(f"these are the tickers: {tickers}")
    st.session_state['tickers'] = tickers
    st.session_state['selected_symbols'] = selected_symbols
    st.session_state['selected_timeframe'] = selected_timeframe
    return tickers, selected_timeframe, selected_start_date
    
tickers, selected_timeframe, selected_start_date = save_inputs(selected_symbols, selected_timeframe, selected_start_date)

# Define function to get portfolio data
@st.cache_data
def get_portfolio_data(tickers, selected_timeframe, selected_start_date):
    # Set timeframe
    if selected_timeframe == 'Day':
        timeframe = str('1Day')
        st.session_state['timeframe'] = timeframe
    elif selected_timeframe == 'Week':
        timeframe = str('1Week')
        st
    elif selected_timeframe == 'Month':
        timeframe = str('1Month')
    else:
        st.error('Invalid timeframe selected')
        
        
    st.session_state['timeframe'] = timeframe
    # Set startdate 
    if selected_start_date == '1 year':
        start_date = pd.Timestamp("2022-02-01", tz="America/New_York").isoformat()
    elif selected_start_date == '3 years':
        start_date = pd.Timestamp("2020-02-01", tz="America/New_York").isoformat()
    elif selected_start_date == '5 years':
        start_date = pd.Timestamp("2018-02-01", tz="America/New_York").isoformat()
    else:
        st.error('Invalid startdate selected')

        
    st.session_state['start_date'] = start_date
    end_date = pd.Timestamp("2023-02-01", tz="America/New_York").isoformat()
    st.session_state['end_date'] = end_date
    # Get closing prices for each ticker
    df_portfolio = alpaca.get_bars(
        tickers,
        timeframe,
        start=start_date,
        end=end_date
    ).df


    st.session_state['df_portfolio'] = df_portfolio
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['timeframe'] = timeframe   
    # Return the portfolio DataFrame
    return df_portfolio, start_date, end_date, timeframe
    




# Create a Streamlit button for Generating Data
if st.button('Create Portfolio'):
    # Call the function to get the portfolio data
    df_portfolio, start_date, end_date, timeframe  = get_portfolio_data(tickers, selected_timeframe, selected_start_date)

    # Prepare an empty dictionary to store dataframes
    dfs = {}

    # Separate ticker data and store in dictionary
    for ticker in tickers:
        df = df_portfolio[df_portfolio['symbol']==ticker].drop('symbol', axis=1)
        dfs[ticker] = df

    # Concatenate the ticker DataFrames
    df_tickers = pd.concat(dfs, axis=1, keys=tickers)

    # Concatenate the DataFrames
    df_assets = df_tickers.dropna().copy()

    # Save the file
    filename = 'portfolio.csv'
    st.session_state['filename'] = filename
    filepath = Path(f'folder/subfolder/{filename}')
    st.session_state['filepath'] = filepath  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_assets.to_csv(filepath)  
    st.success(f"'{filename}' has been saved in the folder/subfolder/ directory!")

    # Display the DataFrame
    st.session_state['df_assets'] = df_assets
    st.dataframe(df_assets)

    # Prepare a new empty dictionary to store dataframes:
    bench_dfs = {}

    bench_tickers = ['SPY', 'QQQ', 'DIA']

    # Separate ticker data and store in dictionary
    for ticker in bench_tickers:
        df = df_portfolio[df_portfolio['symbol']==ticker].drop('symbol', axis=1)
        dfs[ticker] = df

    df_bench_tickers = pd.read_csv("data/benchmark_returns.csv")
    

    # Concatenate the ticker DataFrames
    df_bench_tickers = pd.concat(dfs, axis=1, keys=bench_tickers)

    # Save the benchmark_returns.csv file
    bench_filename = 'benchmark_returns.csv'
    st.session_state['bench_filename'] = bench_filename
    bench_filepath = Path(f'data/{bench_filename}')
    st.session_state['bench_filepath'] = bench_filepath
    bench_filepath.parent.mkdir(parents=True, exist_ok=True)
    df_bench_tickers.to_csv(bench_filepath)
    st.success(f"'{bench_filename}' has been saved in the folder/subfolder/ directory!")


    # Save the benchmark_returns df 
    benchmark_returns = get_benchmark_returns()
    st.session_state['benchmark_returns'] = benchmark_returns

if st.button('Display Benchmark Returns Dataframe'):
# Explain what benchmark returns are used for:
        st.write("Benchmark returns are used to compare the performance of your portfolio to the performance of the market. The benchmark returns are calculated using the following ETFs: SPY, QQQ, and DIA.")
        # Display the benchmark data dataframe:
        bench_filename = 'benchmark_df.csv'
        df_bench_tickers = Path(f'data/{bench_filename}')
        st.dataframe(df_bench_tickers)



#     #     #
# # Create a Streamlit button for Generating Data :
# if st.button('Create Portfolio'):
#     dfs = {}
#     # Call the function to get the portfolio data
#     df_portfolio, start_date, end_date, timeframe  = get_portfolio_data(tickers, selected_timeframe, selected_start_date)
#     # Display the dataframe
#     # Reorganize the DataFrame
#     # Separate ticker data
#     for ticker in tickers:
#         df = df_portfolio[df_portfolio['symbol']==ticker].drop('symbol', axis=1)
#         dfs[ticker] = df
        

#         # Concatenate the ticker DataFrames
#         for ticker in tickers:
#             df_tickers = pd.concat(dfs, axis=1, keys=tickers)
#             # Concatenate the DataFrames
#             # Drop Nulls
#             df_assets = df_tickers.dropna().copy()
#             df_portfolio = df_assets
#             # df_portfolio.dropna(inplace=True)     

#         # Save the file
#         filename = 'portfolio.csv'
#         st.session_state['filename'] = filename
#         filepath = Path(f'folder/subfolder/{filename}')
#         st.session_state['filepath'] = filepath  
#         filepath.parent.mkdir(parents=True, exist_ok=True)  
#         df_assets.to_csv(filepath)  
#         st.success(f"'{filename}' has been saved in the folder/subfolder/ directory!")

#         # Display the DataFrame
#         st.session_state['df_assets'] = df_assets
#         st.dataframe(df_assets)

    
