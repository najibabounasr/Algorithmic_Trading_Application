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
import datetime


# Define Streamlit app
st.title(':bank: Portfolio Generator')

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
        st.session_state['timeframe'] = timeframe
    elif selected_timeframe == 'Month':
        timeframe = str('1Month')
        st.session_state['timeframe'] = timeframe
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
    end_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
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
    

df_portfolio, start_date, end_date, timeframe = get_portfolio_data(tickers, selected_timeframe, selected_start_date)
st.session_state['df_portfolio'] = df_portfolio
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date
st.session_state['timeframe'] = timeframe


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

    # Create MultiIndex for column labels
    mi = pd.MultiIndex.from_tuples([(ticker, col) for ticker in tickers for col in df_assets.columns.levels[1]])

    # Set MultiIndex as column labels
    df_assets.columns = mi

    # Save the file
    filename = 'portfolio_data.csv'
    st.session_state['filename'] = filename
    filepath = Path(f'data/{filename}')
    st.session_state['filepath'] = filepath  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df_assets.to_csv(filepath)  
    st.success(f"'{filename}' has been saved in the folder/subfolder/ directory!")

    # Display the DataFrame
    st.session_state['df_assets'] = df_assets
    st.dataframe(df_assets)

    # Prepare a new empty dictionary to store dataframes:
    bench_dfs = {}

    bench_tickers = ['SPY']

    # Separate ticker data and store in dictionary
    for ticker in bench_tickers:
        df = df_portfolio[df_portfolio['symbol']==ticker].drop('symbol', axis=1)
        dfs[ticker] = df

    df_bench_tickers = pd.read_csv("data/benchmark/benchmark_df.csv")
    

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

    tickers, selected_timeframe, selected_start_date = save_inputs(selected_symbols, selected_timeframe, selected_start_date)
    # Save the benchmark_returns df 
    benchmark_returns = get_benchmark_returns()
    st.session_state['benchmark_returns'] = benchmark_returns

    curated_df = pd.read_csv("data/portfolio_data.csv", 
                        index_col = [0],
                        parse_dates = True,
                        infer_datetime_format = True,)
    # move the value of the first row down to the second row:
    curated_df.iloc[1] = curated_df.iloc[0]
    # remove the first row of the dataframe, as it is not needed:
    curated_df = curated_df.iloc[1:, :]

    # rename the index of the dataframe to 'timeframe':
    curated_df.index.name = 'timestamp'
    location = 3
    closing_prices_df = pd.DataFrame()
    curated_df = df_assets
    # Create the closing prices df:
    for ticker in tickers:
        df = pd.DataFrame()
        df[ticker] = curated_df.iloc[:, location]
        # col_name = f"{ticker}"
        # closing_prices_df.loc[:, col_name] = df[ticker]
        closing_prices_df.loc[:,f"{ticker}_close"] = df[ticker]
        col_name = f"{ticker}_weighted_price"
        location = location + 7 

    df_portfolio = pd.DataFrame(curated_df)


    # Remove the first two rows of the closing_prices_df:
    # closing_prices_df = closing_prices_df.iloc[2:, :]
    closing_prices_df.to_csv("data/grouped_dfs/closing_prices_df.csv")
    # Concatenate first two rows with original df
    first_two_rows = closing_prices_df.iloc[:1]
    closing_prices_df = pd.concat([first_two_rows, closing_prices_df])
    # remove the first row of the dataframe:
    closing_prices_df = closing_prices_df.iloc[2:, :]


    # Comvert the values of the column to floats:
    for ticker in tickers:
        closing_prices_df[f"{ticker}_close"] = closing_prices_df[f"{ticker}_close"].astype(float)

    closing_prices_df.to_csv("data/grouped_dfs/closing_prices_df.csv")


if st.button('Display Benchmark Returns Dataframe'):
# Explain what benchmark returns are used for:
        st.write("Benchmark returns are used to compare the performance of your portfolio to the performance of the market. The benchmark returns are calculated using the S&P 500 index.")
        # Display the benchmark data dataframe:
        bench_filename = 'benchmark/benchmark_df.csv'
        df_bench_tickers = pd.read_csv(
            Path(f'data/{bench_filename}'),
            index_col='timestamp',
            parse_dates=True,
            infer_datetime_format=True
        )
        # Add a title ontop of the dataframe:
        st.title('S&P 500 Benchmark Returns Dataframe')
        st.dataframe(df_bench_tickers)




# Start by Getting the weights:
# Prompt the user for the weights of each asset in their portfolio:
st.sidebar.header("Portfolio Weights")

# Create a dictionary to hold the weights of each asset
weights = {}

# Create a for loop that will prompt the user for the weight of each asset
for ticker in tickers:
    weights[ticker] = st.sidebar.number_input(f"What is the weight of {ticker} in your portfolio? (Please enter a value between 0 and 1)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    # add functionality, so that the weights of the assets add up to 1

# Create a dataframe that holds the weights of each asset
df_weights = pd.DataFrame(weights, index=[0])

# Make sure that the dataframe is labeled correctly, with the asset ticker as the column name
df_weights = df_weights.T
df_weights.columns = ['weights']
df_transposed_weights = df_weights
df_weights = df_weights.transpose()
weights = df_weights.to_dict('records')[0]

st.session_state['weights'] = weights
st.session_state['df_weights'] = df_weights
st.session_state['df_transposed_weights'] = df_transposed_weights
st.session_state['tickers'] = tickers


    
