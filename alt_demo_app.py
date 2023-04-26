import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
import base64
from pathlib import Path  


# Set the page config
st.set_page_config(
    page_title="Portfolio Returns Calculator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Create a multiselect widget with the page names
page = st.sidebar.radio("Select a page", ("Home", "Generate Portfolio", "Forecast Returns"))

if page == "Home": 
    st.title(':house: Home')
    st.subheader('Customizable Portfolio Returns Calculator :smile')
    st.subheader('This app calculates the returns of a portfolio of assets over a given time period. You can select the assets, the timeframe and the startdate. The app will then return a dataframe with the returns of the portfolio.')


# Use an if statement to display the content of the selected page
elif page == "Generate Portfolio":
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

    st.write('These are your selected assets, and their accompanied symbols:')
    for i, asset in enumerate(filtered_assets):
        st.write(f"{i+1}. {asset.name} ({asset.symbol})")



    # Create a list of symbols from filtered assets
    tickers = [asset.symbol for asset in filtered_assets]
    st.write(f"these are the tickers: {tickers}")
    # Create select box for selecting timeframe
    start_date_options = ['1 year', '3 years', '5 years']
    selected_startdate = st.selectbox('Select the startdate (how many years back?)', start_date_options)
    timeframe_options = ['Day', 'Week', 'Month']
    selected_timeframe = st.selectbox('Select timeframe', timeframe_options)


    # Define function to get portfolio data
    @st.cache
    def get_portfolio_data(tickers, selected_timeframe, selected_startdate):
        # Set timeframe
        if selected_timeframe == 'Day':
            timeframe = str('1Day')
        elif selected_timeframe == 'Week':
            timeframe = str('1Week')
        elif selected_timeframe == 'Month':
            timeframe = str('1Month')
        else:
            st.error('Invalid timeframe selected')

        # Set startdate 
        if selected_startdate == '1 year':
            start_date = pd.Timestamp("2022-02-01", tz="America/New_York").isoformat()
        elif selected_startdate == '3 years':
            start_date = pd.Timestamp("2020-02-01", tz="America/New_York").isoformat()
        elif selected_startdate == '5 years':
            start_date = pd.Timestamp("2018-02-01", tz="America/New_York").isoformat()
        else:
            st.error('Invalid startdate selected')
        
        
        end_date = pd.Timestamp("2023-02-01", tz="America/New_York").isoformat()

        # Get closing prices for each ticker
        df_portfolio = alpaca.get_bars(
            tickers,
            timeframe,
            start=start_date,
            end=end_date
        ).df

        # Return the portfolio DataFrame
        return df_portfolio
        

    # Prepare an empty dictionary to store dataframes:
    dfs = {}

    # Create a Streamlit button for Generating Data :
    if st.button('Create Portfolio'):
        # Call the function to get the portfolio data
        df_portfolio = get_portfolio_data(tickers, selected_timeframe, selected_startdate)
        # Display the dataframe
        # Reorganize the DataFrame
        # Separate ticker data
        for ticker in tickers:
            df = df_portfolio[df_portfolio['symbol']==ticker].drop('symbol', axis=1)
            dfs[ticker] = df

        # Concatenate the ticker DataFrames
        for ticker in tickers:
            df_tickers = pd.concat(dfs, axis=1, keys=tickers)
            # Concatenate the DataFrames
            # Drop Nulls
            df_assets = df_tickers.dropna().copy()
            df_portfolio = df_assets
            # df_portfolio.dropna(inplace=True)     

        # Save the file
        filename = 'portfolio.csv'
        filepath = Path(f'folder/subfolder/{filename}')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df_assets.to_csv(filepath)  
        st.success(f"'{filename}' has been saved in the folder/subfolder/ directory!")

        # Display the DataFrame
        st.dataframe(df_assets)



# PART TWO : NORMALIZE DATA
    

elif page == "Forecast Returns":
    st.title(":money_with_wings: Forecast returns")
    st.write("Using Monte Carlo simulation to forecast returns!")






# def get_asset_data(selected_symbols):
#     # Initialize an empty dictionary to store the dataframes
#     asset_data = []

#     # Loop through each selected asset symbol
#     for symbol in selected_symbols:
#         # Filter the portfolio dataframe for the selected symbol
#         df = df_portfolio[df_portfolio['symbol']==symbol].drop('symbol', axis=1)

#         # Add the dataframe to the dictionary with the symbol as the key
#         asset_data.append(df)

#         # Concatenate the dataframes in the list
#         concatenated_df = pd.concat(asset_data, axis=1, keys=selected_symbols)

#     return asset_data, concatenated_df

# concatenated_df = get_asset_data(selected_symbols)



# Get the dataframes for the selected assets
# asset_data = get_asset_data(selected_symbols)

# Loop through the dataframes and do something with each one




