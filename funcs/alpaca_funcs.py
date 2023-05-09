def setup_alpaca(alpaca_api_key=, alpaca_secret_key=)):
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
    from streamlit import caching
    from funcs.save_inputs import save_inputs
    from funcs.get_returns import get_returns
    from funcs.get_weights import get_weights
    from funcs.apply_weights import apply_weights

    # Load the environment variables from the .env file
    #by calling the load_dotenv function
    load_dotenv()



    # Set the variables for the Alpaca API and secret keys
    alpaca_api_key = os.getenv(str(alpaca_api_key))
    alpaca_secret_key = os.getenv(str(alpaca_secret_key))
    base_url = 'https://paper-api.alpaca.markets'

    # pull in the tickers, selected_timeframe and selected_start_date from the previous page
    tickers, selected_timeframe, selected_start_date = save_inputs(selected_symbols, selected_timeframe, selected_start_date)
        

    # Create the Alpaca tradeapi.REST object
    alpaca = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        base_url,
        api_version="v2")
    
    df_assets = pd.read_csv('folder/subfolder/portfolio.csv')
    
# Create a function, that assigns specific weights to each asset in the portfolio:
@st.cache_data
 def get_weights():
    from funcs.save_inputs import save_inputs
    # Get the tickers from the save_inputs function
    tickers, selected_timeframe, selected_start_date = save_inputs(selected_symbols, selected_timeframe, selected_start_date)
    # Create an empty list to store the weights
    weights = []
    # Loop through each asset ticker
    for ticker in tickers:
        # Get the weight of the asset from the user
        weight = st.number_input(f'Enter the weight of {ticker} (in %)', min_value=0.0, max_value=100.0)
        # Append the weight to the list
        weights.append(weight)
    # Return the list of weights
    return weights
    
    
    
# Create a function that applies the weights to the portfolio data:
@st.cache_data
def apply_weights(weights):
        # Create a copy of the portfolio DataFrame
        df_weighted_portfolio = df_assets.copy()
        # Loop through each asset ticker
        for i, ticker in enumerate(tickers):
            # Apply the weight to the asset
            df_weighted_portfolio[ticker] = df_weighted_portfolio[ticker] * weights[i] / 100
        # Return the weighted portfolio DataFrame
        return df_weighted_portfolio
    

    if st.button('Get Weight'):
        # Call the function to get the weights
        weights = get_weights()
        # Call the function to apply the weights
        df_weighted_portfolio = apply_weights(weights)
        # Display the DataFrame
        st.dataframe(df_weighted_portfolio)