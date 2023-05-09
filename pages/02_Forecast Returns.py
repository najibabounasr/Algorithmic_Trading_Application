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



# Initialize the session state for the app
# This will allow us to store data between app runs

st.title(":money_with_wings: Forecast returns")
st.write("Using Monte Carlo simulation to forecast returns!")


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

# Import the functions from the funcs/financial_analysis.py file
from funcs.financial_analysis import calculate_volatility, calculate_variance, calculate_beta, calculate_sharpe_ratio, calculate_returns, plot_returns

# Define all the dataframes needed for the app:
# Create a dataframe that contains the closing prices of the FAANG stocks
# This dataframe will be used to calculate the portfolio returns:

sp500_df = Path("data/benchmark_df.csv")
curated_df = Path()

st.dataframe(sp500_df)
# Create a function that calculates the portfolio Sortino ratio:
@st.cache_data
def calculate_sortino_ratio(df_weighted_portfolio):

    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    st.session_state['daily_returns'] = daily_returns
    # Calculate the daily downside deviation
    daily_downside_deviation = daily_returns[daily_returns < 0].std()
    st.session_state['daily_downside_deviation'] = daily_downside_deviation 
    # Calculate the daily Sortino ratio
    sortino_ratio = daily_returns.mean() / daily_downside_deviation
    st.session_state['sortino_ratio'] = sortino_ratio   
    # Annualize the Sortino ratio
    annual_sortino_ratio = sortino_ratio * np.sqrt(252)
    st.session_state['annual_sortino_ratio'] = annual_sortino_ratio 
    # Return the annualized Sortino ratio
    return annual_sortino_ratio
    
    # # Create a function that calls all the above functions, and displays the results when prompted by streamlit buttons:
@st.cache_data
def display_results(df_weighted_portfolio):
    # Calculate the portfolio returns
    cumulative_returns = calculate_returns(df_weighted_portfolio)
    st.session_state['cumulative_returns'] = cumulative_returns
    # Calculate the portfolio volatility
    annual_std = calculate_volatility(df_weighted_portfolio)
    st.session_state['annual_std'] = annual_std
    # Calculate the portfolio variance
    annual_var = calculate_variance(df_weighted_portfolio)
    st.session_state['annual_var'] = annual_var
    # Calculate the portfolio beta
    beta = calculate_beta(df_weighted_portfolio)
    st.session_state['beta'] = beta
    # Calculate the portfolio Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(df_weighted_portfolio)
    st.session_state['sharpe_ratio'] = sharpe_ratio
    # Calculate the portfolio Sortino ratio
    sortino_ratio = calculate_sortino_ratio(df_weighted_portfolio)
    st.session_state['sortino_ratio'] = sortino_ratio
        
    # Display the cumulative returns
    plot_returns(cumulative_returns)
    # Display the portfolio volatility
    st.write(f'Portfolio Volatility: {annual_std}')
    # Display the portfolio variance
    st.write(f'Portfolio Variance: {annual_var}')
    # Display the portfolio beta
    st.write(f'Portfolio Beta: {beta}')
    # Display the portfolio Sharpe ratio
    st.write(f'Portfolio Sharpe Ratio: {sharpe_ratio}')
    # Display the portfolio Sortino ratio
    st.write(f'Portfolio Sortino Ratio: {sortino_ratio}')

# Create numerous buttons, that when pressed, will display the results of the above functions:
if st.button('Calculate Returns'):
    st.session_state['portfolio_data'] = pd.read_csv('data/portfolio_data.csv')
    df_weighted_portfolio = st.session_state['portfolio_data']
    display_results(df_weighted_portfolio)
if st.button('Calculate Volatility'):
    st.write(f'Portfolio Volatility: {st.session_state["annual_std"]}')
if st.button('Calculate Variance'):
    st.write(f'Portfolio Variance: {st.session_state["annual_var"]}')
if st.button('Calculate Beta'):
    st.write(f'Portfolio Beta: {st.session_state["beta"]}')
if st.button('Calculate Sharpe Ratio'):
    st.write(f'Portfolio Sharpe Ratio: {st.session_state["sharpe_ratio"]}')
if st.button('Calculate Sortino Ratio'):
    st.write(f'Portfolio Sortino Ratio: {st.session_state["sortino_ratio"]}')
if st.button(f'Plot Returns'):
    plot_returns(st.session_state['cumulative_returns'])