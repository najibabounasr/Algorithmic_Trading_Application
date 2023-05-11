# Import the required libraries and dependencies:
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import base64
from dotenv import load_dotenv

# Create a benchamrk returns dataframe, based off the S&P 500:
@st.cache_data
def get_benchmark_returns():

    # Import the required libraries and dependencies
    import streamlit as st
    import pandas as pd
    import numpy as np
    import requests
    from dotenv import load_dotenv
    import alpaca_trade_api as alpaca
    import os

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

    

    # Use the alpaca API to pull in the benchmark data (S&P 500)
    benchmark_ticker = 'SPY'
    benchmark_timeframe = '1D'
    benchmark_start_date = st.session_state['selected_start_date']
    benchmark_end_date = st.session_state['end_date']

    # Set the start and end dates of the benchmark data
    start_date = pd.Timestamp('2016-01-01', tz='America/New_York').isoformat()
    end_date = pd.Timestamp('2021-01-01', tz='America/New_York').isoformat()
    # Get the benchmark data from the Alpaca API
    benchmark_df = alpaca.get_bars(
        benchmark_ticker,
        benchmark_timeframe,
        start = start_date,
        end = end_date
    ).df

    benchmark_df['pct_change'] = benchmark_df['close'].pct_change()
    st.session_state['benchmark_df'] = benchmark_df # Save the benchmark_df to the session state
    # # Create a benchmark_returns_df DataFrame:
    benchmark_returns_df = benchmark_df['pct_change']
    #  Drop the NaN values from the DataFrame
    # benchmark_returns_df = benchmark_returns_df.dropna()
    
    # Create a .csv file called 'benchmark_df.csv' in the data folder:
    benchmark_df.to_csv('data/benchmark_df.csv')
    
    # Create a .csv file called 'benchmark_returns.csv' in the data folder:
    benchmark_returns_df.to_csv('data/benchmark_returns.csv')

    # Return the dataframes
    return benchmark_df, benchmark_returns_df


# Create a function that calculates the portfolio volatility:
@st.cache_data
def calculate_volatility(df_weighted_portfolio):
    #Import the required libraries and dependencies
    import streamlit as st
    import pandas as pd
    import numpy as np

    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the daily standard deviation of returns
    daily_std = daily_returns.std()
    # Calculate the annualized standard deviation of returns
    annual_std = daily_std * np.sqrt(252)
    # Return the annualized standard deviation
    return annual_std
    
    
# Create a function that calculates the portfolio variance:
@st.cache_data
def calculate_variance(df_weighted_portfolio):
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the daily variance of returns
    daily_var = daily_returns.var()
    # Calculate the annualized variance of returns
    annual_var = daily_var * 252
    # Return the annualized variance
    return annual_var


# Create a function that calculates the portfolio beta:
@st.cache_data
def calculate_beta(df_weighted_portfolio):
    # Load benchmark returns from CSV
    benchmark_returns = pd.read_csv('data/benchmark_returns.csv', index_col=0, parse_dates=True)
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    st.session_state['daily_returns'] = daily_returns
    # Calculate the daily covariance of returns
    daily_covariance = daily_returns.cov(benchmark_returns['SPY'])
    st.session_state['daily_covariance'] = daily_covariance
    # Calculate the beta
    beta = daily_covariance / benchmark_returns['SPY'].var()
    st.session_state['beta'] = beta
    # Return the beta
    return beta
    
# Create a function that calculates the portfolio Sharpe ratio:
@st.cache_data
def calculate_sharpe_ratio(df_weighted_portfolio):
    # import the required libraries and dependencies
    import streamlit as st
    import pandas as pd
    import numpy as np
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the daily Sharpe ratio
    sharpe_ratio = daily_returns.mean() / daily_returns.std()
    # Annualize the Sharpe ratio
    annual_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    # Return the annualized Sharpe ratio
    return annual_sharpe_ratio
    
# Create a function that calculates the portfolio Sortino ratio:
@st.cache_data
def calculate_sortino_ratio(df_weighted_portfolio):
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the daily downside deviation
    daily_downside_deviation = daily_returns[daily_returns < 0].std()
    # Calculate the daily Sortino ratio
    sortino_ratio = daily_returns.mean() / daily_downside_deviation
    # Annualize the Sortino ratio
    annual_sortino_ratio = sortino_ratio * np.sqrt(252)
    # Return the annualized Sortino ratio
    return annual_sortino_ratio


# Create a function that calculates the portfolio returns:
@st.cache_data
def calculate_returns(df_weighted_portfolio):
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the cumulative returns of the portfolio
    cumulative_returns = (1 + daily_returns).cumprod()
    # Return the cumulative returns
    return cumulative_returns
    
    
# Create a function that plots the cumulative returns:
@st.cache_data
def plot_returns(cumulative_returns):
    #Import the required libraries and dependencies
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    # Plot the cumulative returns
    fig = px.line(
        x=cumulative_returns.index,
        y=cumulative_returns,
        title='Portfolio Cumulative Returns'
    )

    # Set the y-axis label
    fig.update_yaxes(title_text='Returns')
    # Display the plot
    st.plotly_chart(fig)

    
# Create a function that calculates the portfolio volatility:
@st.cache_data
def calculate_volatility(df_weighted_portfolio):
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the daily standard deviation of returns
    daily_std = daily_returns.std()
    # Calculate the annualized standard deviation of returns
    annual_std = daily_std * np.sqrt(252)
    # Return the annualized standard deviation
    return annual_std
    
    
# Create a function that calculates the portfolio variance:
@st.cache_data
def calculate_variance(df_weighted_portfolio):
    # Calculate the daily returns of the portfolio
    daily_returns = df_weighted_portfolio.sum(axis=1)
    # Calculate the daily variance of returns
    daily_var = daily_returns.var()
# Calculate the annualized variance of returns
    annual_var = daily_var * 252
    # Return the annualized variance
    return annual_var
    
