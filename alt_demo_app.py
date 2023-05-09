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


# Set the page config
st.set_page_config(
    page_title="Algorithmic Trading App",
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
    selected_symbols = st.multiselect("Select assets",options=symbols,default=[])

    # Create select box for selecting timeframe
    start_date_options = ['1 year', '3 years', '5 years']
    selected_start_date = st.selectbox('Select the startdate (how many years back?)', start_date_options)
    timeframe_options = ['Day', 'Week', 'Month']
    selected_timeframe = st.selectbox('Select timeframe', timeframe_options)

    @st.cache_data
    def save_inputs(selected_symbols, selected_timeframe, selected_start_date):
    
        # Filter assets based on search bar input
        filtered_assets = [asset for asset in assets if asset.symbol in selected_symbols]
        st.write('These are your selected assets, and their accompanied symbols:')
        for i, asset in enumerate(filtered_assets):
            st.write(f"{i+1}. {asset.name} ({asset.symbol})")

        # Create a list of symbols from filtered assets
        tickers = [asset.symbol for asset in filtered_assets]
        st.write(f"these are the tickers: {tickers}")

        return tickers, selected_timeframe, selected_start_date
    
    tickers, selected_timeframe, selected_start_date = save_inputs(selected_symbols, selected_timeframe, selected_start_date)

    # Define function to get portfolio data
    @st.cache_data
    def get_portfolio_data(tickers, selected_timeframe, selected_start_date):
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
        if selected_start_date == '1 year':
            start_date = pd.Timestamp("2022-02-01", tz="America/New_York").isoformat()
        elif selected_start_date == '3 years':
            start_date = pd.Timestamp("2020-02-01", tz="America/New_York").isoformat()
        elif selected_start_date == '5 years':
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
        return df_portfolio, start_date, end_date, timeframe
        

    # Prepare an empty dictionary to store dataframes:
    dfs = {}


    # 
    # Create a Streamlit button for Generating Data :
    if st.button('Create Portfolio'):
        # Call the function to get the portfolio data
        df_portfolio, start_date, end_date, timeframe  = get_portfolio_data(tickers, selected_timeframe, selected_start_date)
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
    from dotenv import load_dotenv
    from pathlib import Path
    st.title(":money_with_wings: Forecast returns")
    st.write("Using Monte Carlo simulation to forecast returns!")


    # Load the environment variables from the .env file
    #by calling the load_dotenv function
    load_dotenv()



    # Set the variables for the Alpaca API and secret keys
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
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
        from datetime import datetime
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
    

    timeframe = '1D'
    start_date = pd.Timestamp('2018-01-01', tz='America/New_York').isoformat()
    end_date = pd.Timestamp('2022-12-31', tz='America/New_York').isoformat()
    
    # Create a df_benchmark_returns DataFrame that contains the benchmark returns (S&P 500):
    df_benchmark_returns = alpaca.get_bars(
        'SPY',
        timeframe,
        start=start_date,
        end=end_date
    ).df

    # Calculate the benchmark returns
    df_benchmark_returns = df_benchmark_returns['close'].pct_change().dropna()


    # Create a function that calculates the portfolio beta:
    @st.cache_data
    def calculate_beta(df_weighted_portfolio):
        # Calculate the daily returns of the portfolio
        daily_returns = df_weighted_portfolio.sum(axis=1)
        # Calculate the daily covariance of returns
        daily_covariance = daily_returns.cov(df_benchmark_returns)
        # Calculate the beta
        beta = daily_covariance / df_benchmark_returns.var()
        # Return the beta
        return beta
    
    # Create a function that calculates the portfolio Sharpe ratio:
    @st.cache_data
    def calculate_sharpe_ratio(df_weighted_portfolio):
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
    
    # Create a function that calls all the above functions, and displays the results when prompted by streamlit buttons:
    @st.cache_data
    def display_results(df_weighted_portfolio):
        # Calculate the portfolio returns
        cumulative_returns = calculate_returns(df_weighted_portfolio)
        # Calculate the portfolio volatility
        annual_std = calculate_volatility(df_weighted_portfolio)
        # Calculate the portfolio variance
        annual_var = calculate_variance(df_weighted_portfolio)
        # Calculate the portfolio beta
        beta = calculate_beta(df_weighted_portfolio)
        # Calculate the portfolio Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(df_weighted_portfolio)
        # Calculate the portfolio Sortino ratio
        sortino_ratio = calculate_sortino_ratio(df_weighted_portfolio)
        
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




