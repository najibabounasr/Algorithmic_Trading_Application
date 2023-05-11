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

     







# Create a database engine:
database_connection_string = 'sqlite:///'

# Create a connection to the engine called `conn`
engine = sqlalchemy.create_engine(
    database_connection_string,
    echo=True
)
# Create a connection to the engine called `conn`
conn = engine.connect()


# Initialize the session state for the app
# This will allow us to store data between app runs

st.title(" :chart: Technical Analysis :chart:")
st.subheader("Using Monte Carlo simulation to forecast returns!")
st.warning("Please use the sidebar to select the portfolio weights before proceeding!")
st.markdown("---")
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

sp500_df = pd.read_csv("data/benchmark_df.csv",
                        index_col = [0],
                        parse_dates = True,
                        infer_datetime_format = True,)
# remove the first two rows of the dataframe, as they are not needed:
sp500_df = sp500_df.iloc[10:, :]
# rename the index of the dataframe to 'timeframe':
sp500_df.index.name = 'timestamp'

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


# # Create a subheader, explaining that the dataframe below contains the closing prices of the SPY stocks:
# st.subheader("S&P 500 Dataframe")
# # Display the dataframe in the Streamlit app:
# st.dataframe(sp500_df)

# st.subheader("Portfolio Dataframe")
# # Display the dataframe in the Streamlit app:
# st.dataframe(curated_df)



st.markdown("""
    
    - Here, you may interact with your portfolio, and quickly access numerous financial metrics, such as the portfolio's volatility, variance, beta, and Sharpe ratio.

    - This can be utilized by investors to quickly analyze their portfolio, and make informed decisions on how to best allocate their capital.

    - rather than analyzing a singular asset, find out how adjusting your weights, and your portfolio spread might impact the financial metrics of your portfolio.

    - to learn more please visit:
    [Portfolio](https://www.investopedia.com/terms/p/portfolio.asp)
    [Volatility](https://www.investopedia.com/terms/v/volatility.asp)
    [Variance](https://www.investopedia.com/terms/v/variance.asp)
    [Beta](https://www.investopedia.com/terms/b/beta.asp)
    [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
    [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp)
    [Monte Carlo Simulation](https://www.investopedia.com/terms/m/montecarlosimulation.asp)

""") 
st.markdown("---")
# Prompt the user for how much they intend on investing in their portfolio: ($)
st.header("Portfolio Investment")
investment = st.number_input("How much do you intend on investing in your portfolio? ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

# INITIALIZATION -- GETTING dataframes ready!

tickers = st.session_state['tickers']
st.markdown("---")

# Start by Getting the weights:
# Prompt the user for the weights of each asset in their portfolio:
st.header("Portfolio Weights")
# Create a dictionary to hold the weights of each asset
weights = {}
# Create a for loop that will prompt the user for the weight of each asset
for ticker in tickers:
    tickers = st.session_state['tickers']
    weights[ticker] = st.number_input(f"What is the weight of {ticker} in your portfolio? (Please enter a value between 0 and 1)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    # add functionality, so that the weights of the assets add up to 1

    # Create a dataframe that holds the weights of each asset
    df_weights = pd.DataFrame(weights, index=[0])
    # Also Create a list that holds the weights of each asset:
    
    # Make sure that te dataframe is labeled correctly, with the asset ticker as the column name
    df_weights = df_weights.T
    df_weights.columns = ['weights']
    df_transposed_weights = df_weights
    df_weights = df_weights.transpose()
    weights = df_weights.to_dict('records')[0]



# Display the dataframe in the Streamlit app
if sum(weights.values()) > 1:
        st.sidebar.warning("Your weights add up to more than 1, please adjust your weights.")
        st.sidebar.dataframe(df_weights)
elif sum(weights.values()) < 1:
        st.sidebar.warning("Your weights add up to less than 1, please adjust your weights.")
        st.sidebar.dataframe(df_weights)
else:
        st.sidebar.success("Your weights add up to 1, you may proceed.")
        st.sidebar.dataframe(df_weights)

initial_investment = pd.DataFrame()

# Fihd out how much the value of the initial investment is:
for ticker in tickers:
    initial_investment[f"{ticker}_initial_investment"] = investment * df_weights[ticker]
st.success("Your initial investments:")
st.dataframe(initial_investment)

# Initialize a dataframe that will hold the calculated metrics:
# This dataframe will hold any metrics that are calculated using the closing prices of the assets in the portfolio, such as returns. 
updated_df = pd.DataFrame()

# initialize the location variables:
closing_prices_df = pd.DataFrame()



location = 3
# Create the closing prices df:
for ticker in tickers:
    df = pd.DataFrame()
    df[ticker] = curated_df.iloc[:, location]
    col_name = f"{ticker}_close"
    closing_prices_df.loc[:, col_name] = df[ticker]
    col_name = f"{ticker}_weighted_price"
    location = location + 7 



# Remove the first two rows of the closing_prices_df:
closing_prices_df = closing_prices_df.iloc[2:, :]
closing_prices_df.to_csv("data/grouped_dfs/closing_prices_df.csv")
# Comvert the values of the column to floats:
for ticker in tickers:
    closing_prices_df[f"{ticker}_close"] = closing_prices_df[f"{ticker}_close"].astype(float)

# Create a streamlit button, with mutliple options for the user to choose from:
# The user will be able to choose which metrics they would like to calculate, and the app will display the metrics in a dataframe.
# The user will also be able to choose which metrics they would like to display in a graph.


# (initial_investment.loc[:,f"{ticker}_initial_investment"].values[0]


daily_returns_df = pd.DataFrame()
cumulative_returns_df = pd.DataFrame()
closing_prices_df = pd.read_csv("data/grouped_dfs/closing_prices_df.csv", index_col=0)
# Create a for loop, that will calculate the daily returns using the closing prices df:
for ticker in tickers:
    df = pd.DataFrame()
    df[f"{ticker}_daily_returns"] = closing_prices_df[f"{ticker}_close"].pct_change()
    daily_returns_df = pd.concat([daily_returns_df, df], axis=1)
    # Add the daily_returns values to the updated_df dataframe:
    updated_df[f"{ticker}_daily_returns"] = daily_returns_df[f"{ticker}_daily_returns"]
    # Add a cumulative returns column to the dataframe:
    cumulative_returns_df[f"{ticker}_cumulative_returns"] = ((1) + daily_returns_df[f"{ticker}_daily_returns"]).cumprod()
    updated_df[f"{ticker}_cumulative_returns"] = ((1) + daily_returns_df[f"{ticker}_daily_returns"]).cumprod()
    # Apply weights to dataframe:
    # Add a weighted close column to the dataframe:
    weighted_close = pd.DataFrame()
    weighted_close[f"{ticker}_weighted_close"] = closing_prices_df[f"{ticker}_close"] * weights[ticker]
    updated_df[f"{ticker}_weighted_close"] = closing_prices_df[f"{ticker}_close"] * weights[ticker] 
    # Add a weighted returns column to the dataframe:
    weighted_returns = pd.DataFrame()
    weighted_returns[f"{ticker}_weighted_returns"] = daily_returns_df[f"{ticker}_daily_returns"] * weights[ticker]
    updated_df[f"{ticker}_weighted_returns"] = daily_returns_df[f"{ticker}_daily_returns"] * weights[ticker]
    # Add a weighted cumulative returns column to the dataframe:
    weighted_cumulative_returns = pd.DataFrame()
    weighted_cumulative_returns[f"{ticker}_weighted_cumulative_returns"] = ((1) + updated_df[f"{ticker}_weighted_returns"]).cumprod()
    updated_df[f"{ticker}_weighted_cumulative_returns"] = ((1) + updated_df[f"{ticker}_weighted_returns"]).cumprod()




# # Transpose dataframe
# cumulative_returns_df = cumulative_returns_df.T

# # Rename row label




weighted_df = pd.DataFrame()


# Create the weighted prices df:
for ticker in tickers:
    df = pd.DataFrame()
    weight_value = df_weights[ticker]
    df[ticker] = closing_prices_df[f"{ticker}_close"]
    col_name = f"{ticker}_weighted_price"
    # Apply the weights to the weighted_df:
    df_weights = pd.DataFrame(weights, index=[0])


# Calculate 
trading_days = 252

df = pd.read_csv(f"data/grouped_dfs/closing_prices_df.csv")

annualized_std = daily_returns_df.std() * np.sqrt(trading_days)
annualized_std.sort_values()
average_annual_return = daily_returns_df.mean() * trading_days 

sharpe_ratios = average_annual_return / annualized_std
sharpe_ratios.sort_values()

st.markdown("---")



spy_close = sp500_df['close']
spy_returns = spy_close.pct_change().dropna()
spy_cumulative_returns = (1 + spy_returns).cumprod() - 1


st.markdown("---")
st.markdown("## Summary Statistics")
st.markdown("""

    - The summary statistics section will allow the user to generate summary statistics for any asset in the portfolio.
    - The user will be able to select the asset they would like to generate summary statistics for from a dropdown menu.
    - If you are unfamiliar with the terminology used in this section, please spend some time reading the [Investopedia](https://www.investopedia.com/) articles linked below:
        - [Standard Deviation](https://www.investopedia.com/terms/s/standarddeviation.asp)
        - [Annualized Standard Deviation](https://www.investopedia.com/terms/a/annualized-standard-deviation.asp)
        - [Annualized Volatility](https://www.investopedia.com/terms/v/volatility.asp)
        - [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
        - [Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp)
        - [Beta](https://www.investopedia.com/terms/b/beta.asp)
        - [Alpha](https://www.investopedia.com/terms/a/alpha.asp)
        - [R-Squared](https://www.investopedia.com/terms/r/r-squared.asp)
        - [Treynor Ratio](https://www.investopedia.com/terms/t/treynorratio.asp)
        - [Information Ratio](https://www.investopedia.com/terms/i/informationratio.asp)


""")
for ticker in tickers:
    ticker = closing_prices_df[f"{ticker}_close"]
    options = tickers

option = st.selectbox("Generate Summary Statistics For:", options)

if st.button("Generate Summary Statistics:"):
    st.write(f"Summary Statistics for {option}:")
    df = pd.read_csv(f"data/individual_dfs/{option}_df.csv")
    # select only the 'close' 
    df_sliced = df['close']
    std = df_sliced.std()
    st.balloons()
    st.title(f"Summary Statistics for {option}:")
    st.write(df_sliced.describe())
    st.write(f"Standard Deviation: {std}")
    # Calculate the annualized standard deviation (252 trading days)
    annualized_std = std * np.sqrt(252)
    st.write(f"Annualized Standard Deviation: {annualized_std}")
    # Calculate the annualized volatility (252 trading days)
    annualized_volatility = annualized_std * np.sqrt(252)
    st.write(f" Annualized Volatility : {annualized_volatility}")
    # Calculate the Sharpe Ratio
    sharpe_ratio = (df_sliced.mean() * 252) / (annualized_std)
    st.write(f"Sharpe Ratio : {sharpe_ratio}")
    # Calculate the Sortino Ratio
    sortino_ratio = (df_sliced.mean() * 252) / (annualized_std)
    st.write(f"The asset in question also has a Sortino Ratio of: {sortino_ratio}")



st.subheader(f" {option} Prices")
st.dataframe(df)

if st.button("Compare Assets"):
    trading_days = 252
    st.write("Comparing Assets")
    df = pd.read_csv(f"data/grouped_dfs/closing_prices_df.csv")
    st.write(df.describe())
    annualized_std = daily_returns_df.std() * np.sqrt(trading_days)
    annualized_std.sort_values()
    average_annual_return = daily_returns_df.mean() * trading_days 
    st.write("Average Annual Returns:")
    st.write(average_annual_return.sort_values())
    sharpe_ratios = average_annual_return / annualized_std
    sharpe_ratios.sort_values()
    st.write("Sharpe Ratios")
    st.write(sharpe_ratios.sort_values())







st.markdown("---")


# drop the first row of the dataframe:
updated_df = updated_df.iloc[1:, :]
cumulative_returns_df = cumulative_returns_df.iloc[1:, :]

st.title(":pencil: View Dataframes")
# Define the options for the button
options = ["All Metrics", "S&P-500 (Benchmark)", "Portfolio Dataframe", "Closing Prices", "Daily Returns", "Cumulative Returns"]

# Create the selectbox
option = st.selectbox("Choose a Dataframe to display:", options)
# Create more butons


for ticker in tickers:
    if st.button(f"{ticker} Dataframe"):
        st.write(f"{ticker} Dataframe")
        df = pd.read_csv(f"data/individual_dfs/{ticker}_df.csv")
        st.dataframe(updated_df)

# remove the first row from the daily returns df:
daily_returns_df = daily_returns_df.iloc[1:, :]

# Show different content based on the selected option
if option == "S&P-500 (Benchmark)":
    # Display the dataframe in the Streamlit app:
    st.subheader("S&P 500 Dataframe")
    st.dataframe(sp500_df)
elif option == "Portfolio Dataframe":
    st.subheader("Portfolio Dataframe")
    # Display the dataframe in the Streamlit app:
    st.dataframe(curated_df)
elif option == "Closing Prices":
     st.subheader(f"Closing Prices:")
     st.dataframe(closing_prices_df)
elif option == "Weighted Data":
    st.write("Weighted Data:")
    st.dataframe(weighted_df)
elif option == "Daily Returns":
    st.subheader(f"Daily Returns:")
    st.dataframe(daily_returns_df)
elif option == "Cumulative Returns":
    st.subheader(f"Cumulative Returns:")
    st.dataframe(cumulative_returns_df)
elif option == "All Metrics":
    st.subheader(f"All Metrics:")
    st.dataframe(updated_df)
else:
    st.subheader("Portfolio Dataframe")
    st.dataframe(curated_df)




st.markdown("---")

# Read in the dataframes:
benchmark_df = pd.read_csv("data//benchmark_df.csv")


st.title(":chart_with_upwards_trend: View Plots")
# Define the options for the button
options = ["Daily Returns", "Cumulative Returns", "Standard Deviation [Volatility]", "Closing Prices", "Sharpe Ratio", "Probability Distribution"]
option = st.selectbox("Choose a Plot to display:", options)



st.set_option('deprecation.showPyplotGlobalUse', False)
# Show different content based on the selected option
if option == "Rolling 30-Day Returns":
    # Calculate the rolling 30-day returns:
    rolling_returns = closing_prices_df.rolling(window=30).sum()
    # Plot the rolling returns:
    ax = rolling_returns.plot(kind = 'line', figsize=(20, 10), title="Rolling 30-Day Returns", xlabel="Date", ylabel="Returns", fontsize=12, color="blue")
    for ticker in rolling_returns.columns.values:
        rolling_returns[ticker].plot(ax=ax, lw=2, label=ticker)
    ax.legend(loc="upper right")
    st.pyplot()
elif option == "Standard Deviation [Volatility]":
    df = closing_prices_df.pct_change().dropna()
    ax = df.plot(kind='box', figsize=(20, 10), title="Standard Deviation [Volatility]", xlabel="Date", ylabel="Returns", fontsize=12, color="blue")
    ax.legend(loc="upper right")
    st.pyplot()
elif option == "Cumulative Returns":
    # Plot the cumulative returns:
    ax = cumulative_returns_df.plot(figsize=(20, 10), title="Cumulative Returns", xlabel="Date", ylabel="Returns", fontsize=12, color="blue")
    for ticker in cumulative_returns_df.columns.values:
        cumulative_returns_df[ticker].plot(ax=ax, lw=2, label=ticker)
    ax.legend(loc="upper right")
    st.pyplot()
elif option == "Daily Returns":
    ax = daily_returns_df.plot(figsize=(20, 10), title="Daily Returns", xlabel="Date", ylabel="Returns", fontsize=12, color="blue")
    for ticker in daily_returns_df.columns.values:
        daily_returns_df[ticker].plot(ax=ax, lw=2, label=ticker)
    ax.legend(loc="upper right")
    st.pyplot()
elif option == "Closing Prices":
    ax = closing_prices_df.plot(figsize=(20, 10), title="Closing Prices", xlabel="Date", ylabel="Returns", fontsize=12, color="blue")
    for ticker in closing_prices_df.columns.values:
        closing_prices_df[ticker].plot(ax=ax, lw=2, label=ticker)
    ax.legend(loc="upper right")
    st.pyplot()
elif option == "Sharpe Ratio":
    # convert the list to a dataframe:
    sharpe_ratios = pd.DataFrame(sharpe_ratios)
    ax = sharpe_ratios.plot(kind="bar", figsize=(20, 10), title="Sharpe Ratios", xlabel="Date", ylabel="Returns", fontsize=10, color="blue")
    for ticker in sharpe_ratios.columns.values:
        sharpe_ratios[ticker].plot(ax=ax, lw=2, label=ticker)
    ax.legend(loc="upper right")
    st.pyplot()
elif option == "Probability Distribution":
    ax = daily_returns_df.plot.hist(title=' Probability Distribution')
    st.pyplot()
else:
    st.subheader("Portfolio Dataframe")
    # display an image of a graph in markdown:
    st.markdown(""""
    ![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")
    """)




st.session_state['weights'] = weights