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
from funcs.MCForecastTools import MCSimulation


st.title(":money_with_wings: Forecast Returns")
st.subheader("Using Monte Carlo simulation to forecast returns!")
st.warning(":warning: Past Results are not Indicative of Future Returns :warning:")
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

tickers = st.session_state['tickers']
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

# Display the dataframe in the Streamlit app
if sum(weights.values()) > 1:
        st.sidebar.warning("Your weights add up to more than 1, please adjust your weights in the sidebar.")
        st.sidebar.dataframe(df_weights)
elif sum(weights.values()) < 1:
        st.sidebar.warning("Your weights add up to less than 1, please adjust your weights in the sidebar.")
        st.sidebar.dataframe(df_weights)
else:
        st.sidebar.success("Your weights add up to 1, you may proceed in the sidebar.")
        st.sidebar.dataframe(df_weights)
     

weights = st.session_state['weights']
# Import the closing prices dataframe:
closing_prices_df = pd.read_csv(
    Path("data/grouped_dfs/closing_prices_df.csv"), index_col=[0], infer_datetime_format=True, parse_dates=True
)
weights = {}
for ticker in tickers:
    weights[ticker] = df_weights[ticker][0]

# create a list out of the df_weights
weights_list = list(weights.values())



st.subheader("Weights:")

# Display the dataframe in the Streamlit app
if sum(weights.values()) > 1:
        st.warning("Your weights add up to more than 1, please adjust your weights.")
        
elif sum(weights.values()) < 1:
        st.warning("Your weights add up to less than 1, please adjust your weights.")
        
else:
        st.success("Your weights add up to 1, you may proceed.")
       
st.write(df_weights)


# Prompt the user for how much they intend on investing in their portfolio: ($)
st.sidebar.header("Portfolio Investment")
investment = st.sidebar.number_input("How much do you intend on investing in your portfolio? ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

initial_investment = pd.DataFrame()

# Fihd out how much the value of the initial investment is:
for ticker in tickers:
    initial_investment[f"{ticker}_initial_investment"] = investment * df_weights[ticker]
st.subheader("Your initial investments:")
st.dataframe(initial_investment)
st.session_state['initial_investment'] = initial_investment

num_simulation = st.slider("Enter the number of simulation to run:", 0, 500)

st.warning("NOTE: Running the Monte Carlo simulation may take a few minutes. More simulations may require more time.")
num_trading_days = st.slider("Enter the number of years to forecast:", 0, 30)




# Create a Dataframe that holds the prices and returns
price_returns_df = pd.DataFrame(closing_prices_df)
for ticker in tickers:
    # Generate column label with additional ticker label
    col_label = f"{ticker}_returns"
    
    # Perform your desired operation on the column with the updated label
    price_returns_df[col_label] = closing_prices_df[f"{ticker}_close"].pct_change()

price_returns_df = price_returns_df.dropna()
price_returns_df.to_csv("data/grouped_dfs/price_returns_df.csv")



# Create a dataframe that holds the returns of each asset
returns_df = pd.DataFrame()
for ticker in tickers:
    # Generate column label with additional ticker label
    col_label = f"{ticker}_returns"
    
    # Perform your desired operation on the column with the updated label
    returns_df[col_label] = closing_prices_df[f"{ticker}_close"].pct_change()

returns_df = returns_df.dropna()
returns_df.to_csv("data/grouped_dfs/returns_df.csv")


tickers = st.session_state['tickers']
timeframe = st.session_state['timeframe']
start_date = st.session_state['start_date']
end_date = st.session_state['end_date']


### START
price_df = alpaca.get_bars(
    tickers,
    timeframe,
    start=start_date,
    end=end_date,
).df

opts = st.multiselect("Select which variables you remove from the dataframe",("open", "high", "low", "volume", "trade_count", "vwap") )
st.markdown(""" NOTE: Only the 'close' variable is required for the forecast to run.
                    For accurate results, remove the other features. 

    Too many variables may skew the forecast,
    and also keep in mind that your options 
    here will be saved for the rest of the app session.
    If you want to reset your options, please refresh the page.
    Which variables you wish to include will affect the algorithm's performance in the next page, aswell.
        
            """)


price_df = price_df.drop(columns=opts)

# Check if tickers is not empty and contains valid ticker symbols
if not tickers:
    st.write("Error: tickers list is empty!")
    raise ValueError("tickers list is empty")

# Check for missing values in price_df
if price_df.isna().any().any():
    st.write("Warning: price_df contains missing values!")
    price_df = price_df.interpolate()

# Create an empty dictionary to store dataframes
dfs = {}

# Separate ticker data and store in dictionary
for ticker in tickers:
    df = price_df[price_df['symbol']==ticker].drop('symbol', axis=1)
    dfs[ticker] = df

# Concatenate the ticker DataFrames
df_tickers = pd.concat(dfs, axis=1, keys=tickers)

# Concatenate the DataFrames
df_assets = df_tickers.dropna().copy()

# Create MultiIndex for column labels
mi = pd.MultiIndex.from_tuples([(ticker, col) for ticker in tickers for col in df_assets.columns.levels[1]])

# Set MultiIndex as column labels
df_assets.columns = mi

df_pct_change = df_assets.pct_change().dropna()

st.dataframe(df_assets)
# st.warning("Daily Returns or 'pct_change' data is generated used for forecasting.")
# opt_64 = st.selectbox("Select Data to Be Used", ("Price Data", "Pct Change Data"))
# if opt_64 == "Price Data":
#     st.dataframe(df_assets)
# elif opt_64 == "Pct Change Data":
#     st.dataframe(df_pct_change)
#     df_assets = df_pct_change
# else:
#     st.write("Please Select an Option :wink:")





# Fihd out how much the value of the initial investment is:
for ticker in tickers:
    initial_investment[f"{ticker}_initial_investment"] = investment * df_weights[ticker]
st.success("Your initial investments:")
st.dataframe(initial_investment)
st.session_state['initial_investment'] = initial_investment

### END
if st.button("Run Monte Carlo Simulation"):
    # Configure the Monte Carlo simulation to forecast 30 years cumulative returns
    # Run 500 samples.
    MC_thirty_year = MCSimulation(
        portfolio_data = df_assets,
        weights = weights_list,
        num_simulation = num_simulation,
        num_trading_days = num_trading_days
    )

    MC_thirty_year.calc_cumulative_return()

    # Visualize the 30-year Monte Carlo simulation by creating an
    # overlay line plot
    MC_sim_line_plot = MC_thirty_year.plot_simulation()

    st.write(MC_sim_line_plot)
    # Save the plot for future usage


    # Visualize the probability distribution of the 30-year Monte Carlo simulation 
    # by plotting a histogram
    # MC_sim_dist_plot = MC_thirty_year.plot_distribution()

    # Review the simulation input data
    MC_thirty_year.portfolio_data.head()

    MC_summary_statistics = MC_thirty_year.summarize_cumulative_return()


    MC_sim_line_plot.get_figure().savefig("data/forecasts/MC_ten_year_sim_plot.png", bbox_inches="tight")
    # MC_sim_dist_plot.get_figure().savefig("data/forecasts/MC_ten_year_dist_plot.png", bbox_inches="tight")


option = st.selectbox("Options", ("Line Plot", "Distribution Plot", "Summary Statistics"))

if option == "Line Plot":
    # Configure the Monte Carlo simulation to forecast 30 years cumulative returns
    # Run 500 samples.
    MC_thirty_year = MCSimulation(
        portfolio_data = df_assets,
        weights = weights_list,
        num_simulation = num_simulation,
        num_trading_days = num_trading_days
    )

    MC_thirty_year.calc_cumulative_return()

    # Visualize the 30-year Monte Carlo simulation by creating an
    # overlay line plot
    MC_sim_line_plot = MC_thirty_year.plot_simulation()

    # Review the simulation input data
    MC_thirty_year.portfolio_data.head()

    MC_summary_statistics = MC_thirty_year.summarize_cumulative_return()

    MC_sim_line_plot.get_figure().savefig("data/forecasts/MC_ten_year_sim_plot.png", bbox_inches="tight")
    st.image("data/forecasts/MC_ten_year_sim_plot.png")
elif option == "Summary Statistics":
    # Configure the Monte Carlo simulation to forecast 30 years cumulative returns
    # Run 500 samples.
    MC_thirty_year = MCSimulation(
        portfolio_data = df_assets,
        weights = weights_list,
        num_simulation = num_simulation,
        num_trading_days = num_trading_days
    )

    MC_thirty_year.calc_cumulative_return()

    MC_summary_statistics = MC_thirty_year.summarize_cumulative_return()

    # Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes for the current stock/bond portfolio
    ci_lower_thirty_cumulative_return = MC_summary_statistics[8] * initial_investment
    ci_upper_thirty_cumulative_return = MC_summary_statistics[9] * initial_investment

    st.write(initial_investment)
    # Print the result of your calculations

    for ticker in tickers:
        investment = initial_investment[f"{ticker}_initial_investment"]
        investment = float(round(investment))
        low = float(round(ci_lower_thirty_cumulative_return[f'{ticker}_initial_investment']))
        upper = float(round(ci_upper_thirty_cumulative_return[f'{ticker}_initial_investment']))
        mean = investment * MC_summary_statistics[1]
        growth = mean - investment
        diff = (upper - low) 
        pct_change = (growth / investment) * 100
        future_value = investment * (1 + (pct_change/100))
        st.markdown(f""" {ticker} 
                
        

            There is a 95% chance that an initial investment of: 
        ${investment}
        in {ticker} will result in returns within the range of:
        ${low} and ${upper}, within the specified timeframe. 

        This would mean that, with a range between: ${low} and
        ${upper}.

        This would mean a difference of ${int(diff)} between the
        lower and upper bounds.

        The investment is forecast to be worth: 
        ${int(mean)} by the end of the simulation period. 
        This would mean that the investment would have
        grown by ${int(growth)} over the period.
        This is a change of {pct_change}%.


        {ticker} Results:
        - initial investment = ${investment}
        - mean = ${mean}
        - 95% confidence interval = ${low} - ${upper}
        - 95% lower confidence interval = ${low}
        - 95% upper confidence interval = ${upper}
        - 95% confidence interval spread = ${int(diff)}
        - growth = ${growth}
        - growth % = {int(pct_change)}%

                    """)
elif option == "Distribution Plot":
    # Configure the Monte Carlo simulation to forecast 30 years cumulative returns
    # Run 500 samples.
    MC_thirty_year = MCSimulation(
        portfolio_data = df_assets,
        weights = weights_list,
        num_simulation = num_simulation,
        num_trading_days = num_trading_days
    )

    MC_thirty_year.calc_cumulative_return()

    # Visualize the 30-year Monte Carlo simulation by creating an
    # overlay line plot
    MC_sim_dist_plot = MC_thirty_year.plot_distribution()


    # Review the simulation input data
    MC_thirty_year.portfolio_data.head()

    MC_summary_statistics = MC_thirty_year.summarize_cumulative_return()

    MC_sim_dist_plot.get_figure().savefig("data/forecasts/MC_ten_year_dist_plot.png", bbox_inches="tight")

    st.image("data/forecasts/MC_ten_year_dist_plot.png")


