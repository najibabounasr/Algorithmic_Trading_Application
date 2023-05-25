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
from dotenv import load_dotenv
from funcs.financial_analysis import get_benchmark_returns

# Set the page config
st.set_page_config(
    page_title="MarketMaven",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(":money_with_wings: MarketMaven :money_with_wings:")



st.sidebar.markdown("""
# Market Maven

## Developed by *Najib Abou Nasr*

[My Linkedin](https://www.linkedin.com/in/najib-abou-nasr-a43520258/)

[My GitHub](https://github.com/najibabounasr)
""")
                    
st.markdown("""
![Image](https://www.master-of-finance.org/wp-content/uploads/2020/06/What-Does-a-Stockbroker-Do-1024x683.jpg)
""")


# Create a st.markdown header for the app, including a describption refernecing the alpaca API's link, posting an image of the BUC Fintech bootcamp logo, and with links to my github 'najibabounasr':
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    ## Background

This project aims to create a financial analysis app built with the Streamlit framework. The app allows users to select a portfolio of stocks, adjust the weights of each stock in the portfolio, and analyze various financial metrics of the portfolio. The code retrieves historical stock data from the Alpaca API and calculates various metrics such as volatility, variance, beta, and Sharpe ratio using functions from a separate financial analysis module. The app displays the portfolio data as well as the data for individual stocks in the portfolio in dataframes, and provides explanations of the financial metrics displayed. The app also uses a Monte Carlo simulation to forecast returns based on the historical data. Finally, the app prompts users to learn more about portfolios and financial metrics by providing links to relevant Investopedia articles. 

The application will allow users to quickly and easily interact with live stock data, and add stocks together to create customizable portfolios, with the added functionality of indicating the specific asset weights, and initial investment value for each asset in the portfolio. 

The application pulls is live stock data via. the Alpaca API, and creates a dataframe with the assets information over the specified timeframe. The API data is pulled in in real-time-- data as recent as one day prior may be pulled in, which is a limitation of utilizing the free Alpaca paper account. 

The project may be used to generate summary statistics, and retrieve historical trends data from numerous assets, and also allows for assets to be compared to one-another, meaning the application's functionality exceeds that of a simple portfolio curator, as the application is not meant to curate a portfolio, and is instead meant to inform trade decisions. 

The project focuses on utilizing stock data to train different algorithmic trading models, which will try and predict trade signal calls. The user may directly interact with both the short and long moving averages, aswell as the choose the specific sklearn model they wish to fit to the training data. By reviewing the summary statistics and numerous visualization options available, this project aims to provide users with an understanding of backtesting, model training, and model optimization for algorithmic trading. The main segment of the application, and the focus of the whole project is the 'Algoritmic Trading' Page.
    

    """,
    unsafe_allow_html=True,
)


st.markdown("""
        
[Alpaca API](https://alpaca.markets/docs/api-documentation/)
    

""")

# Initialize the session state for the app
# This will allow us to store data between app runs
# and share data between different app pages
# This is useful for storing the portfolio data
# and the portfolio returns data :
st.session_state['portfolio_data'] = None

# Set the variables for the Alpaca API and secret keys
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = 'https://paper-api.alpaca.markets'

