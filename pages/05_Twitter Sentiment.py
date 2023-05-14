# Import dependencies
import re
import tweepy
import pandas as pd
from alpaca.data import CryptoDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from io import StringIO
# Import a new classifier from SKLearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import hvplot.pandas 
import holoviews as hv



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

     


# Initialize the session state for the app
# This will allow us to store data between app runs

st.title(" :iphone: Twitter Sentiment :signal_strength:")
st.subheader(" Pulling in twitter sentiment, to further inform our trade decisions")
st.warning("Social Media sentiment can be a powerful tool in predicting the direction of a stock")
st.success("Alongside our financial analysis, we can use this to make more informed decisions, which will be based on the psychological forces that drive the market, rather than just the financial ones")
st.markdown("---")
# Load the environment variables from the .env file
#by calling the load_dotenv function
load_dotenv()


