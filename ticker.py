import yfinance as yf
import pandas as pd
import streamlit as st

st.title('Portfolio Closing Prices')

# Define ticker options
tickers = ['AAPL', 'MSFT', 'BTC-USD']

# Define timeframe options
timeframes = {
    '1 year': '1y',
    '3 years': '3y',
    '5 years': '5y'
}

# Allow user to select ticker and timeframe
ticker = st.selectbox('Select Ticker', tickers)
timeframe = st.selectbox('Select Timeframe', list(timeframes.keys()))


# Get closing prices using Yahoo Finance API
df_portfolio = yf.download(ticker, period=timeframes[timeframe], group_by='ticker')
st.dataframe(df_portfolio)
# Review the first 5 rows of the Yahoo Finance DataFrame
st.write(df_portfolio.head())

# Reorganize the DataFrame
# Separate ticker data
df_ticker = df_portfolio[ticker].drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
df_ticker.columns = ['Close']

# Review the first 5 rows of the ticker DataFrame
st.write(df_ticker.head())

# Plot the closing prices
st.line_chart(df_ticker)
