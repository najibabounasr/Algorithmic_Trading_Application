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

st.title(":money_with_wings: Algorithmic Trading App")
st.write("Using Monte Carlo simulation to forecast returns!")   
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

