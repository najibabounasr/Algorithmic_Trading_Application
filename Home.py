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

# Set the page config
st.set_page_config(
    page_title="Algorithmic Trading App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(":money_with_wings: Algorithmic Trading App")
st.subheader("Developed by Najib Abou Nasr")
st.write("Using Monte Carlo simulation to forecast returns!")   

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
    ## This is my algorithmic trading application, developed with the UC Berkeley Fintech Bootcamp
    
    * This application uses the Alpaca API to pull stock data and forecast returns using Monte Carlo simulation.
    * It also uses the Streamlit library to create the user interface.
    * Deployed on Heroku, the app is also open source and available on my [GitHub](https://github.com/najibabounasr/UCB_Project_2.git).
    * The application is for educational purposes only.
    
    
    [Alpaca API](https://alpaca.markets/docs/api-documentation/)
    
    [Linkedin](https://www.linkedin.com/in/najib-abou-nasr-a43520258/)

    ![Image](https://www.master-of-finance.org/wp-content/uploads/2020/06/What-Does-a-Stockbroker-Do-1024x683.jpg)
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

