import streamlit as st
# Import the required libraries and dependencies
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation

%matplotlib inline


# Load the environment variables from the .env file
#by calling the load_dotenv function
load_dotenv()
