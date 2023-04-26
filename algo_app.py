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


import os
import alpaca_trade_api as tradeapi
import pandas as pd

# set url for paper or live environment
# initialize API object with keys for paper or live account

# os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
# os.environ["APCA_API_BASE_URL"] = "https://api.alpaca.markets"

# api = tradeapi.REST(#Key ID, #Secret Key, api_version='v2')

count = 0
search = True

while search:

    if count < 1:
        # get most recent activities
        data = api.get_activities()
        # Turn the activities list into a dataframe for easier manipulation
        data = pd.DataFrame([activity._raw for activity in data])
        # get the last order id for pagination
        split_id = data.id.iloc[-1]

        trades = data

    else:
        data = api.get_activities(direction='desc', page_token=split_id)
        data = pd.DataFrame([activity._raw for activity in data])

        if data.empty:
            search = False

        else:
            split_id = data.id.iloc[-1]
            trades = trades.append(data)

    count += 1
    
# filter out partially filled orders
trades = trades[trades.order_status == 'filled']
trades = trades.reset_index(drop=True)
trades = trades.sort_index(ascending=False).reset_index(drop=True)

print(trades)

dfProfit = trades

# convert filled_at to date
trades['transaction_time'] = pd.to_datetime(trades['transaction_time'], format="%Y-%m-%d")

# remove time
trades['transaction_time'] = trades['transaction_time'].dt.strftime("%Y-%m-%d")

# sort first based on symbol, then type as per the list above, then submitted date
trades.sort_values(by=['symbol', 'transaction_time', 'type'], inplace=True, ascending=True)

# reset index
trades.reset_index(drop=True, inplace=True)
# add empty 'profit' column
dfProfit['profit'] = ''

totalProfit = 0.0
profitCnt   = 0
lossCnt     = 0
slCnt       = 0
ptCnt       = 0
trCnt       = 0
qty         = 0
profit      = 0
sign        = {'buy': -1, 'sell': 1, 'sell_short': 1}


for index, row in trades.iterrows():

    if index > 0:
        if trades['symbol'][index - 1] != trades['symbol'][index]:
            qty    = 0
            profit = 0

    side      = trades['side'][index]
    filledQty = int(trades['cum_qty'][index]) * sign[side]
    qty       = qty + filledQty
    price     = float(trades['price'][index])
    pl        = filledQty * price
    profit    = profit + pl

    if qty==0:
        # complete trade
        trCnt = trCnt + 1
        # put the profit in its column
        #dfProfit['profit'][index] = profit
        dfProfit.loc[index, 'profit'] = round(profit, 2)
        totalProfit = totalProfit + profit
        if profit >= 0:
            profitCnt = profitCnt + 1
            if trades['type'][index] == 'limit':
                ptCnt = ptCnt + 1
        else:
            lossCnt = lossCnt + 1
            if trades['type'][index] == 'stop_limit':
                slCnt = slCnt + 1
        profit = 0

print(dfProfit)
