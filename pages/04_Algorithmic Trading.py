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
import selenium
from selenium import webdriver
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initiate the model instance
logreg_model = LogisticRegression()
hv.extension('bokeh',logo=False)

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

st.title(" :chart: Algorithmic Trading Analysis :chart:")
st.subheader(" Using a binary classification model, to predict trade signals based on short and long-term simple moving average (SMA)")
st.warning("For now, you may only generate trade signal predictions for individual assets-- future updates will allow for more advanced functionality")
st.success("The page will continue to load as you select your desired parameters, but please be patient as the data is being pulled from the Alpaca API")
st.markdown("---")
# Load the environment variables from the .env file
#by calling the load_dotenv function
load_dotenv()

# List the selected tickers, so that the user can choose one to analyze:
asset = st.selectbox("Select Asset", tickers)

st.header(f"Analyzing {asset}:")
# Import the portfolio dataset into a Pandas Dataframe
ohlcv_df = pd.read_csv(
    Path(f"data/individual_dfs/{asset}_df.csv"), 
    index_col=[0], 
    infer_datetime_format=True, 
    parse_dates=True
)


# Review the DataFrame
ohlcv_df.head()

# Filter the date index and close columns
signals_df = ohlcv_df.loc[:, ["close"]]

# Use the pct_change function to generate  returns from close prices
signals_df["Actual Returns"] = signals_df["close"].pct_change()

# Drop all NaN values from the DataFrame
signals_df = signals_df.dropna()


st.markdown("""[Simple Moving Average (SMA)](https://www.investopedia.com/terms/s/sma.asp)
""")
short_window = st.slider("Select Short SMA",5, 20)
long_window = st.slider("Select Long SMA", 50, 200)

# Generate the fast and slow simple moving averages (4 and 100 days, respectively)
signals_df['SMA_Fast'] = signals_df['close'].rolling(window=short_window).mean()
signals_df['SMA_Slow'] = signals_df['close'].rolling(window=long_window).mean()

signals_df = signals_df.dropna()

# Initialize the new Signal column
signals_df['Signal'] = 0.0

# When Actual Returns are greater than or equal to 0, generate signal to buy stock long
signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1

# When Actual Returns are less than 0, generate signal to sell stock short
signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1

# Calculate the strategy returns and add them to the signals_df DataFrame
signals_df['Strategy Returns'] = signals_df['Actual Returns'] * signals_df['Signal'].shift()

# Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
X = signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna()


# Create the target set selecting the Signal column and assiging it to y
y = signals_df['Signal']

# Review the value counts
y.value_counts()

# Select the start of the training period
training_begin = X.index.min()

# Select the ending period for the training data with an offset of 3 months
training_end = X.index.min() + DateOffset(months=3)

# Generate the X_train and y_train DataFrames
X_train = X.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]

# Review the X_train DataFrame
X_train.head()

# Generate the X_test and y_test DataFrames
X_test = X.loc[training_end+DateOffset(hours=1):]
y_test = y.loc[training_end+DateOffset(hours=1):]

# Review the X_test DataFrame
X_train.head()

# Scale the features DataFrames

# Create a StandardScaler instance
scaler = StandardScaler()

# Apply the scaler model to fit the X-train data
X_scaler = scaler.fit(X_train)

# Transform the X_train and X_test DataFrames using the X_scaler
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# From SVM, instantiate SVC classifier model instance
svm_model = svm.SVC()
 
# Fit the model to the data using the training data
svm_model = svm_model.fit(X_train_scaled, y_train)
 
# Use the testing data to make the model predictions
svm_pred = svm_model.predict(X_test_scaled)

# Review the model's predicted values
svm_pred[:10]

svm_testing_report = classification_report(y_test, svm_pred)

# Print the classification report


# Create a new empty predictions DataFrame.

# Create a predictions DataFrame
predictions_df = pd.DataFrame(index=X_test.index)

# Add the SVM model predictions to the DataFrame
predictions_df['Predicted'] = svm_pred

# Add the actual returns to the DataFrame
predictions_df['Actual Returns'] = signals_df["Actual Returns"]

# Add the strategy returns to the DataFrame
predictions_df['Strategy Returns'] = (predictions_df["Actual Returns"] * predictions_df["Predicted"]
    )

























opt = st.selectbox("Review Input Data", ["Features","Target Set","Train Data", "Test Data","Scaled Train Data","Scaled Test Data","None"])

if opt == "Test Data":
    st.markdown(" [Testing Data?](//www.obviously.ai/post/the-difference-between-training-data-vs-test-data-in-machine-learning)")
    st.dataframe(X_test)
elif opt == "Train Data":
    st.markdown(" [Training Data?](//monkeylearn.com/blog/training-data/)")
    st.dataframe(X_train)
elif opt == "Features":
    st.markdown(" [Features?](//docs.continual.ai/feature-sets/)")
    st.dataframe(X)
elif opt == "Target Set":
    st.markdown(" [Target Set?](//www.datarobot.com/wiki/target/)")
    st.dataframe(y)
elif opt == "Scaled Train Data":
    st.markdown(" [Why Scale the Data?](//towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35)")
    st.dataframe(X_train_scaled)
elif opt == "Scaled Test Data":
    st.markdown(" [Why Scale the Data?](//towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35)")
    st.dataframe(X_test_scaled)

opt = st.selectbox("Review Output Data", ["Signal Values","Model Predictions","None"])


if opt == "Signal Values":
    st.write(signals_df["Signal"].value_counts())
    # Create a markdown link that links to google
    st.markdown("[What's a Trade Signal?](//www.investopedia.com/terms/t/trade-signal.asp)")
elif opt == "None":
    st.write(":eye: :lips: :eye:")
elif opt == "Model Predictions":
    st.dataframe(predictions_df)
else:
    st.write("")



opt = st.selectbox("Model Information", ["Model Accuracy", "Model Report","None"])

if opt == "Model Accuracy":
    st.write(f" The model accuracy is == {svm_model.score(X_test_scaled, y_test)}")
    st.markdown(" [What's Model Accuracy?](//developers.google.com/machine-learning/crash-course/classification/accuracy)")
elif opt == "Model Report":
    # Convert the string object to a pandas DataFrame
    svm_testing_report = pd.read_csv(StringIO(svm_testing_report))
    st.markdown(" [What's a Classification Report?](//www.statology.org/sklearn-classification-report/)")
    # Display the DataFrame using st.dataframe()
    st.dataframe(svm_testing_report)




opt = st.selectbox("Visualize Data", ["Strategy Returns", "Strategy Returns vs. Actual Returns","None"])


if opt == "Strategy Returns":
    # Plot Strategy Returns to examine performance
    ax = (1 + signals_df['Strategy Returns']).cumprod().plot()
    st.pyplot()
elif opt == "Strategy Returns vs. Actual Returns":
    axax = ((1 + predictions_df[["Actual Returns", "Strategy Returns"]]).cumprod().plot())
    st.pyplot()
elif opt == "None":
    st.markdown(""" :ear: :eye: :lips: :eye:
    :eye: :lips: :eye:
    :eye: :lips: :eye: :nose:
    :eye: :lips: :eye: 
    :eye: :lips: :eye:
    :eye: :lips: :eye: :ear:
    """)
else:
    st.write("")

st.markdown("---")

st.title("Improve Model Performance")
st.write("Based on the information provided above, use the paramaters of this section to improve the model performance.")
st.markdown("""
 - [How to Improve Model Performance?](//www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/)
 - [AdaBoost](//www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/#:~:text=AdaBoost%2C%20also%20called%20Adaptive%20Boosting,are%20also%20called%20Decision%20Stumps.)
 - [DecisionTreeClassifier](//www.mastersindatascience.org/learning/machine-learning-algorithms/decision-tree/#:~:text=A%20decision%20tree%20is%20a,that%20contains%20the%20desired%20categorization.)
 - [LogisticRegression](//www.ibm.com/topics/logistic-regression#:~:text=Resources-,What%20is%20logistic%20regression%3F,given%20dataset%20of%20independent%20variables.)
 - [RandomForestClassifier](//www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)
 - [SVC](//medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)
 - [None](//www.blank.org/)
""")

model_choice = st.selectbox("Select the Model to be Used", ["AdaBoost", "DecisionTreeClassifier","LogisticRegression","RandomForestClassifier","SVC","None"])

st.write("Signal Predictions:")
# Initiate the model instance

if model_choice == 'AdaBoost':
    model = AdaBoostClassifier()
elif model_choice == 'DecisionTreeClassifier':
    model = DecisionTreeClassifier()
elif model_choice == 'LogisticRegression':
    model = LogisticRegression()
elif model_choice == 'RandomForestClassifier':
    model = RandomForestClassifier()
elif model_choice == 'SVC':
    st.warning("NOTE: The SVC model is the standard model used in this application. Meaning performance will not change from the default model.")
    model = svm.SVC()
elif model_choice == 'None':
    st.write(" :wave: :wave: :wave: ")
else:
    st.write("")

# Fit the model using the training data
model = model.fit(X_train_scaled, y_train)

# Use the testing dataset to generate the predictions for the new model
pred = model.predict(X_test)

# Review the model's predicted values
pred[:10]



# Create a predictions DataFrame
predictions = pd.DataFrame(index=X_test.index)

# Add the SVM model predictions to the DataFrame
predictions['svm_pred'] = pred

# Add the actual returns to the DataFrame
predictions['actual_returns'] = signals_df['Actual Returns']

# Add the strategy returns to the DataFrame
predictions['strategy_returns'] =   predictions['svm_pred'] * predictions['actual_returns']

opt_3 = st.selectbox("Visualize Model Performance", ["Strategy Returns", "Strategy Returns vs. Actual Returns","None"])

if opt_3 == "Strategy Returns":
    # Plot Strategy Returns to examine performance
    ax = (1 + predictions['strategy_returns']).cumprod().hvplot()
    # st.pyplot()
    st.bokeh_chart(hv.render(ax, backend='bokeh'))
    if st.button("Save Plot"):
        hv.save(ax, 'data/plots/strategy_returns.png', backend='bokeh')
        # Save a .png image of the reconfigured algoritghm,
        st.success("Plot Saved in data/plots!")
        plt.show()
elif opt_3 == "Strategy Returns vs. Actual Returns":
    axax = ((1 + predictions[["actual_returns", "strategy_returns"]]).cumprod().hvplot())
    st.bokeh_chart(hv.render(axax, backend='bokeh'))
    if st.button("Save Plot"):
        hv.save(axax, 'data/plots/strategy_vs_actual_returns.png', backend='bokeh')
        # Save a .png image of the reconfigured algoritghm, 
        st.success("Plot Saved in data/plots")
        plt.show()
elif opt_3 == "None":
    st.markdown(""" :ear: :eye: :lips: :eye:
    :eye: :lips: :eye:
    :eye: :lips: :eye: :nose:
    :eye: :lips: :eye: 
    :eye: :lips: :eye:
    :eye: :lips: :eye: :ear:
    """)
else:
    st.write("")

st.markdown("---")

st.title("Trade Evaluation")

initial_investment = st.session_state['initial_investment']
weights = st.session_state['weights']
if initial_investment is None:
    raise ValueError("The initial investment is not set. Please set the initial investment in the sidebar. If the error persists, please try and set the initial investment in the previous pages.")
if weights is None:
    raise ValueError("The weights is not set. Please set the weights in the sidebar. If the error persists, please try and set the weights in the previous pages.")
###############UNRELATED:#########################################################################################################################################################################################################################################################
tickers = st.session_state['tickers']
st.success(" :warning: You may readjust the initial investment and weights of the assets in the sidebar. :smile: ")
st.sidebar.header("Portfolio Investment")
investment = st.sidebar.number_input("How much do you intend on investing in your portfolio? ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
st.sidebar.markdown("---")
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
        st.sidebar.warning("Your weights add up to more than 1, please adjust your weights.")
        st.sidebar.dataframe(df_weights)
elif sum(weights.values()) < 1:
        st.sidebar.warning("Your weights add up to less than 1, please adjust your weights.")
        st.sidebar.dataframe(df_weights)
else:
        st.sidebar.success("Your weights add up to 1, you may proceed.")
        st.sidebar.dataframe(df_weights)

# Fihd out how much the value of the initial investment is:
for ticker in tickers:
    initial_investment[f"{ticker}_initial_investment"] = investment * df_weights[ticker]
st.session_state['initial_investment'] = initial_investment
st.session_state['weights'] = weights
initial_investment = st.session_state['initial_investment']
weights = st.session_state['weights']
tickers = st.session_state['tickers']
################RELATED:#####################################################################################################################################################

investment = initial_investment[f"{asset}_initial_investment"]
investment = int(investment)

initial_capital =float(investment)

closing_df = pd.read_csv(f"data/individual_dfs/{asset}_df.csv", index_col="timestamp", infer_datetime_format=True, parse_dates=True)
closing_df = closing_df['close']
last_price = closing_df[-1]
st.session_state['last_price'] = last_price
share_size = initial_capital / last_price
st.session_state['share_size'] = share_size


hv.extension('bokeh',logo=False)

if st.button("Show Information"):
    st.write(f" Based on a closing price of {last_price} USD, and an initial investment of ${investment} in {asset}, you would be able to purchase {share_size} shares of {asset}.")
    st.write("Below, select the value of the ENtry/Exit price, which would be how many shares you would be buying or selling at each signal.")
    st.write("You can use the information provided above to chose a realistic value.")
    st.write("For whatever value you select, that amount of shares will be bought or sold at every signal.")
share_size = st.number_input("Entry/Exit Position", min_value=0.0, max_value=100000.0, value=last_price, step=0.01)

# Buy a 500 share position when the dual moving average crossover Signal equals 1 (SMA50 is greater than SMA100)
# Sell a 500 share position when the dual moving average crossover Signal equals 0 (SMA50 is less than SMA100)
signals_df['Position'] = share_size * signals_df['Signal']
signals_df['Entry/Exit'] = signals_df['Signal'].diff()
signals_df.tail(10)
predictions = predictions.rename(columns={
   	 "svm_pred": "Predicted"
})

# 2. 
predictions = predictions.rename(columns={
   	 "actual_returns": "Actual Returns"
})

# 3. 
predictions = predictions.rename(columns={
   	 "strategy_returns": "Strategy Returns"
})
# Create a 'close' columns in our two model dataframes:
predictions_df['close'] = signals_df['close']
predictions['close'] = signals_df['close']


predictions_df['Position'] = share_size * signals_df['Signal']

# Determine the points in time where a 500 share position is bought or sold
predictions_df['Entry/Exit Position'] = predictions_df['Predicted'].diff()

# Multiply the close price by the number of shares held, or the Position
predictions_df['Portfolio Holdings'] = signals_df['close'] * predictions_df['Predicted']

# Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
predictions_df['Portfolio Cash'] = initial_capital - (predictions_df['close'] * predictions_df['Entry/Exit Position']).cumsum()

# Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
predictions_df['Portfolio Total'] = predictions_df['Portfolio Cash'] + predictions_df['Portfolio Holdings']

# Calculate the portfolio daily returns
predictions_df['Portfolio Daily Returns'] = predictions_df['Portfolio Total'].pct_change()

# Calculate the portfolio cumulative returns
predictions_df['Portfolio Cumulative Returns'] = (1 + predictions_df['Portfolio Daily Returns']).cumprod() - 1


predictions['Position'] = share_size * signals_df['Signal']

# Determine the points in time where a 500 share position is bought or sold
predictions['Entry/Exit Position'] = predictions['Predicted'].diff()

# Multiply the close price by the number of shares held, or the Position
predictions['Portfolio Holdings'] = signals_df['close'] * predictions['Predicted']

# Subtract the amount of either the cost or proceeds of the trade from the initial capital invested
predictions['Portfolio Cash'] = initial_capital - (predictions['close'] * predictions['Entry/Exit Position']).cumsum()

# Calculate the total portfolio value by adding the portfolio cash to the portfolio holdings (or investments)
predictions['Portfolio Total'] = predictions['Portfolio Cash'] + predictions['Portfolio Holdings']

# Calculate the portfolio daily returns
predictions['Portfolio Daily Returns'] = predictions['Portfolio Total'].pct_change()

# Calculate the portfolio cumulative returns
predictions['Portfolio Cumulative Returns'] = (1 + predictions['Portfolio Daily Returns']).cumprod() - 1

# Create a list for the column name
columns = ['Backtest']

# Create a list holding the names of the new evaluation metrics
metrics = [
            'Annualized Return',
            'Cumulative Returns',
            'Annual Volatility',
            'Sharpe Ratio',
            ]

# Initialize the DataFrame with index set to the evaluation metrics and the column
base_portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)



# Create a list for the column name
columns = ['Backtest']

# Create a list holding the names of the new evaluation metrics
metrics = [
            'Annualized Return',
            'Cumulative Returns',
            'Annual Volatility',
            'Sharpe Ratio',
            ]

# Initialize the DataFrame with index set to the evaluation metrics and the column
tuned_portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)




# Calculate annualized return
base_portfolio_evaluation_df.loc['Annualized Return'] = (
predictions_df['Strategy Returns'].mean() * 252
)

# Calculate cumulative return
predictions_df['Portfolio Cumulative Returns'] = (1 + predictions_df['Strategy Returns']).cumprod() - 1
base_portfolio_evaluation_df.loc['Cumulative Returns'] = predictions_df['Portfolio Cumulative Returns'][-1]

# Calculate annual volatility
base_portfolio_evaluation_df.loc['Annual Volatility'] = (
    predictions_df['Strategy Returns'].std() * np.sqrt(252)
)

# Calculate Sharpe ratio
base_portfolio_evaluation_df.loc['Sharpe Ratio'] = (
    predictions_df['Strategy Returns'].mean() * 252) / (
    predictions_df['Strategy Returns'].std() * np.sqrt(252)
)



# Calculate annualized return
tuned_portfolio_evaluation_df.loc['Annualized Return'] = (
predictions['Strategy Returns'].mean() * 252
)

# Calculate cumulative return
predictions['Portfolio Cumulative Returns'] = (1 + predictions['Strategy Returns']).cumprod() - 1
tuned_portfolio_evaluation_df.loc['Cumulative Returns'] = predictions['Portfolio Cumulative Returns'][-1]

# Calculate annual volatility
tuned_portfolio_evaluation_df.loc['Annual Volatility'] = (
    predictions['Strategy Returns'].std() * np.sqrt(252)
)

# Calculate Sharpe ratio
tuned_portfolio_evaluation_df.loc['Sharpe Ratio'] = (
    predictions['Strategy Returns'].mean() * 252) / (
    predictions['Strategy Returns'].std() * np.sqrt(252)
)



opt_final = st.selectbox("Advanced Options", ["Predictions","Visualize Strategy Performance", "Backtesting","Performance Metrics", "Trade Evaluation"])

if opt_final == "Visualize Strategy Performance":
    st.write("Below, you will find the performance of the strategy, as well as the performance of the asset itself. The strategy is based on the dual moving average crossover, and the asset is the closing price of the asset.")
    st.write("In the graph, you can visualize the specific entry and exit points of the strategy (which are in green and red respectively) as well as the moving averages that were used to generate the signals.")
    st.write("The entry and exit positions appear ontop of the closing price of the asset, which is in light gray.")
    exit = signals_df[signals_df['Entry/Exit'] == -2.0]['close'].hvplot.scatter(
        color='red',
        marker='v',
        size=100,
        ylabel='Price in $',
        width=1000,
        height=400,
        legend = False
    )   

    entry = signals_df[signals_df['Entry/Exit'] == 2.0]['close'].hvplot.scatter(
        color='green',
        marker='^',
        size=100,
        ylabel='Price in $',
        width=1000,
        height=400,
        legend = False
    )

    # Visualize close price for the investment
    security_close = signals_df[['close']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400,
    )

    # Visualize moving averages
    moving_avgs = signals_df[['SMA_Fast', 'SMA_Slow']].hvplot(
    ylabel='Price in $',
        width=1000,
        height=400,
        legend = True
    )

    # Create the overlay plot
    entry_exit_plot = security_close * moving_avgs * entry * exit

    # Show the plot
    entry_exit_plot.opts(
        title="Portfolio - SMA_Short, SMA_Long, Entry and Exit Points"
    )
    # Display the plot in Streamlit
    st.bokeh_chart(hv.render(entry_exit_plot, backend='bokeh'))
    # save the plot as a png file
    hv.save(entry_exit_plot, 'data/plots/entry_exit_plot.png', backend='bokeh')
    st.success("The plot has been saved to the data/plots folder as entry_exit_plot.png!")
elif opt_final == "Performance Metrics":
    st.write(" - Tuned Model Performance")
    signals_df['Portfolio Holdings'] = signals_df['close'] * signals_df['Signal']
    signals_df['Entry/Exit'] = (signals_df['Entry/Exit'] / 2)
    st.dataframe(tuned_portfolio_evaluation_df)
    signals_df['Entry/Exit Position'] = (signals_df['Portfolio Holdings'] * (signals_df['Entry/Exit'])) / signals_df['close']
elif opt_final == "Trade Evaluation":
    st.write(" Signal Data")
    signals_df['Entry/Exit'] = (signals_df['Entry/Exit'] / 2)
    signals_df['Portfolio Holdings'] = signals_df['close'] * signals_df['Signal']
    signals_df = signals_df.dropna()
    signals_df['Entry/Exit Position'] = (signals_df['Portfolio Holdings'] * (signals_df['Entry/Exit'])) / signals_df['close']
    trade_evaluation_df = pd.DataFrame(
        columns=[
        'Stock',
            'Entry Date',
            'Exit Date',
            'Shares',
            'Entry Share Price',
            'Exit Share Price',
            'Entry Portfolio Holding',
            'Exit Portfolio Holding',
            'Profit/Loss']
        )
    # Initialize iterative variables
    entry_date = ""
    exit_date = ""
    entry_portfolio_holding = 0.0
    exit_portfolio_holding = 0.0
    share_size = 0
    entry_share_price = 0.0
    exit_share_price = 0.0
    # Loop through signal DataFrame
    # If `Entry/Exit` is 1, set entry trade metrics
    # Else if `Entry/Exit` is -1, set exit trade metrics and calculate profit
    # Then append the record to the trade evaluation DataFrame
    for index, row in signals_df.iterrows():
        if row['Entry/Exit'] == 1:
            entry_date = index
            entry_portfolio_holding = row['Portfolio Holdings']
            share_size = row['Entry/Exit Position']
            entry_share_price = row['close']

        elif row['Entry/Exit'] == -1:
            exit_date = index
            exit_portfolio_holding = abs(row['close'] * row['Entry/Exit Position'])
            exit_share_price = row['close']
            profit_loss = exit_portfolio_holding - entry_portfolio_holding
            trade_evaluation_df = trade_evaluation_df.append(
                {
                    'Stock': 'Emerging Markets',
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Shares': share_size,
                    'Entry Share Price': entry_share_price,
                    'Exit Share Price': exit_share_price,
                    'Entry Portfolio Holding': entry_portfolio_holding,
                    'Exit Portfolio Holding': exit_portfolio_holding,
                    'Profit/Loss': profit_loss
                },
                ignore_index=True)
    # trade_evaluation_df = trade_evaluation_df.set_index(pd.to_datetime(trade_evaluation_df['Entry Date'], infer_datetime_format=True)).drop(columns=['Entry Date'])
    trade_evaluation_df.drop('Stock',axis=1,inplace=True)
    trade_evaluation_df.drop('Shares',axis=1,inplace=True)
    st.subheader(f" {asset} Trade Evaluation")
    # save the dataframe to a csv file
    trade_evaluation_df.to_csv(f'data/algorithm/{asset}_trade_evaluation.csv')

    if st.button("Info"):
        st.write("A trade evaluation dataframe is typically used to keep track of the performance of trades made in a portfolio. The dataframe usually includes columns for the stock symbol, the entry and exit dates, the number of shares traded, the entry and exit share prices, the entry and exit portfolio holdings, and the profit or loss incurred from the trade.")
        st.write("In this case, we are only interested in the entry and exit dates, the entry and exit portfolio holdings, and the profit or loss incurred from the trade.")
        st.write("By looking at the trade evaluation dataframe, you can see the details of each trade made in the portfolio, including when the trade was entered and exited, the price at which the shares were bought and sold, and how much profit or loss was incurred. This information can be useful in analyzing the overall performance of the portfolio, identifying trends in trading behavior, and making informed decisions about future trades.")
    x = st.selectbox("Select the 'x' variable", ('Entry Date', 'Exit Date','Entry Share Price', 'Exit Share Price', 'Profit/Loss', 'Entry Portfolio Holding', 'Exit Portfolio Holding',))
    y = st.selectbox("Select the 'y' variable", ('Entry Date', 'Exit Date','Entry Share Price', 'Exit Share Price', 'Profit/Loss', 'Entry Portfolio Holding', 'Exit Portfolio Holding',))
    c = st.selectbox("Select the 'c' variable", ('Entry Share Price', 'Exit Share Price', 'Profit/Loss', 'Entry Portfolio Holding', 'Exit Portfolio Holding','None'))
    trade_evaluation_df['Entry Date'] = pd.to_datetime(trade_evaluation_df['Entry Date'], infer_datetime_format=True)
    trade_evaluation_df['Exit Date'] = pd.to_datetime(trade_evaluation_df['Exit Date'], infer_datetime_format=True)
    # remove the first row of the dataframe
    trade_evaluation_df = trade_evaluation_df.iloc[1:]
    # set te index column name to'Trade'
    trade_evaluation_df.index.name = 'Trade Number'
    st.write(trade_evaluation_df)
    if c == 'None':
        c = None
    ax = trade_evaluation_df.hvplot.scatter(
        x=x,
        y=y,
        c=c,
        colormap='viridis',
    )
    st.bokeh_chart(hv.render(ax, backend='bokeh'))
    name = st.text_input("Enter the name of the plot")
    if st.button("Save Visualization"):
        hv.save(ax, f"data/plots/{name}.png", fmt="png")
        st.success(f"Plot saved successfully to data/plots as {name}.png")
elif opt_final == "Backtesting":
    # Use a classification report to evaluate the model using the predictions and testing data
    report = classification_report(y_test, pred)
    report = pd.read_csv(StringIO(report))
    # Print the classification report
    st.write(report)
elif opt_final == "Predictions":
    st.dataframe(predictions.dropna())
    st.write(f"Using the model, we made a profit/loss of & {predictions_df['Strategy Returns'].sum():.2f}")
    st.write(f"A the end of the timeperiod, we are left with a total of (USD){predictions_df['Portfolio Holdings'].iloc[-1]:.2f}, which is a return of (USD){predictions_df['Portfolio Holdings'].iloc[-1] - predictions_df['Portfolio Holdings'].iloc[0]:.2f}")
else:
    st.write(":eye:")

st.title("Save Model")