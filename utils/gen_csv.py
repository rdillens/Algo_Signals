import pandas as pd
import yfinance as yf
import csv
import requests
import numpy as np
from pathlib import Path
import sqlalchemy as sql

# Pulling S&P Data from wiki and outputing html
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Read html
sp500_html = pd.read_html(url)

# Obtain first table
sp500_html = sp500_html[0]

# Create dataframe
sp500_df = pd.DataFrame(sp500_html)
# sp500_df.head()

sp500_all_sectors_df = pd.DataFrame(
    columns=['GICS Sector', 'Symbol'],
    data=sp500_df
    )
# sp500_all_sectors_df.head()

# Delete index
sp500_df_wo_index = sp500_all_sectors_df.set_index("Symbol")
# sp500_df_wo_index

# isolate symbols in order to pass list to yfinance to get market cap info
sp500_all_symbols = sp500_all_sectors_df['Symbol'].values.tolist()
# sp500_all_symbols

# one issue with how the wikipedia symbols come is that they come with a "." instead of a "-"
# yahoo finance needs to have the "-" in order to pull the data
# this step might need to go in front of the part where we break the sectors out individually
stocks = []

for stock_ticker in sp500_all_symbols:
    ticker = stock_ticker.replace(".","-")
    stocks.append(ticker)

# print(stocks)

def market_cap(stocks):

    market_cap = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        market_cap[stock] = ticker.info['marketCap']

      
    # we want to return a sorted Pandas DataFrame based on market cap
    # since the columns will originally be the ticker we us ".T" to transpose the table
    # then we use .sort_values to sort by the "first column" [0] and sort in decending order
    # on average this takes 2400 seconds (37 minutes) to run for entire SP500
    return pd.DataFrame(market_cap, index=[0]).T.sort_values(by=[0], ascending=False)

market_cap_df = market_cap(stocks)
# market_cap_df

# rename the column and index to be merged

market_cap_df.columns = ['Market_Cap']
market_cap_df.index.names = ['Symbol']

# merge sp500_df_wo_index and market_cap_df to create 1 complete data frame to be sliced for analysis
stock_industry_marketcap = pd.merge(sp500_df_wo_index, market_cap_df, left_index=True, right_index=True)

stock_industry_marketcap.sort_values(by=['GICS Sector', 'Market_Cap'], ascending=False, inplace=True)

# save new dataframe to csv to be used in other code
stock_industry_marketcap.to_csv("stock_industry_marketcap.csv")