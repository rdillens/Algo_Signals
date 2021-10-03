import questionary
import utils.helpful_methods as hm
import shelve
import fire
from utils.yfinance_ticker_cols import (scan_col_list)
import pandas as pd
import sqlalchemy
from pathlib import Path
import os

path = Path("./Resources")
if os.path.isdir(path):
    print('Yes')
else: 
    os.makedirs(path)
    print('No')

db_connection_string = 'sqlite:///./Resources/products.db'

engine  = sqlalchemy.create_engine(db_connection_string)

def main(user=None, market=None, exchange=None, product_type=None, ticker=None):
    if ticker is None: ticker = hm.get_ticker()

    product_df = hm.process_ticker_info(ticker)
    product_df.to_sql(ticker + "_Info", con=engine, if_exists='replace')

    candle_1d_df = hm.process_ticker_hist(ticker, interval='1d')
    candle_1d_df.to_sql(ticker + "_1_Day_Candles", con=engine, if_exists='replace')

    # candle_1m_df = hm.process_ticker_hist(ticker, interval='1m')
    candle_1m_df = hm.get_minute_candles(ticker)
    candle_1m_df.to_sql(ticker + "_1_Min_Candles", con=engine, if_exists='replace')
    # else:
    #     # Check for username in shelf, create new record if one does not exist
    #     user = hm.get_username(user)
    #     # Choose what market to scan: Stock, Crypto, or Forex
    #     market = hm.choose_market()
    #     # Choose the exchange for crypto, stocks use 'US', 
    #     exchange = hm.choose_exchange(market)
    #     # Choose type of product
    #     product_type = hm.choose_product_type(market, exchange)
    #     # Create list of products to scan
    #     product_df = hm.gen_product_df(market, exchange, product_type).sort_values().reset_index(drop=True)
    #     # Query with each product in dataframe to get current market info
    #     product_df = hm.get_market_info(product_df, market, exchange, product_type, engine)
    #     print(product_df.head(10))
    #     # print(len(product_df), type(product_df))
    #     # scan_product_df = product_df[scan_col_list]
    #     # print(scan_product_df.head())

    #     # product_df.to_sql("Stock_Company_Info", con=engine, if_exists='replace')

    return user



if __name__ == "__main__":
    user = fire.Fire(main)
    if user:
        print(f"Goodbye, {user}")