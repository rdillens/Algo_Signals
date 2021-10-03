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

engine = sqlalchemy.create_engine(db_connection_string)


def main(ticker=None):
    if ticker is None:
        ticker = hm.input_ticker()

    product_df = hm.process_ticker_info(ticker)
    product_df.to_sql(ticker + "_Info", con=engine, if_exists='replace')

    candle_1d_df = hm.process_ticker_hist(ticker, interval='1d')
    candle_1d_df.to_sql(ticker + "_1_Day_Candles",
                        con=engine, if_exists='replace')

    candle_1m_df = hm.get_minute_candles(ticker)
    candle_1m_df.to_sql(ticker + "_1_Min_Candles",
                        con=engine, if_exists='replace')

    df = hm.add_trade_signals(candle_1m_df)

    return


if __name__ == "__main__":
    fire.Fire(main)
