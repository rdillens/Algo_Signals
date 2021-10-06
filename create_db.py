import questionary
import utils.helpful_functions as hf
# import shelve
import fire
# from utils.yfinance_ticker_cols import (scan_col_list)
import pandas as pd
import sqlalchemy
from pathlib import Path
import os
import MLNN

path = Path("./Resources")
if os.path.isdir(path):
    print('Yes')
else:
    os.makedirs(path)
    print('No')

db_connection_string = 'sqlite:///./Resources/product.db'

engine = sqlalchemy.create_engine(db_connection_string)


def main(ticker=None):
    if ticker is None:
        ticker = hf.input_ticker()

    product_df = hf.process_ticker_info(ticker)
    product_df.to_sql(ticker + "_Info", con=engine, if_exists='replace')

    candle_1d_df = hf.process_ticker_hist(ticker, interval='1d')
    candle_1d_df.to_sql(ticker + "_1_Day_Candles",
                        con=engine, if_exists='replace')

    candle_1m_df = hf.get_minute_candles(ticker)
    candle_1m_df.to_sql(ticker + "_1_Min_Candles",
                        con=engine, if_exists='replace')

    # df = hf.add_support_resistance(candle_1m_df, candle_1d_df)

    df = hf.add_trade_signals(candle_1m_df)
    df = hf.add_overlap_studies(df)
    df = hf.add_momentum_indicators(df)
    # df = hf.add_volume_indicators(df)
    df = hf.add_volatility_indicators(df)
    # df = hf.add_price_transform_functions(df)
    # df = hf.add_cycle_indicator_functions(df)
    # df = hf.add_statistic_functions(df)


    df = df.dropna()
    print(df.head())

    if(questionary.confirm("Save to database?").ask()):
        df.to_sql(ticker + '_Indicators', con=engine, if_exists='replace')
    




    # print(list(df.columns))
    # print(list(df.dtypes))
    # df = df.drop('Previous Date')
    print(80*'-')
    print(f"Shallow Neural Network: 1 input layer, 1 output layer.")
    print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
    print(80*'-')
    print(MLNN.mlnn(df))

    # print(80*'-')
    # print(f"Shallow Neural Network: 1 input layer, 1 output layer")
    # print(f"Binary clasifier identifying bullish or bearish signals, 0, 1, or -1.")
    # print(80*'-')
    # print(MLNN.mlnn(df, 2))

    print(80*'-')
    print(f"Deep Neural Network: 1 input layer, 1 hidden layer, 1 output layer.")
    print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
    print(80*'-')
    print(MLNN.dlnn(df))

    # print(80*'-')
    # print(f"Deep Neural Network: 1 input layer, 1 hidden layer, 1 output layer.")
    # print(f"Binary clasifier identifying bullish or bearish signals, 0, 1, or -1.")
    # print(80*'-')
    # print(MLNN.dlnn(df, 2))

    print(80*'-')
    print(f"Support Vector Classification")
    print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
    print(80*'-')
    print(MLNN.svc(df))


    return




if __name__ == "__main__":
    fire.Fire(main)
