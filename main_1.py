from DLNN_1 import dataframe
# from MLNN import svc
import MLNN
from SVC_1 import svc
from LSTM import lstm_df
from MLNN_copy import mlnn
import questionary
import utils.helpful_functions_1 as hf
# import shelve
import fire
# from utils.yfinance_ticker_cols import (scan_col_list)
import pandas as pd
import sqlalchemy
from pathlib import Path
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
# import h5py
import hvplot.pandas
# import bokeh
from holoviews.plotting.links import RangeToolLink
# from DLNN import dataframe


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

    candle_1m_df, dt_end, dt_start = hf.get_minute_candles(ticker)
    candle_1m_df.to_sql(ticker + "_1_Min_Candles",
                        con=engine, if_exists='replace')

    # df = hf.add_support_resistance(candle_1m_df, candle_1d_df)

    df = hf.add_trade_signals(candle_1m_df)
    df = hf.add_overlap_studies(df)
    df = hf.add_momentum_indicators(df)
    df = hf.add_volume_indicators(df)
    df = hf.add_volatility_indicators(df)
    df = hf.add_price_transform_functions(df)
    df = hf.add_cycle_indicator_functions(df)
    df = hf.add_statistic_functions(df)
    # df = hf.add_support_resistance(df, candle_1d_df)
    df.dropna(inplace=True)
    print(df.tail())

    if(questionary.confirm("Save to database?").ask()):
        df.dropna().to_sql(ticker + '_Indicators', con=engine, if_exists='replace')
    
    mlnn_model_loss, mlnn_model_accuracy = mlnn(df)
    
    dlnn_summary, dlnn_model_loss, dlnn_model_accuracy = dataframe(dt_start, dt_end, df)
    
    svc_results, svc_conf_matrix, svc_class_report = svc(df)

    lstm_RMSE = lstm_df(df)

    
    print(80*'-')
    print(f"Shallow Neural Network: 1 input layer, 1 output layer.")
    print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")    
    print("Model Results")
    print(f"Loss: {mlnn_model_loss}, Accuracy: {mlnn_model_accuracy}")

    print(80*'-')
    print(f"Deep Neural Network: 1 input layer, 1 hidden layer, 1 output layer.")
    print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
    print(dlnn_summary)
    print(f"Loss: {dlnn_model_loss}, Accuracy: {dlnn_model_accuracy}")

    print(80*'-')
    print(f"Support Vector Classification")
    print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
    print(svc_results)
    print(svc_conf_matrix, svc_class_report)

    print(80*'-')
    print(f"Long Short Term Memory")
    print(f"**Waiting on Andrews Description**.")
    print(f"RMSE: {lstm_RMSE}")
    # print(svc_conf_matrix, svc_class_report)

#     print(dataframe_SVC(df))
#     print(mlnn(df))
#     print(dataframe(dt_start, dt_end, df))
#     print(dataframe_SVC(df))
#     print(lstm_df(df))
    return

if __name__ == "__main__":
    fire.Fire(main)
