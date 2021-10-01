import shelve
from numpy import save
import questionary
import pandas as pd
# from questionary.constants import DEFAULT_STYLE
import helpful_methods as hm
import finnhubIO as fh
import datetime
import sqlalchemy


# Create an engine to interact with the SQLite database
engine = sqlalchemy.create_engine(hm.database_connection_string)

def save_candle_db(df, table_name):
    df.to_sql(table_name, con=engine, if_exists='replace')
    return


def main(display_string):
    print(display_string)

    # Get username
    username = hm.get_username()
    with shelve.open('../Resources/shelf') as sh:
        for i, asset in enumerate(sh[username]['portfolio']):
            df = analyze_asset(asset)
            save_candle_db(df, username + "_Portfolio_" + str(i))

    return username


def analyze_asset(asset):
    ticker = asset['ticker']
    print("Analyzing...")
    print(f"{ticker}")

    current_time = int(round(datetime.datetime.now().timestamp(),0))
    print(datetime.datetime.fromtimestamp(current_time))
    max_offset = 86400 * 10 * 365
    dt_end = current_time
    dt_start = dt_end - max_offset
    candles = fh.get_stock_candles(
        ticker, 
        dt_start=dt_start, 
        dt_end=dt_end, 
        resolution='1'
        )
    print(candles)
    # This is causing an error
    try:
        candle_df = pd.DataFrame(candles)
    except Exception as e:
        print(f'Error: {e} {type(e)}')
        raise e
    candle_df.rename(
        columns={
            't': 'Time', 
            'c': 'Close', 
            'h': 'High', 
            'l': 'Low', 
            'o': 'Open', 
            's': 'Status', 
            'v': 'Volume',
            },
        inplace=True,
        )
    candle_df['Time'] = candle_df['Time'].apply(lambda df: datetime.datetime.fromtimestamp(df))
    candle_df.set_index('Time', inplace=True, drop=True)
    print(candle_df)
    return candle_df


if __name__ == "__main__":
    username = main("The Four Headless Horsemen - Portfolio Analyzer")
    print(f"Goodbye, {username}.")