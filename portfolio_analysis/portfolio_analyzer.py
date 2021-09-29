import shelve
# import questionary
import pandas as pd
from questionary.constants import DEFAULT_STYLE
import helpful_methods as hm
import finnhubIO as fh
import datetime

def main(display_string):
    print(display_string)

    # Get username
    username = hm.get_username()
    with shelve.open('../Resources/shelf') as sh:
        for asset in sh[username]['portfolio']:
            analyze_asset(asset)

    return username


def analyze_asset(asset):
    ticker = asset['ticker']
    print("Analyzing...")
    print(f"{ticker}")

    current_time = int(round(datetime.datetime.now().timestamp(),0))
    print(current_time)
    max_offset = 86400 * 10 * 365
    # dt_end = 1591852249
    dt_end = current_time
    dt_start = dt_end - max_offset
    candles = fh.get_stock_candles(
        ticker, 
        dt_start=dt_start, 
        dt_end=dt_end, 
        resolution='D'
        )
    candle_df = pd.DataFrame(candles)
    print(candle_df)
    return


if __name__ == "__main__":
    username = main("The Four Headless Horsemen - Portfolio Analyzer")
    print(f"Goodbye, {username}.")