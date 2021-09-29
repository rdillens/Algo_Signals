import finnhub
import pandas as pd
import os
from dotenv import load_dotenv
import json

from polygon import RESTClient

# Get API key from .env file
load_dotenv()
polygon_api_key = os.getenv("POLYGON_API_KEY")
if type(polygon_api_key) == str:
    print('Polygon API OK')
else:
    print('API NOT OK', type(polygon_api_key))
    print('Check your .env file for the POLYGON_API_KEY value.')
    print('Sign-up and get an API key at https://polygon.io/')


def get_ticker_open_close(key, ticker, date):
    # RESTClient can be used as a context manager to facilitate closing the underlying http session
    # https://requests.readthedocs.io/en/master/user/advanced/#session-objects
    with RESTClient(key) as client:
        return client.stocks_equities_daily_open_close(ticker, date)

def get_stock_financials(stock, key=polygon_api_key):
    with RESTClient(key) as client:
        # client.stocks_equities_daily_open_close(ticker, date)
        financials = client.reference_stock_financials(stock)
        print(financials)
    return financials

def main():
    resp = get_ticker_open_close(polygon_api_key, 'AAPL', '2021-07-14')
    print(f"On: {resp.from_} Apple opened at {resp.open} and closed at {resp.close}")


if __name__ == '__main__':
    main()