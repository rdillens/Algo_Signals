import finnhub
import pandas as pd
import os
from dotenv import load_dotenv
import json

import polygon

load_dotenv()
polygon_api_key = os.getenv("POLYGON_API_KEY")
if type(polygon_api_key) == str:
    print('API OK')
else:
    print('API NOT OK', type(polygon_api_key))


from polygon import RESTClient


def main():
    key = polygon_api_key

    # RESTClient can be used as a context manager to facilitate closing the underlying http session
    # https://requests.readthedocs.io/en/master/user/advanced/#session-objects
    with RESTClient(key) as client:
        resp = client.stocks_equities_daily_open_close("AAPL", "2021-06-11")
        print(f"On: {resp.from_} Apple opened at {resp.open} and closed at {resp.close}")


if __name__ == '__main__':
    main()