import questionary
import finnhubIO as fh
import pandas as pd

default_portfolio = ['AAPL', 'MMM']
market_list = ['stock', 'crypto']

def build_portfolio(dict):
    print(f"Let's get you set up.")
    dict['portfolio'] = ['ABC', '123']
    market = choose_market()
    if market == 'crypto':
        exchange = choose_crypto_exchange()
        ticker_list = fh.get_crypto_tickers(exchange=exchange)
        ticker_df = pd.DataFrame(ticker_list)
        ticker = questionary.select(
                "Ok, what symbols do you like?",
                choices=ticker_df['displaySymbol'],
                ).ask()

    return dict

def init_portfolio(dict):
    print(f"Let's get you set up.")
    dict['portfolio'] = []
    return dict

def choose_market():
    market = questionary.select(
        "What market are you looking at?",
        choices=market_list,
        ).ask()
    return market

def choose_crypto_exchange():
    exchange = questionary.select(
        "What exchange do you want to use?",
        choices = fh.crypto_exchange_list,
        ).ask()
    return exchange
