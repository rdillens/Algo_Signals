import questionary
import finnhubIO as fh
import pandas as pd

market_list = ['stock', 'crypto']

def build_portfolio(dict):
    market = choose_market()
    if market == 'crypto':
        exchange = choose_crypto_exchange()
        ticker_list = fh.get_crypto_tickers(exchange=exchange)
        ticker_df = pd.DataFrame(ticker_list)
        ticker_df['baseCurrency'] = ticker_df['displaySymbol'].apply(lambda x: x[x.find('/')+1:])
        ticker_df['quoteCurrency'] = ticker_df['displaySymbol'].apply(lambda x: x[:x.find('/')])

        while(questionary.confirm("Add a product?").ask()):
            ticker = choose_crypto(ticker_df)
            dict['portfolio'].append(ticker)
    if market == 'stock':
        stock_type = questionary.select(
            "What type of stock?",
            choices=fh.stocks_df['type'].unique()
        )
        symbol = questionary.select(
            "What symbol?",
            choices=fh.stocks_df['symbol'].unique(),
        )
        print(symbol)
        # dict['portfolio'].append(symbol)

    return dict

def choose_crypto(ticker_df):
    base_currency = questionary.select(
            "What currency do you use?",
            choices=ticker_df['baseCurrency'].unique(),
            ).ask()

    ticker = questionary.select(
            "Ok, what symbols do you like?",
            choices=ticker_df.loc[lambda df: df['baseCurrency'] == base_currency]['displaySymbol'],
            ).ask()

    return ticker

def init_portfolio(dict):
    print(f"Let's get you set up.")
    dict['portfolio'] = []
    return build_portfolio(dict)

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
