import questionary
import finnhubIO as fh
import pandas as pd


market_list = ['stock', 'crypto']


def build_portfolio(dict):
    while(questionary.confirm("Add a product?").ask()):
        product_dict = {}
        market = choose_market()
        product_dict['market'] = market
        # Add crypto market item to portfolio
        if market == 'crypto':
            exchange = choose_crypto_exchange()
            product_dict['exchange'] = exchange
            ticker_list = fh.get_crypto_tickers(exchange=exchange)
            ticker_df = pd.DataFrame(ticker_list)
            ticker_df['baseCurrency'] = ticker_df['displaySymbol'].apply(lambda x: x[x.find('/')+1:])
            ticker_df['quoteCurrency'] = ticker_df['displaySymbol'].apply(lambda x: x[:x.find('/')])

            ticker = choose_crypto(ticker_df)
            product_dict['ticker'] = ticker
            print(product_dict)
            dict['portfolio'].append(product_dict)

        # Add stock market item to portfolio
        if market == 'stock':
            product_dict['exchange'] = 'US'
            stock_type = questionary.select(
                "What type of stock?",
                choices=fh.stocks_df['type'].unique()
            ).ask()

            ticker = questionary.select(
                "What symbol?",
                choices=fh.stocks_df['symbol'].unique(),
            ).ask()
            product_dict['ticker'] = ticker
            print(ticker)
            dict['portfolio'].append(product_dict)

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


def add_funds(dict):
    funds = questionary.text("How much do you want to invest?").ask()
    try:
        if 'balance' in dict:
            dict['balance'] += float(funds)
        else:
            dict['balance'] = float(funds)
    except:
        print(f"{funds} is a invalid entry!")
        if 'balance' not in dict:
            dict['balance'] = 0
    else:
        print(funds)
    return dict
