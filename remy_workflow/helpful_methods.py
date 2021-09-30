import questionary
import shelve
import pandas as pd
from pathlib import Path
import remy_workflow.finnhubIO as fh


# Create a temporary SQLite database and populate the database with content from the etf.db seed file
database_connection_string = 'sqlite:///../Resources/portfolio.db'
shelf_path = './Resources/shelf'
market_list = ['stock', 'crypto']


def gen_df(
    table_name,
    engine,
):
    df = pd.read_sql_table(
        table_name,
        con=engine,
        index_col='Time',
        parse_dates=True,
    )
    return df


def get_username(username=None):
    if username is None:
        username = questionary.text(
            "What is your name?",
            qmark='',
            ).ask()
    with shelve.open(shelf_path) as sh:
        # Check to see if username exists in shelf
        if username in sh:
            message = f"Hello, {username}!"
        # If username does not exist, create empty dictionary
        else:
            sh[username] = {}
            message = f"It's nice to meet you, {username}!"
            sh.sync()
        print(message)

    return username


def choose_market():
    default_market = 'crypto'
    if default_market in market_list:
        market = questionary.select(
            "What market are you looking at?",
            choices=market_list,
            qmark='',
            default=default_market,
            ).ask()
    else:
        market = questionary.select(
            "What market are you looking at?",
            choices=market_list,
            qmark='',
            ).ask()

    return market


def choose_exchange(market=None):
    default_exchange = 'COINBASE'
    if market is None:
        market = choose_market()
    if market == 'stock':
        exchange = 'US'
    if market == 'crypto':
        crypto_list = fh.crypto_exchange_list
        if default_exchange in crypto_list:
            exchange = questionary.select(
                "What crypto exchange do you want to use?",
                choices = sorted(fh.crypto_exchange_list),
                qmark='',
                default=default_exchange,
                ).ask()
        else:
            exchange = questionary.select(
                "What crypto exchange do you want to use?",
                choices = sorted(fh.crypto_exchange_list),
                qmark='',
                ).ask()
    return exchange


def choose_product_type(market=None, exchange=None):
    if market == None:
        market = choose_market()
    if exchange == None:
        exchange = choose_exchange(market)

    if market == 'crypto':
        default_base = 'USD'
        ticker_df = gen_crypto_df(exchange)

        base_list = sorted(ticker_df['baseCurrency'].unique())
        print(type(base_list))
        if default_base in base_list:
            print(f'found {default_base}')

            product_type = questionary.select(
                    "What currency do you use?",
                    choices=base_list,
                    qmark='',
                    default='USD',
                    ).ask()

        else:
            product_type = questionary.select(
                    "What currency do you use?",
                    choices=base_list,
                    qmark='',
                    ).ask()


    if market == 'stock':
        default_type = 'Common Stock'
        stock_types = sorted(fh.stocks_df['type'].unique())
        if default_type in stock_types:
            product_type = questionary.select(
                "What type of stock?",
                choices=stock_types,
                qmark='',
                default=default_type,
            ).ask()
        else:
            product_type = questionary.select(
                "What type of stock?",
                choices=stock_types,
                qmark='',
            ).ask()

    return product_type

def gen_product_df(market=None, exchange=None, product_type=None):
    if market == None:
        market = choose_market()
    if exchange == None:
        exchange = choose_exchange(market)
    if product_type == None:
        product_type = choose_product_type(market, exchange)

    if market == 'stock':
        product_df=fh.stocks_df.loc[lambda df: df['type'] == product_type]['symbol'].reset_index(drop=True)

    if market == 'crypto':
        crypto_df = gen_crypto_df(exchange)
        product_df=crypto_df.loc[lambda df: df['baseCurrency'] == product_type]['displaySymbol'].reset_index(drop=True)

    return product_df

def gen_crypto_df(exchange=None):
    if exchange == None:
        exchange = choose_exchange()
    crypto_list = fh.get_crypto_tickers(exchange=exchange)
    df = pd.DataFrame(crypto_list)
    df['baseCurrency'] = df['displaySymbol'].apply(lambda x: x[x.find('/')+1:])
    df['quoteCurrency'] = df['displaySymbol'].apply(lambda x: x[:x.find('/')])
    
    return df


def get_market_info(df):
    for item in df:
        info = get_product_info(item)
        # print(f'{item}', info)
    return df


def get_product_info(symbol):
    prod_dict = {}
    return prod_dict
