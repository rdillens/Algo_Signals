from datetime import datetime
# from logging import info
# from multiprocessing import Value
# from os import symlink
import questionary
import shelve
import pandas as pd
import sqlalchemy
# from pathlib import Path
import remy_workflow.finnhubIO as fh
# from time import sleep
import yfinance as yf
import concurrent.futures


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
    default_market = 'stock'
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


def get_market_info(df, market, exchange, product_type, engine):
    inspector = sqlalchemy.inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        df = pd.read_sql_table(table_names[0], con=engine)
        print(f"Loaded db {len(df)} items")
    # else:
    #     df = pd.DataFrame(df)
    # sleep_time = 1.0/10.0
    # print(market, type(market))
    if market == 'stock':
        time_start = datetime.now()
        # df.set_index('symbol', inplace=True)
        print(f'Found {len(df)}')
        symbol_list = list(df['symbol'])

        search_limit = 10
        sliced_symbol_list = symbol_list[:search_limit]
        df = get_threaded_info(sliced_symbol_list, df).copy()

        run_time = datetime.now() - time_start
        estimated_total_run_time = len(symbol_list) * (run_time.total_seconds()/search_limit) / 60
        print(f"Time to run: {run_time}\nEstimated time to run entire market: {estimated_total_run_time} minutes.")
    return df


def get_product_info(symbol, market, exchange, product_type):
    print(f"Getting info for {symbol} {market} {exchange} {product_type}")
    if market=='stock':
        ticker = yf.Ticker(symbol)
        print(ticker.info)
        # return fh.finnhub_client.aggregate_indicator(symbol, 'D')
    return {}

def get_threaded_info(stocks, df):
    exception_count = 0
    cap_count = 0
    exception_list = []
    key_error_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dict = {executor.submit(get_stock_info, stock): stock for stock in stocks}
        for future in concurrent.futures.as_completed(future_to_dict):
            stock = future_to_dict[future]
            try:
                info = future.result()

                for k, v in info.items():
                    if k not in df.columns:
                        df[k] = pd.NA
                    try:
                        df[k].loc[df['symbol'] == stock] = v
                    except ValueError:
                        df[k].loc[df['symbol'] == stock] = pd.NA
                    except Exception as e:
                        print(stock, k, v, type(e), e)
                        exception_count += 1

                cap_count += 1
            except KeyError:
                key_error_list.append(stock)
            except Exception as exc:
                exception_count += 1
                exception_list.append(stock)
                print(f'\n{stock} generated a {type(exc)} exception: {exc}', end='\n')
            else:
                pass
                print(f"\r{stock:<6s}", end="")
        print(f'\rDone! {cap_count} stocks, {len(key_error_list)} key errors, {exception_count} unhandled exceptions.')
        print(f'Tickers with no market cap data:\n{key_error_list}')
    # return info_dict
    return df


def get_stock_info(stock):
    try:
        ticker = yf.Ticker(stock)
    except Exception as e:
        print(f"{e}")
        return e
    return ticker.info


def get_stock_market_cap(stock):
#     print(f'\rGetting {stock} ticker...', end='')
    ticker = yf.Ticker(stock)
    return ticker.info['marketCap']

def process_ticker_info(ticker):
    print(ticker)
    stock_info = get_stock_info(ticker)
    stock_info_list = list(stock_info.items())
    product_df = pd.DataFrame(stock_info_list, columns=['Info', ticker])
    product_df = product_df.set_index("Info")
    # product_df = product_df.T
    drop_rows = ['zip', 'sector', 'fullTimeEmployees', 'longBusinessSummary', 'city', 'phone', 'state', 'country', 'companyOfficers', 'website', 'maxAge', 'address1', 'address2', 'industry', 'logo_url', 'tradeable', 'fromCurrency', ]        
    # print(product_df.index)
    for row in drop_rows:
        if row in product_df.index:
            product_df = product_df.drop(index=row)
    # print(product_df.head(20))
    # print(product_df.tail(20))

    return product_df


def process_ticker_hist(ticker):
    candle_df = yf.download(ticker, period="max")
    print(candle_df.head())
    candle_df.rename_axis('Datetime', inplace=True)
    candle_df = candle_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return candle_df
      
def get_minute_candles(ticker):
    current_time = int(round(datetime.now().timestamp(),0))
    print(datetime.fromtimestamp(current_time))
    max_offset = 86400 * 10 * 365
    dt_end = current_time
    dt_start = dt_end - max_offset
    candles = fh.get_stock_candles(
        ticker, 
        dt_start=dt_start, 
        dt_end=dt_end, 
        resolution='1'
        )
    # This is causing an error
    try:
        candle_df = pd.DataFrame(candles)
    except Exception as e:
        print(f'Error: {e} {type(e)}')
        raise e
    candle_df.rename(
        columns={
            't': 'Datetime', 
            'c': 'Close', 
            'h': 'High', 
            'l': 'Low', 
            'o': 'Open', 
            's': 'Status', 
            'v': 'Volume',
            },
        inplace=True,
        )
    candle_df['Datetime'] = candle_df['Datetime'].apply(lambda df: datetime.fromtimestamp(df))
    candle_df.set_index('Datetime', inplace=True, drop=True)
    # print(candle_df)
    candle_df = candle_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return candle_df
