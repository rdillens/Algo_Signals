from datetime import datetime, timedelta
import questionary
import shelve
import pandas as pd
import sqlalchemy
import utils.finnhubIO as fh
import yfinance as yf
import concurrent.futures
import utils.ta_lib_indicators as ti
import talib
from pandas.tseries.offsets import BDay
from datetime import date


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


def input_ticker():
    resp = questionary.text(
        "What stock ticker should I look up?",
        qmark='',
    ).ask()
    with shelve.open(shelf_path) as sh:
        # Check to see if resp exists in shelf
        if resp not in sh:
            sh[resp] = {}
            message = f"Let me add {resp} to my files..."
        # If resp does not exist, create empty dictionary
        else:
            message = f"Ok, let's look at {resp}."
        print(message)
        return resp


def choose_patterns():
    default = 'Doji'
    pattern_list = []
    pattern_df = pd.DataFrame(
        list(ti.pattern_recognition.items()),
        columns=['Index', 'Pattern'],
    )
    pattern_df = pattern_df.set_index('Index')
    patterns_list = list(pattern_df['Pattern'])
    choice = choose_from_list(
        patterns_list,
        default=default,
        prompt_string="Choose a Pattern:"
    )
    patterns_list.remove(choice)
    pattern_list.append(choice)
    while(questionary.confirm("Add another pattern?").ask()):
        choice = choose_from_list(
            patterns_list,
            default=default,
            prompt_string="Choose a Pattern:"
        )
        patterns_list.remove(choice)
        pattern_list.append(choice)

    pattern_index_list = pattern_df[pattern_df['Pattern'].isin(
        pattern_list)].index
    # print(pattern_index_list)
    return pattern_index_list


def choose_functions(function_dict, function_name, default=None):
    function_list = []
    function_df = pd.DataFrame(
        list(function_dict.items()),
        columns=['Index', function_name],
    )
    function_df = function_df.set_index('Index')
    functions_list = list(function_df[function_name])
    choice = choose_from_list(
        functions_list,
        default=default,
        prompt_string=f"Choose a {function_name}:"
    )
    functions_list.remove(choice)
    function_list.append(choice)

    while(questionary.confirm(f"Add another {function_name}?").ask() and len(functions_list) > 0):
        choice = choose_from_list(
            functions_list,
            # default=default,
            prompt_string=f"Choose a {function_name}:"
        )
        functions_list.remove(choice)
        function_list.append(choice)

    function_index_list = function_df[function_df[function_name].isin(
        function_list)].index
    return function_index_list


def choose_from_list(
    choice_list,
    default=None,
    prompt_string=None
):
    if prompt_string is None:
        prompt_string = "Choose from the list:"
    if default in choice_list:
        resp = questionary.select(
            prompt_string,
            choices=choice_list,
            qmark='',
            default=default,
        ).ask()
    else:
        resp = questionary.select(
            prompt_string,
            choices=choice_list,
            qmark='',
        ).ask()

    return resp


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
                choices=sorted(fh.crypto_exchange_list),
                qmark='',
                default=default_exchange,
            ).ask()
        else:
            exchange = questionary.select(
                "What crypto exchange do you want to use?",
                choices=sorted(fh.crypto_exchange_list),
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
        product_df = fh.stocks_df.loc[lambda df: df['type']
                                      == product_type]['symbol'].reset_index(drop=True)

    if market == 'crypto':
        crypto_df = gen_crypto_df(exchange)
        product_df = crypto_df.loc[lambda df: df['baseCurrency']
                                   == product_type]['displaySymbol'].reset_index(drop=True)

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
    # inspector = sqlalchemy.inspect(engine)
    # table_names = inspector.get_table_names()
    # if table_names:
    #     df = pd.read_sql_table(table_names[0], con=engine)
    #     print(f"Loaded db {len(df)} items")
    # else:
    #     df = pd.DataFrame(df)
    # sleep_time = 1.0/10.0
    # print(market, type(market))
    if market == 'stock':
        time_start = datetime.now()
        # df.set_index('symbol', inplace=True)
        print(f'Found {len(df)}')
        # print(df.head())
        # symbol_list = list(df['symbol'])

        search_limit = 10
        # sliced_symbol_list = symbol_list[:search_limit]
        # df = get_threaded_info(sliced_symbol_list, df).copy()

        run_time = datetime.now() - time_start
        # estimated_total_run_time = len(symbol_list) * (run_time.total_seconds()/search_limit) / 60
        # print(f"Time to run: {run_time}\nEstimated time to run entire market: {estimated_total_run_time} minutes.")
    return df


def get_product_info(symbol, market, exchange, product_type):
    print(f"Getting info for {symbol} {market} {exchange} {product_type}")
    if market == 'stock':
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
        future_to_dict = {executor.submit(
            get_stock_info, stock): stock for stock in stocks}
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
                print(
                    f'\n{stock} generated a {type(exc)} exception: {exc}', end='\n')
            else:
                pass
                print(f"\r{stock:<6s}", end="")
        print(
            f'\rDone! {cap_count} stocks, {len(key_error_list)} key errors, {exception_count} unhandled exceptions.')
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
    drop_rows = ['zip', 'sector', 'fullTimeEmployees', 'longBusinessSummary', 'city', 'phone', 'state', 'country',
                 'companyOfficers', 'website', 'maxAge', 'address1', 'address2', 'industry', 'logo_url', 'tradeable', 'fromCurrency', ]
    # print(product_df.index)
    for row in drop_rows:
        if row in product_df.index:
            product_df = product_df.drop(index=row)
    # print(product_df.head(20))
    # print(product_df.tail(20))

    return product_df


def process_ticker_hist(ticker, interval='1d'):
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    candle_df = yf.download(
        ticker,
        period="max",
        interval=interval,
    )
    # print(candle_df.head())
    candle_df.rename_axis('Datetime', inplace=True)
    candle_df = candle_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return candle_df


def get_minute_candles(ticker):
    current_time = int(round(datetime.now().timestamp(), 0))
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
    candle_df['Datetime'] = candle_df['Datetime'].apply(
        lambda df: datetime.fromtimestamp(df))
    candle_df.set_index('Datetime', inplace=True, drop=True)
    # print(candle_df)
    candle_df = candle_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return candle_df, dt_end, dt_start



# def sub_years(years):
#     today = date.today()
#     try:
#         return today.replace(year = today.year - years)
#     except ValueError:
#         return today + (date(today.year + years, 1, 1) - date(today.year, 1, 1))


# def start_end():
#     # years = 10   
#     today = date.today()
#     # try:
#     #     return today.replace(year = today.year - years)
#     # except ValueError:
#     #     return today + (date(today.year + years, 1, 1) - date(today.year, 1, 1))
#     # # historical data - define START and END dates
#     # # to calculate the start_date we must use the sub_years function defined above to get today's date and subtract 10 years
#     # # then using the .strftime('%Y-%m-%d') we format it so that it can be passed to yahoo finance

#     start_date = sub_years(10).strftime('%Y-%m-%d')

#     # # for the end_date we just have to reformat the today variable with the .strftime('%Y-%m-%d') we format it so that it can be passed to yahoo finance 
#     end_date = today.strftime('%Y-%m-%d')

#     return start_date, end_date 



def add_trade_signals(df):
    pattern_list = choose_patterns()
    # print(pattern_list)
    for pattern in pattern_list:

        pattern_function = getattr(talib, pattern)
        try:
            result = pattern_function(
                df['Open'], df['High'], df['Low'], df['Close'])
            df[pattern] = result
        except Exception as e:
            print(f"{type(e)} Exception! {e}")
    # print(df.head())

    len(pattern_list)
    df['Sum Patterns'] = df.iloc[:, -(len(pattern_list)):].sum(axis=1)

    df['Trade Signal'] = 0.0

    threshold_value = 0.0

    def check_sum_value(sum_value):
        if sum_value > threshold_value:
            return 1
        elif sum_value < -threshold_value:
            return -1
        else:
            return 0.0

    df['Trade Signal'] = df['Sum Patterns'].apply(lambda x: check_sum_value(x))
    df.drop(columns='Sum Patterns', inplace=True)

    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Trade Signal']].dropna()


def add_overlap_studies(df):
    if(questionary.confirm('Add overlap study?').ask()):
        function_list = choose_functions(
            ti.overlap_studies, 'Overlap Study', default='Bollinger Bands')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'BBANDS':
                # upperband, middleband, lowerband = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
                df['Upper Band'], df['Middle Band'], df['Lower Band'] = function(
                    df['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            if f == 'DEMA':
                # real = DEMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30) # wish list item
            if f == 'EMA':
                # real = EMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)
            if f == 'HT_TRENDLINE':
                # real = HT_TRENDLINE(close)
                df[f] = function(df['Close'])
            if f == 'KAMA':
                # real = KAMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)

            # There are only 8 types of moving averages, enumerated 0 to 8:
            # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3
            ma_types = ["SMA", "EMA", "WMA", "DEMA",
                        "TEMA", "TRIMA", "KAMA", "MAMA", "T3"]

            if f == 'MA':
                # real = MA(close, timeperiod=30, matype=0)
                df[f] = function(df['Close'], timeperiod=30, matype=0)
            if f == 'MAMA':
                # mama, fama = MAMA(close, fastlimit=0.5, slowlimit=0.05)
                df['MAMA'], df['FAMA'] = function(
                    df['Close'], fastlimit=0.5, slowlimit=0.05)

            # TypeError: Argument 'periods' has incorrect type (expected numpy.ndarray, got int)
            # if f == 'MAVP':
            # periods needs to be an array of the same length as df['Close']
            #     # real = MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
            #     df[f] = function(df['Close'], 3, minperiod=2, maxperiod=30, matype=0)

            if f == 'MIDPOINT':
                # real = MIDPOINT(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'MIDPRICE':
                # real = MIDPRICE(high, low, timeperiod=30)
                df[f] = function(df['High'], df['Low'], timeperiod=14)
            if f == 'SAR':
                # real = SAR(high, low, acceleration=0, maximum=0)
                df[f] = function(df['High'], df['Low'],
                                 acceleration=0, maximum=0)
            if f == 'SAREXT':
                # real = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
                df[f] = function(df['High'], df['Low'], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0,
                                 accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
            if f == 'SMA':
                # real = SMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)
            if f == 'T3':
                # real = T3(close, timeperiod=5, vfactor=0)
                df[f] = function(df['Close'], timeperiod=5, vfactor=0)
            if f == 'TEMA':
                # real = TEMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)
            if f == 'TRIMA':
                # real = TRIMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)
            if f == 'WMA':
                # real = WMA(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)

    return df


def add_momentum_indicators(df):
    if(questionary.confirm('Add momentum indicator?').ask()):
        function_list = choose_functions(
            ti.momentum_indicators, 'Momentum Indicator', default='Moving Average Convergence/Divergence')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'ADX':
                # real = ADX(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'ADXR':
                # real = ADXR(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'APO':
                # real = APO(close, fastperiod=12, slowperiod=26, matype=0)
                df[f] = function(df['Close'], fastperiod=12,
                                 slowperiod=26, matype=0)
            if f == 'AROON':
                # aroondown, aroonup = AROON(high, low, timeperiod=14)
                df['AROONDOWN'], df['AROONUP'] = function(
                    df['High'], df['Low'], timeperiod=14)
            if f == 'AROONOSC':
                # real = AROONOSC(high, low, timeperiod=14)
                df[f] = function(df['High'], df['Low'], timeperiod=14)
            if f == 'BOP':
                # real = BOP(open, high, low, close)
                df[f] = function(df["Open"], df['High'],
                                 df['Low'], df['Close'])
            if f == 'CCI':
                # real = CCI(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'CMO':
                # real = CMO(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'DX':
                # real = DX(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'MACD':
                # macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = function(
                    df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            if f == 'MACDEXT':
                # macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
                df['MACDEXT'], df['MACDEXT_SIGNAL'], df['MACDEXT_HIST'] = function(
                    df['Close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
            if f == 'MACDFIX':
                # macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)
                df['MACDFIX'], df['MACDFIX_SIGNAL'], df['MACDFIX_HIST'] = function(
                    df['Close'], signalperiod=9)
            if f == 'MFI':
                # real = MFI(high, low, close, volume, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], df['Volume'], timeperiod=14)
            if f == 'MINUS_DI':
                # real = MINUS_DI(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'MINUS_DM':
                # real = MINUS_DM(high, low, timeperiod=14)
                df[f] = function(df['High'], df['Low'], timeperiod=14)
            if f == 'MOM':
                # real = MOM(close, timeperiod=10)
                df[f] = function(df['Close'], timeperiod=10)
            if f == 'PLUS_DI':
                # real = PLUS_DI(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'PLUS_DM':
                # real = PLUS_DM(high, low, timeperiod=14)
                df[f] = function(df['High'], df['Low'], timeperiod=14)
            if f == 'PPO':
                # real = PPO(close, fastperiod=12, slowperiod=26, matype=0)
                df[f] = function(df['Close'], fastperiod=12,
                                 slowperiod=26, matype=0)
            if f == 'ROC':
                # real = ROC(close, timeperiod=10)
                df[f] = function(df['Close'], timeperiod=10)
            if f == 'ROCP':
                # real = ROCP(close, timeperiod=10)
                df[f] = function(df['Close'], timeperiod=10)
            if f == 'ROCR':
                # real = ROCR(close, timeperiod=10)
                df[f] = function(df['Close'], timeperiod=10)
            if f == 'ROCR100':
                # real = ROCR100(close, timeperiod=10)
                df[f] = function(df['Close'], timeperiod=10)
            if f == 'RSI':
                # real = RSI(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'STOCH':
                # slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                df['STOCH_SLOWK'], df['STOCH_SLOWD'] = function(
                    df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            if f == 'STOCHF':
                # fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
                df['STOCHF_FASTK'], df['STOCHF_FASTD'] = function(
                    df['High'], df['Low'], df['Close'], fastk_period=5, fastd_period=3, fastd_matype=0)
            if f == 'STOCHRSI':
                # fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
                df['STOCHRSI_FASTK'], df['STOCHRSI_FASTD'] = function(
                    df['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
            if f == 'TRIX':
                # real = TRIX(close, timeperiod=30)
                df[f] = function(df['Close'], timeperiod=30)
            if f == 'ULTOSC':
                # real = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
                df[f] = function(df['High'], df['Low'], df['Close'],
                                 timeperiod1=7, timeperiod2=14, timeperiod3=28)
            if f == 'WILLR':
                # real = WILLR(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)

    return df


def add_volume_indicators(df):
    if(questionary.confirm('Add volume indicator?').ask()):
        function_list = choose_functions(
            ti.volume_indicators, 'Volume Indicator')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'AD':
                # real = AD(high, low, close, volume)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], df['Volume'])
            if f == 'ADOSC':
                # real = ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
                df[f] = function(df['High'], df['Low'], df['Close'],
                                 df['Volume'], fastperiod=3, slowperiod=10)
            if f == 'OBV':
                # real = OBV(close, volume)
                df[f] = function(df['Close'], df['Volume'])
    return df


def add_volatility_indicators(df):
    if(questionary.confirm('Add volatility indicator?').ask()):
        function_list = choose_functions(
            ti.volatility_indicators, 'Volatility Indicator', default='Average True Range')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'ATR':
                # real = ATR(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'NATR':
                # real = NATR(high, low, close, timeperiod=14)
                df[f] = function(df['High'], df['Low'],
                                 df['Close'], timeperiod=14)
            if f == 'TRANGE':
                # real = TRANGE(high, low, close)
                df[f] = function(df['High'], df['Low'], df['Close'])
    return df


def add_price_transform_functions(df):
    if(questionary.confirm('Add price transform function?').ask()):
        function_list = choose_functions(
            ti.price_transform, 'Price Transform Function', default='Weighted Close Price')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'AVGPRICE':
                # real = AVGPRICE(open, high, low, close)
                df[f] = function(df['Open'], df['High'],
                                 df['Low'], df['Close'])
            if f == 'MEDPRICE':
                # real = MEDPRICE(high, low)
                df[f] = function(df['High'], df['Low'])
            if f == 'TYPPRICE':
                # real = TYPPRICE(high, low, close)
                df[f] = function(df['High'], df['Low'], df['Close'])
            if f == 'WCLPRICE':
                # real = WCLPRICE(high, low, close)
                df[f] = function(df['High'], df['Low'], df['Close'])
    return df


def add_cycle_indicator_functions(df):
    if(questionary.confirm('Add cycle indicator?').ask()):
        function_list = choose_functions(
            ti.cycle_indicators, 'Cycle Indicator Function')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'HT_DCPERIOD':
                # real = HT_DCPERIOD(close)
                df[f] = function(df['Close'])
            if f == 'HT_DCPHASE':
                # real = HT_DCPHASE(close)
                df[f] = function(df['Close'])
            if f == 'HT_PHASOR':
                # inphase, quadrature = HT_PHASOR(close)
                df['INPHASE'], df['QUADRATURE'] = function(df['Close'])
            if f == 'HT_SINE':
                # sine, leadsine = HT_SINE(close)
                df['SINE'], df['LEADSINE'] = function(df['Close'])
            if f == 'HT_TRENDMODE':
                # integer = HT_TRENDMODE(close)
                df['INTEGER'] = function(df['Close'])
    return df


def add_statistic_functions(df):
    if(questionary.confirm('Add statistic function?').ask()):
        function_list = choose_functions(
            ti.statistic_functions, 'Statistic Function', default='Linear Regression')
        for f in function_list:
            function = getattr(talib, f)
            if f == 'BETA':
                # real = BETA(high, low, timeperiod=5)
                df[f] = function(df['High'], df['Low'], timeperiod=5)
            if f == 'CORREL':
                # real = CORREL(high, low, timeperiod=30)
                df[f] = function(df['High'], df['Low'], timeperiod=30)
            if f == 'LINEARREG':
                # real = LINEARREG(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'LINEARREG_ANGLE':
                # real = LINEARREG_ANGLE(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'LINEARREG_INTERCEPT':
                # real = LINEARREG_INTERCEPT(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'LINEARREG_SLOPE':
                # real = LINEARREG_SLOPE(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'STDDEV':
                # real = STDDEV(close, timeperiod=5, nbdev=1)
                df[f] = function(df['Close'], timeperiod=5, nbdev=1)
            if f == 'TSF':
                # real = TSF(close, timeperiod=14)
                df[f] = function(df['Close'], timeperiod=14)
            if f == 'VAR':
                # real = VAR(close, timeperiod=5, nbdev=1)
                df[f] = function(df['Close'], timeperiod=5, nbdev=1)
    return df


def check_b_date(check_date, date_list):
    if (check_date in date_list):

        return check_date
    else:
        return check_date - BDay(1)


def get_support(check_date, daily_df):
    return daily_df.loc[check_date]['Low']


def get_resistance(check_date, daily_df):
    return daily_df.loc[check_date]['High']

def add_support_resistance(minutely_df, daily_df):
    day_list = list(daily_df.index)
    # print(day_list)
    df = minutely_df.copy()
    df['Previous Day'] = df.index.date - BDay(1)
    df['Previous Day'] = df['Previous Day'].apply(
        lambda x: check_b_date(
            x,
            day_list
        )
    )
    df['Previous Day'] = df['Previous Day'].apply(
        lambda x: check_b_date(
            x,
            day_list
        )
    )
    sr_type = choose_from_list(sr_types_list, 'Traditional', 'Choose pivot points (support/resistance)')
    if sr_type == 'Traditional':
        sr_df = sr_traditional(daily_df)
    if sr_type == 'Fibonacci':
        sr_df = sr_fibonacci(daily_df)
    if sr_type == 'Woodie':
        sr_df = sr_woodie(daily_df)
    if sr_type == 'Classic':
        sr_df = sr_classic(daily_df)
    if sr_type == 'Denmark':
        sr_df = sr_denmark(daily_df)
    if sr_type == 'Camarilla':
        sr_df = sr_camarilla(daily_df)
        
    # df['Support'] = df['Previous Day'].apply(
    #     lambda x: get_support(
    #         x,
    #         daily_df,
    #     )
    # )

    # df['Resistance'] = df['Previous Day'].apply(
    #     lambda x: get_resistance(
    #         x,
    #         daily_df,
    #     )
    # )

    # print(df.iloc[-2000:].head(20))
    return df


sr_types_list = ['Traditional', 'Fibonacci', 'Woodie', 'Classic', 'Denmark', 'Camarilla']


def sr_traditional(df):
    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = df['PP'] * 2 - df['Low'].shift(1)
    df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['R3'] = df['PP'] * 2 + (df['High'].shift(1) - 2 * df['Low'].shift(1))
    df['R4'] = df['PP'] * 3 + (df['High'].shift(1) - 3 * df['Low'].shift(1))
    df['R5'] = df['PP'] * 4 + (df['High'].shift(1) - 4 * df['Low'].shift(1))
    df['S1'] = df['PP'] * 2 - df['High'].shift(1)
    df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))
    df['S3'] = df['PP'] * 2 - (2 * df['High'].shift(1) - df['Low'].shift(1))
    df['S4'] = df['PP'] * 3 - (3 * df['High'].shift(1) - df['Low'].shift(1))
    df['S5'] = df['PP'] * 4 - (4 * df['High'].shift(1) - df['Low'].shift(1))
    
    return df


def sr_fibonacci(df):
    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = df['PP'] + 0.382 * (df['High'].shift(1) - df['Low'].shift(1))
    df['S1'] = df['PP'] - 0.382 * (df['High'].shift(1) - df['Low'].shift(1))
    df['R2'] = df['PP'] + 0.618 * (df['High'].shift(1) - df['Low'].shift(1))
    df['S2'] = df['PP'] - 0.618 * (df['High'].shift(1) - df['Low'].shift(1))
    df['R3'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S3'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))
    
    return df


def sr_woodie(df):
    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + 2 * df['Open']) / 4
    df['R1'] = 2 * df['PP'] - df['Low'].shift(1)
    df['S1'] = 2 * df['PP'] - df['High'].shift(1)
    df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))
    df['R3'] =  df['High'].shift(1) + 2 * (df['PP'] -  df['Low'].shift(1))
    df['S3'] =  df['Low'].shift(1) - 2 * (df['High'].shift(1) - df['PP'])
    df['R4'] = df['R3'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S4'] = df['S3'] - (df['High'].shift(1) - df['Low'].shift(1))

    return df


def sr_classic(df):
    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2 * df['PP'] - df['Low'].shift(1)
    df['S1'] = 2 * df['PP'] - df['High'].shift(1)
    df['R2'] = df['PP'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S2'] = df['PP'] - (df['High'].shift(1) - df['Low'].shift(1))
    df['R3'] = df['PP'] + 2 * (df['High'].shift(1) - df['Low'].shift(1))
    df['S3'] = df['PP'] - 2 * (df['High'].shift(1) - df['Low'].shift(1))
    df['R4'] = df['PP'] + 3 * (df['High'].shift(1) - df['Low'].shift(1))
    df['S4'] = df['PP'] - 3 * (df['High'].shift(1) - df['Low'].shift(1))

    return df


def sr_denmark(df):
    df['X'] = pd.NA
    df['PP'] = pd.NA
    df['R1'] = pd.NA
    df['S1'] = pd.NA

    # Denmark S/R
    for index, row in df.iterrows():
        if df['Open'].shift(1) == df['Close'].shift(1):
            df.loc[index, 'X'] = df['High'].shift(1) + df['Low'].shift(1) + 2 * df['Close'].shift(1)
        elif df['Close'].shift(1) > df['Open'].shift(1):
            df.loc[index, 'X'] = 2 * df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)
        else:
            df.loc[index, 'X'] = 2 * df['Low'].shift(1) + df['High'].shift(1) + df['Close'].shift(1)
            df.loc[index, 'PP'] = df['X'] / 4
            df.loc[index, 'R1'] = df['X'] / 2 - df['Low'].shift(1)
            df.loc[index, 'S1'] = df['X'] / 2 - df['High'].shift(1)
        
    # IF  OPENprev == CLOSEprev
    #     X = HIGHprev + LOWprev + 2 * CLOSEprev
    # ELSE IF CLOSEprev >  OPENprev
    #     X = 2 * HIGHprev + LOWprev + CLOSEprev
    # ELSE
    #     X = 2 * LOWprev + HIGHprev + CLOSEprev
    #     PP = X / 4
    #     R1 = X / 2 - LOWprev
    #     S1 = X / 2 - HIGHprev    
    return df

def sr_camarilla(df):
    df['PP'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = df['Close'].shift(1) + 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 12
    df['S1'] = df['Close'].shift(1) - 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 12
    df['R2'] = df['Close'].shift(1) + 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 6
    df['S2'] = df['Close'].shift(1) - 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 6
    df['R3'] = df['Close'].shift(1) + 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 4
    df['S3'] = df['Close'].shift(1) - 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 4
    df['R4'] = df['Close'].shift(1) + 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 2
    df['S4'] = df['Close'].shift(1) - 1.1 * (df['High'].shift(1) - df['Low'].shift(1)) / 2    
    return df

