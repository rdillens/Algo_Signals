import questionary
import finnhubIO as fh
import polygonIO as pg
import pandas as pd
import concurrent.futures


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
            print(len(ticker_list))

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
            print(fh.stocks_df.head())
            stock_type = questionary.select(
                "What type of stock?",
                choices=fh.stocks_df['type'].unique()
            ).ask()
            print(f"Selected: {stock_type}")

            # Filter stocks by type
            # stock_choices=fh.stocks_df.loc[lambda df: df['type'] == stock_type]['symbol'],
            stock_choices=fh.stocks_df.loc[lambda df: df['type'] == stock_type]['symbol']
            print(stock_choices)
            # stocks_list = fh.stocks_df['symbol'].unique()
            print(len(stock_choices))
            stock_df = analyze_stocks(stock_choices)
            ticker = questionary.select(
                "What symbol?",
                choices=stock_choices,
            ).ask()
            product_dict['ticker'] = ticker
            print(ticker)
            dict['portfolio'].append(product_dict)

    return dict


def analyze_stocks(stocks):
    print("Analyzing...")
    stock_df = pd.DataFrame(stocks).reset_index(drop=True)
    print(stock_df.head())

    market_cap_dict = {}
    exception_count = 0
    cap_count = 0
    exception_list = []
    key_error_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_mkt_cap = {executor.submit(get_stock_market_cap, stock): stock for stock in stocks[:100]}
        for future in concurrent.futures.as_completed(future_to_mkt_cap):
            stock = future_to_mkt_cap[future]
    #         print(stock, future.result())
            try:
                market_cap_dict[stock] = future.result()
                cap_count += 1
            except KeyError:
                key_error_list.append(stock)
            except Exception as exc:
                exception_count += 1
                exception_list.append(stock)
                print(f'\n{stock} generated a {type(exc)} exception: {exc}', end='\n')
            else:
                print(f'\r{stock}: {market_cap_dict[stock]}', end='')
        print(f'\rDone! {cap_count} stocks, {len(key_error_list)} key errors, {exception_count} unhandled exceptions.')
        print(f'Tickers with no market cap data:\n{key_error_list}')


    return stock_df

def get_stock_market_cap(stock):
#     print(f'\rGetting {stock} ticker...', end='')
    # ticker = yf.Ticker(stock)
    financials = pg.get_stock_financials(stock)

    # financials = fh.get_stock_basic_financials(stock)
    # profile = fh.get_stock_company_profile(stock)
    # print(financials)
    # return ticker.info['marketCap']
    return financials
      

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
