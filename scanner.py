import questionary
import remy_workflow.helpful_methods as hm
import shelve
import fire
from remy_workflow.yfinance_ticker_cols import (col_list, scan_col_list)
# import pandas
import sqlalchemy
db_connection_string = 'sqlite:///./Resources/products.db'

engine  = sqlalchemy.create_engine(db_connection_string)

def main(user=None):
    # Check for username in shelf, create new record if one does not exist
    username = hm.get_username(user)

    # Choose what market to scan: Stock, Crypto, or Forex
    market = hm.choose_market()
    # Choose the exchange for crypto, stocks use 'US', 
    exchange = hm.choose_exchange(market)
    # Choose type of product
    product_type = hm.choose_product_type(market, exchange)
    # Create list of products to scan
    product_df = hm.gen_product_df(market, exchange, product_type)
    # Sort values and reset index
    product_df = product_df.sort_values().reset_index(drop=True)
    # Query with each product in dataframe to get current market info
    product_df = hm.get_market_info(product_df, market, exchange, product_type)
    print(product_df.head(10))
    # print(len(product_df), type(product_df))
    scan_product_df = product_df[scan_col_list]
    print(scan_product_df.head())

    product_df.to_sql("Stock_Company_Info", con=engine)

    return username



if __name__ == "__main__":
    # fire.Fire(main)
    print(f"Goodbye, {fire.Fire(main)}")