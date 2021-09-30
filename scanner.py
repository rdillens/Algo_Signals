import questionary
import remy_workflow.helpful_methods as hm
import shelve
import fire
# import pandas


def main(username=None):
    # Check for username in shelf, create new record if one does not exist
    username = hm.get_username(username)

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
    hm.get_market_info(product_df)
    print(product_df.head(10))
    print(len(product_df), type(product_df))
    

    return username



if __name__ == "__main__":
    # fire.Fire(main)
    print(f"Goodbye, {fire.Fire(main)}")