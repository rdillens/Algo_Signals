import questionary
import shelve
import pandas as pd


# Create a temporary SQLite database and populate the database with content from the etf.db seed file
database_connection_string = 'sqlite:///../Resources/portfolio.db'


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

def get_username():
    username = questionary.text("What is your name?").ask()
    with shelve.open('../Resources/shelf') as sh:
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




