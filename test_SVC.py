# import utils.ta_lib_indicators as ti
import sqlalchemy
import utils.helpful_functions as hf
import pandas as pd
from SVC_1 import dataframe_SVC


engine = sqlalchemy.create_engine(hf.db_connection_string)

inspector = sqlalchemy.inspect(engine)
table_names = inspector.get_table_names()
print(table_names)

ticker = 'FB'
df = pd.read_sql_table(ticker + '_Indicators', con=engine, index_col='Datetime')

print(df.head())
print(df.columns)
print(df['Trade Signal'].value_counts())
# print(dataframe_SVC(df))