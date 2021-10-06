import sqlalchemy
import utils.helpful_functions as hf
import pandas as pd
import MLNN


engine = sqlalchemy.create_engine(hf.db_connection_string)

inspector = sqlalchemy.inspect(engine)
table_names = inspector.get_table_names()

ticker = 'FB'
df = pd.read_sql_table(ticker + '_Indicators', con=engine, index_col='Datetime')

# print(df.head())
# print(df.columns)
# print(df['Trade Signal'].value_counts())

print(MLNN.mlnn(df))
print(MLNN.mlnn(df, 2))

print(MLNN.dlnn(df))
print(MLNN.dlnn(df, 2))

print(MLNN.svc(df))

