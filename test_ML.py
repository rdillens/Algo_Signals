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

print(80*'-')
print(f"Shallow Neural Network: 1 input layer, 1 output layer.")
print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
print(80*'-')
print(MLNN.mlnn(df))

# print(80*'-')
# print(f"Shallow Neural Network: 1 input layer, 1 output layer")
# print(f"Binary clasifier identifying bullish or bearish signals, 0, 1, or -1.")
# print(80*'-')
# print(MLNN.mlnn(df, 2))

print(80*'-')
print(f"Deep Neural Network: 1 input layer, 1 hidden layer, 1 output layer.")
print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
print(80*'-')
print(MLNN.dlnn(df))

# print(80*'-')
# print(f"Deep Neural Network: 1 input layer, 1 hidden layer, 1 output layer.")
# print(f"Binary clasifier identifying bullish or bearish signals, 0, 1, or -1.")
# print(80*'-')
# print(MLNN.dlnn(df, 2))

print(80*'-')
print(f"Support Vector Classification")
print(f"Binary clasifier identifying bullish signals, 0 or 1 only.")
print(80*'-')
print(MLNN.svc(df))

