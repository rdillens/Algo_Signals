import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import sqlalchemy
# import h5py
import hvplot.pandas
# import bokeh
from holoviews.plotting.links import RangeToolLink
from datetime import date
import matplotlib.pyplot as plt
import mplfinance as mpf


engine  = sqlalchemy.create_engine(hf.db_connection_string)

inspector = sqlalchemy.inspect(engine)
table_names = inspector.get_table_names()

def dataframe(dt_start, dt_end, df):
    plot_width = 1400
    plot_date = dt_end
    plot_start = dt_start
    plot_end = dt_end 
    # plot_df = df.loc[plot_start:plot_end,:].reset_index()
    # plot_df = indicators_df.loc[plot_date,:].reset_index()
    # plot_df = indicators_df.iloc[-3000:,:].reset_index()
    df.tail()
    
    df.rename(
        columns={
            'Datetime': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        },
        inplace=True
    )
    mpf.plot(df, type="candle")

    print(df['Trade Signal'].value_counts())

    X = df.drop(columns='Trade Signal')
    y = df['Trade Signal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Define the the number of inputs (features) to the model
    number_input_features = len(list(X.columns))

    # Define the number of neurons in the output layer
    number_output_neurons = 1

    # Define the number of hidden nodes for the first and second hidden layer
    hidden_nodes_layer1 =  int(round((number_input_features + number_output_neurons)/2, 0))
    hidden_nodes_layer2 =  int(round((hidden_nodes_layer1 + number_output_neurons)/2, 0))

    # Create the Sequential model instance
    nn = Sequential()

    # Add the first hidden layer
    nn.add(Dense(units=hidden_nodes_layer1, activation="relu", input_dim=number_input_features))

    # Add the second hidden layer
    nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))

    # Add the output layer to the model specifying the number of output neurons and activation function
    nn.add(Dense(units=number_output_neurons, activation="sigmoid"))

    # Display the Sequential model summary
    print(nn.summary())

    # Compile the Sequential model
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model using 50 epochs and the training data
    model = nn.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=True)

    # Display the model loss and accuracy results
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    