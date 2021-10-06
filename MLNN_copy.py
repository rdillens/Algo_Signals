import pandas as pd
# from pathlib import Path
# import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import sqlalchemy
# import h5py
# import hvplot.pandas
# import bokeh
# from holoviews.plotting.links import RangeToolLink
import utils.helpful_functions as hf
from sklearn.decomposition import PCA

engine  = sqlalchemy.create_engine(hf.db_connection_string)

inspector = sqlalchemy.inspect(engine)
table_names = inspector.get_table_names()

def mlnn(df):
    # # Create a list of categorical variables 
    # categorical_variables = ['Trade Signal']
    # # Create a OneHotEncoder instance
    # enc = OneHotEncoder(sparse=False)
    # # Encode the categorcal variables using OneHotEncoder
    # encoded_data = enc.fit_transform(df[categorical_variables])
    # # Create a DataFrame with the encoded variables
    # encoded_df = pd.DataFrame(
    #     encoded_data,
    #     columns = enc.get_feature_names(categorical_variables)
    # )
    # encoded_df.rename(columns={'Trade Signal_-1.0': 'Bearish', 'Trade Signal_0.0': 'None', 'Trade Signal_1.0':'Bullish'}, inplace=True)
    # encoded_df.drop(columns='None', inplace=True)
    
    # Define the features set X and the target set y
    y = df['Trade Signal'].copy()
    X = df.drop(columns=['Trade Signal'])
    # Split the preprocessed data into a training and testing dataset
    # Assign the function a random_state equal to 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)    

    # Create a StandardScaler instance
    scaler = StandardScaler()
    # Fit the scaler to the features training dataset
    X_scaler = scaler.fit(X_train)
    # Fit the scaler to the features training dataset
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Define the the number of inputs (features) to the model
    number_input_features = len(list(X.columns))
    # Define the number of neurons in the output layer
    number_output_neurons = 1
    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer1 =  int(round((number_input_features + number_output_neurons)/2, 0))

    # Create the Sequential model instance
    nn = Sequential()
    # Add the first hidden layer
    nn.add(Dense(units=hidden_nodes_layer1, activation="relu", input_dim=number_input_features))
    # Add the output layer to the model specifying the number of output neurons and activation function
    nn.add(Dense(units=number_output_neurons, activation="sigmoid"))
    # Display the Sequential model summary
    nn.summary()

    # Compile the Sequential model
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Fit the model using 50 epochs and the training data
    model = nn.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    # print("Model Results")
    # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=True)
    # Display the model loss and accuracy results
    # print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    return model_loss, model_accuracy