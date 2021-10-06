import pandas as pd
# from pathlib import Path
# import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
# import sqlalchemy
# import h5py
# import hvplot.pandas
# import bokeh
# from holoviews.plotting.links import RangeToolLink
# import utils.helpful_functions as hf
# from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from numpy import array
import numpy as np
import hvplot.pandas


# engine  = sqlalchemy.create_engine(hf.db_connection_string)
# inspector = sqlalchemy.inspect(engine)
# table_names = inspector.get_table_names()

def mlnn(df, output_nodes=None):
    # Create a list of categorical variables 
    categorical_variables = ['Trade Signal']
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)
    # Encode the categorcal variables using OneHotEncoder
    encoded_data = enc.fit_transform(df[categorical_variables])
    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )
    encoded_df.rename(columns={'Trade Signal_-1.0': 'Bearish', 'Trade Signal_0.0': 'None', 'Trade Signal_1.0':'Bullish'}, inplace=True)
    # encoded_df.drop(columns='None', inplace=True)
    
    # Define the features set X and the target set y
    if output_nodes == 2:
        y = encoded_df[['Bearish', 'None', 'Bullish']]
    else:
        y = encoded_df['Bullish']
        output_nodes = 1
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
    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer1 =  int(round((number_input_features + output_nodes)/2, 0))

    # Create the Sequential model instance
    nn = Sequential()
    # Add the first hidden layer
    nn.add(Dense(units=hidden_nodes_layer1, activation="relu", input_dim=number_input_features))
    # Add the output layer to the model specifying the number of output neurons and activation function
    nn.add(Dense(units=output_nodes, activation="sigmoid"))
    # Display the Sequential model summary
    nn.summary()

    # Compile the Sequential model
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Fit the model using 50 epochs and the training data
    model = nn.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    print("Model Results")
    # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=True)
    # Display the model loss and accuracy results
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    y_pred = nn.predict(X_test_scaled)

    # print(y_test)
    # print(X_test_scaled)
    # print(y_pred)
    y_pred_df = pd.DataFrame(y_pred)
    # print(y_pred_df.head())
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)



    results = pd.concat([y_test_df, y_pred_df], axis=1)
    # print(results.head())
    results.rename(columns={'Bullish': 'Actual', 0: 'Predictions'}, inplace=True)
    # results.to_csv(Path('./Resources/results.csv'))
    results['Predictions'] = results['Predictions'].apply(lambda x: int(round(x, 0)))
    results['Actual'] = results['Actual'].apply(lambda x: int(round(x, 0)))

    # print(results.value_counts())

    # print(results)
    # print(results['Actual'].value_counts())
    # print(results['Predictions'].value_counts())
    cm = confusion_matrix(results['Actual'], results['Predictions'])
    cm_df = pd.DataFrame(cm)
    print(cm_df)

    # print(cm_df)
    cr = classification_report(results['Actual'], results['Predictions'], zero_division='warn')
    print(cr)

    cm_df.to_csv('./Resources/mlnn_confusion_matrix.txt')
    with open('./Resources/mlnn_classification_report.txt', 'w') as f:
        f.write(cr)    
    return 
    

def dlnn(df, output_nodes=None):
    # Create a list of categorical variables 
    categorical_variables = ['Trade Signal']
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)
    # Encode the categorcal variables using OneHotEncoder
    encoded_data = enc.fit_transform(df[categorical_variables])
    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )
    encoded_df.rename(columns={'Trade Signal_-1.0': 'Bearish', 'Trade Signal_0.0': 'None', 'Trade Signal_1.0':'Bullish'}, inplace=True)
    encoded_df.drop(columns='None', inplace=True)
    
    # Define the features set X and the target set y
    if output_nodes == 2:
        y = encoded_df[['Bullish', 'Bearish']]
    else:
        y = encoded_df['Bullish']
        output_nodes = 1
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
    # Define the number of hidden nodes for the first and second hidden layer
    hidden_nodes_layer1 =  int(round((number_input_features +output_nodes)/2, 0))
    hidden_nodes_layer2 =  int(round((hidden_nodes_layer1 + output_nodes)/2, 0))
    # Create the Sequential model instance
    nn = Sequential()
    # Add the first hidden layer
    nn.add(Dense(units=hidden_nodes_layer1, activation="relu", input_dim=number_input_features))
    # Add the second hidden layer
    nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))
    # Add the output layer to the model specifying the number of output neurons and activation function
    nn.add(Dense(units=output_nodes, activation="sigmoid"))
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

    y_pred = nn.predict(X_test_scaled)
    y_pred_df = pd.DataFrame(y_pred)
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)

    results = pd.concat([y_test_df, y_pred_df], axis=1)
    results.rename(columns={'Bullish': 'Actual', 0: 'Predictions'}, inplace=True)
    results['Predictions'] = results['Predictions'].apply(lambda x: int(round(x, 0)))
    results['Actual'] = results['Actual'].apply(lambda x: int(round(x, 0)))

    # print(results)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred, zero_division='warn'))
    cm = confusion_matrix(results['Actual'], results['Predictions'])
    cm_df = pd.DataFrame(cm)
    print(cm_df)

    # print(cm_df)
    cr = classification_report(results['Actual'], results['Predictions'], zero_division='warn')
    print(cr)

    cm_df.to_csv('./Resources/dlnn_confusion_matrix.txt')
    with open('./Resources/dlnn_classification_report.txt', 'w') as f:
        f.write(cr)    
    return 


def svc(df):
    # Create a list of categorical variables 
    categorical_variables = ['Trade Signal']
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)
    # Encode the categorcal variables using OneHotEncoder
    encoded_data = enc.fit_transform(df[categorical_variables])
    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(
        encoded_data,
        columns = enc.get_feature_names(categorical_variables)
    )
    encoded_df.rename(columns={'Trade Signal_-1.0': 'Bearish', 'Trade Signal_0.0': 'None', 'Trade Signal_1.0':'Bullish'}, inplace=True)
    encoded_df.drop(columns='None', inplace=True)
    
    # Define the features set X and the target set y
    y = encoded_df['Bullish']
    X = df.drop(columns=['Trade Signal'])
    # y = df["Trade Signal"]

    model = SVC(kernel='linear')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # print(X.tail())
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({"Predictions": y_pred, "Actual":y_test}).reset_index(drop=True)

    # print(results)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred, zero_division='warn'))

    cm = confusion_matrix(results['Actual'], results['Predictions'])
    cm_df = pd.DataFrame(cm)
    print(cm_df)

    # print(cm_df)
    cr = classification_report(results['Actual'], results['Predictions'], zero_division='warn')
    print(cr)

    cm_df.to_csv('./Resources/svc_confusion_matrix.csv')
    with open('./Resources/svc_classification_report.txt', 'w') as f:
        f.write(cr)    

    return 


#Function to call to the main.py for lstm
def lstm(df):
    #defining the dataframe for testing and the targeted results
    y = df['Trade Signal']
    scaler=MinMaxScaler(feature_range=(0,1))
    y_scaled =scaler.fit_transform(array(y).reshape(-1,1))
        
    #Setting the training percentage, and valuing the information for training vs testing
    training_pct = 0.60
    training_size=int(len(y_scaled)*training_pct) 
    test_size=len(y_scaled)-training_size 
    train_data,test_data=y_scaled[0:training_size,:],y_scaled[training_size:len(y_scaled),:1]
        
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return array(dataX), array(dataY)

    #Setting up the timesteps
    time_step = 5
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    #Reshaping the data to pull in and match the dimensions of LSTM for Machine Learning
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # def create_dataset(dataset, time_step=1):
    #     dataX, dataY = [], []
    #     for i in range(len(dataset)-time_step-1):
    #         a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
    #         dataX.append(a)
    #         dataY.append(dataset[i + time_step, 0])
    #     return array(dataX), array(dataY)

    #Defining layers and features for the ML model to process the information
    neurons = 50
    timesteps = time_step #referenced above in reshaping block (currently 100)
    data_dimension = 1
    dense_layer = 1

    #LSTM Model Call, and pull of different layers to run RNN
    model=Sequential()
    model.add(LSTM(neurons,input_shape=(timesteps, data_dimension),return_sequences=True))
    model.add(LSTM(neurons,return_sequences=True))
    model.add(LSTM(neurons))
    model.add(Dense(dense_layer))
    model.compile(loss='mean_squared_error',optimizer='adam')

    #Review the Layers
    model.summary()

    #Fit the model
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=0)

    #Prediction and Performance Metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
        
    #Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
        
    #Calculate RMSE performance metrics - Root Mean Square Error, this is for the data set comparison
    lstm_RMSE = math.sqrt(mean_squared_error(y_train,train_predict))


    # shift train predictions for plotting
    look_back = timesteps #this is your timesteps from earlier
    trainPredictPlot = np.empty_like(y_scaled)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        
    #shift test predictions for plotting
    testPredictPlot = np.empty_like(y_scaled)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(y_scaled)-1, :] = test_predict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(y_scaled))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


    #converting the information to a list that has been scaled from 0 - 1 from earlier
    test_values = test_size - timesteps

    x_input=test_data[test_values:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next 3 Ticks
    lst_output=[]
    n_steps=time_step
    i=0
    prediction_days = 3

    while(i<prediction_days): #this predicts the next 30 days
        
        if(len(temp_input)>timesteps): 
            #print(temp_input)
            x_input=array(temp_input[1:])
            print("{} price input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} price output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
            

    tick_new=np.arange(1,timesteps + 1)
    tick_pred=np.arange(timesteps + 1,timesteps + 1 + prediction_days)

    #Plotting the information with the prediction
    tick_value = len(y_scaled) - timesteps

    prediction_df = y_scaled.tolist()
    prediction_df.extend(lst_output)
    plt.plot(prediction_df [tick_value:])


    prediction_df = scaler.inverse_transform(prediction_df).tolist()

    return lstm_RMSE
