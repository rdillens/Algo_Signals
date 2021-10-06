#imports
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import array

#Setting the dataset up for the LSTM model



#Function to call to the main.py for lstm
def lstm_df(df):


    #defining the dataframe for testing and the targeted results
    y = df['Trade Signal']
    scaler=MinMaxScaler(feature_range=(0,1))
    y_scaled =scaler.fit_transform(np.array(y).reshape(-1,1))
        
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
        return np.array(dataX), np.array(dataY)

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
    #     return np.array(dataX), np.array(dataY)

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
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=True)

    #Prediction and Performance Metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
        
    #Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
        
    #Calculate RMSE performance metrics - Root Mean Square Error, this is for the data set comparison
    lstm_RMSE = math.sqrt(mean_squared_error(y_train,train_predict))


    # shift train predictions for plotting
    look_back= timesteps #this is your timesteps from earlier
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
            x_input=np.array(temp_input[1:])
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