import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder

def svc(df):
    # print(df['Trade Signal'].value_counts())

    # df = df.copy()
    # X = df.drop(columns=['Trade Signal'])
    # y = df["Trade Signal"]

    # model = SVC(kernel='linear')

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # print(X.tail())
    # scaler = StandardScaler()
    # X_scaler = scaler.fit(X_train)
    # X_train_scaled = X_scaler.transform(X_train)
    # X_test_scaled = X_scaler.transform(X_test)

    # # model = SVC(kernel='linear')
    # model.fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)
    # results = pd.DataFrame({"Predictions": y_pred, "Actual":y_test}).reset_index(drop=True)

    # print(results)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred, zero_division='warn'))

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
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division='warn')

    return results, conf_matrix, class_report
