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

def dataframe_SVC(df):
    print(df['Trade Signal'].value_counts())

    df = df.copy()
    X = df.drop(columns=['Trade Signal'])
    y = df["Trade Signal"]

    model = SVC(kernel='linear')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print(X.tail())
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results = pd.DataFrame({"Predictions": y_pred, "Actual":y_test}).reset_index(drop=True)

    print(results)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division='warn'))

    return 
