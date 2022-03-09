# **AlgoTrader**
**This software is not intended to be used as financial advice!**

**Created by The Four Headless Horseman**
> *"The algo-trader you set and forget."*

---
## **Executive Summary**
There are a multitude of trading strategies, indicators, and methods to enter the market in an effective way. We, at The Four Headless Horsemen (T4HH), want to allow you, as a user, to personally select or utilize a pre-selected strategy for our algorithm to make winning trades from your selections. The ideology is that we ultimately want you, as a user, to have full control of trades to match (if not exceed) the market in performance based on your personal preferences, while providing a unique user experience that will ultimately lead to a “Set it and Forget it” trading profile.

---
## **Problems to Solve**
1. Not confident in how trades work in the market, not having the ability or information to make a trade.
2. Fear of making the wrong choices, not being able to detect trades in an effective manner.
3. Fear of missing out and trying to enter the market during the fad, entering the market at the wrong time.
4. Time to manage and maintain a trading profile confidently, it takes time to do.

---
## **Our Objective**
We want you as a user to be able to select a short term trading strategy utilizing our algorithm and machine learning models to identify and execute winning trades based on your preferences without being managed (headless).

---
## **Model Selection**
We chose 4 models and compared their accruacy and recall in predicting market signals based on historical price data:

1. Support Vector Machine (SVM)
2. Neural Network (MLNN)
3. Deep-Learning Neural Network (DLNN)
4. Long Short Term Memory (LSTM)

---
## **Our Process**
[Click here to read more about our project plan](./summary.md)

---
## **Technologies Used**

```python
import questionary
import fire
import pandas as pd
import sqlalchemy
from pathlib import Path
import shelve
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from holoviews.plotting.links import RangeToolLink
import matplotlib.pyplot as plt
import math
from numpy import array
import numpy as np
from datetime import date
import mplfinance as mpf
```

## **Steps to Operate the Program**

1. To initiate the programs please ensure all libraries are included and installed (see above "Technologies Used.")
2. User must create a ```.env``` file with a finnhub api key [(Obtained here)](https://finnhub.io/) and save it in the root folder in the following format:
```shell
FINNHUB_API_KEY="XXXXXXXXXXXXXXXXXXXX"
```
3. Start the program using a CLI by typing the following:
```shell
python analyze_ticker.py
```
4. You will be prompted with a question in the CLI asking: 
```shell
What stock ticker should I look up?
```
5. Once prompted, input a ticker into the CLI (all caps).

6. Once the stock is selected, you will be prompted with a list of candlestick patterns you are welcome to choose. For more information about various candlestick patterns please check out [this link](https://patternswizard.com/)

7. You will be prompted with overlap studies that you are welcome to choose. For more information about overlap studies, please check out [this link](http://www.tadoc.org/index.htm)

8. You will be prompted with momentum indicators you are welcome to choose. For more information about mometum indicators, please check out [this link](http://www.tadoc.org/index.htm)


9. You will be prompted with volatility indicators that you are welcome to choose. For more information about volatility indicators, please check out [this link](http://www.tadoc.org/index.htm)
 

10. Lastly, you will be prompted if you would like to save to the program database. From there, the 4 models will run and output the results. 

## **Contributors**

This program was originally started as a student project
-[Andrew Au](https://github.com/AndrewAu42)
-[Billy Bishop](https://github.com/billybishop21)
-[Scott Slusher](https://github.com/scottslusher) 
-Remy Dillenseger

## **License**

**This software is not intended to be used as financial advice!**

Copyright (c) 2021

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---


