from flask import Flask, render_template, request
from patterns import patterns
import yfinance as yf
import os
import pandas as pd
import talib    
import csv

app = Flask(__name__)

@app.route('/')
def index():
    pattern = request.args.get('pattern', None)
    stocks = {}
    with open('datasets/companies.csv') as f:
        for row in csv.reader(f):
            stocks[row[0]] = {'industry': row[2]}
    print(stocks)
    if pattern: 
        for filename in os.listdir('datasets/daily'): 
            df = pd.read_csv('datasets/daily/{}'.format(filename))
            pattern_function = getattr(talib, pattern)
            symbol = filename.split('.')[0]
            try:
                result = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
                # print(result)
                last = result.tail(1).values[0]
                # print(last)
                if last > 0:
                    stocks[symbol][pattern] = 'bullish'
                elif last < 0:
                    stocks[symbol][pattern] = 'bearish'
                else:
                    stocks[symbol][pattern] = None
                    
            except:
                pass
    return render_template('index.html', patterns=patterns, stocks=stocks, current_pattern=pattern)

@app.route('/snapshot')
def snapshot():
    # pulled in the SP500 csv file from Remy's create_csv.ipynb
    with open('datasets/companies.csv') as f:
        # loop through all companies listed in the csv and pulled 8 months worth of daily data
        companies = f.read().splitlines()
        for company in companies:
            symbol = company.split(',')[0]
            df = yf.download(symbol, start="2020-09-28", end="2021-09-28")
            df.to_csv('datasets/daily/{}.csv'.format(symbol))
    return {
        'code': 'success'
    }

