import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas_datareader.data as web
import datetime
import yfinance as yf

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime.now()

stock = 'TSLA'

df = yf.download(
    tickers=stock, 
    period='7d',
    interval='1m',
    auto_adjust=True,
    prepost=True,
)

app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(children='''
        symbol to graph:
    '''),

    dcc.Input(id='input', value='', type='text'),
    html.Div(id='output-graph')
])

@app.callback(
    Output(component_id='output_graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
)

def update_graph(input_data):
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime.now()
    df = yf.download(
        tickers=stock, 
        period='7d',
        interval='1m',
        auto_adjust=True,
        prepost=True,
    )

    return dcc.Graph(
        id='example-graph',
        figure={
            'data':[
                {'x':df.index, 'y':df.Close, 'type':'line', 'name':input_data},
            ],
            'layout':{
                'title':input_data
            }
        }
    )

if __name__ == '__main__':
    app.run_server(debug=True)

