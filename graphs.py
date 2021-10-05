import matplotlib.pyplot as plt
import mplfinance as mpf

def plot_ohlc_graph(dt_start, dt_end, df):
    plot_width = 1400
    plot_date = dt_end
    plot_start = dt_start
    plot_end = dt_end 
    plot_df = df[-600:]
    # plot_df = indicators_df.loc[plot_date,:].reset_index()
    # plot_df = indicators_df.iloc[-3000:,:].reset_index()
    plot_df.head()
    plot_df.rename(
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
    mpf.plot(plot_df, type="candle", volume=True, show_nontrading=True)