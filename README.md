# Algo_Signals
Users choose a market product to track, then choose a model to fit its historical trade data. 

- Run create_csv.py
    - This step requires the yfinance package to be installed
---

# The Four Headless Horseman


## An AlgoTrader that allows the user to build / test / deploy their strategy without needing the coding background necessary to build from scratch


## Stock / Crypto Screener (@Remy D initial design)
- Allow the user to select which market they want to choose (stocks, or crypto currency)
- Once the market is selected we will screen the market to come up with the most volatile stocks/cryptos for each trading day to put through the algo


## Determine what is classified as a "winning trade" for the buy / sell in our historical dataset
- Are we looking for short term trades that exceed the fees we would pay to take the trade? aka anything that is remotely profitable
- OR, are we wanting only "home run" trades of 5% or more within a certain time frame


## Customer input technical indicators from a list (@Andrew initial design)
- The customer can select what technical indicators they would like the algo to consider when trading the stocks/cryptos
- These could be pulled through the "TA" package in Python or created by us as "modules" that can be called / plugged into the code
- The indicators would then be put into the dataset along with the stock historical price data


## Entry / Exit Strategy (@Scott Slusher initial design)
- This would allow the user to select between a fixed entry / exit or a dynamic entry / exit based off of risk tolerance
  - Example:
  - Fixed:
      - Entry: Invest $1000 or 1,000 shares regardless of price or volume
      - Exit: A stop loss set at 5% below entry with an exit set 10% above entry
  - Dynamic:
      - Entry: Kelly Criterion based off a formula that takes in the algo's P/L Ratio, Win %, Lose %, Capital available to trade, OR 10% of average trading volume over last 5           periods (whichever is smaller)
      - Exit: Based off the 2.5X ATR (Average True Range) of the stock for the given time period as a stop, then use support/resistant levels as "soft targets" for exits


## Build Algo
- Basic structure will be provided (train - test - split, etc)
- User will need to determine if they would like to tweak the amount of layers in the neural network and how many neurons per layer
    - this can also be "optimized" per Remy with a for loop to run several variations to determine the best performance


## Back Test
- Allow the algo to paper trade with live data (crypto 24/7/365, stocks during open hours or historical data from last trading day) to further confirm / deny if the algo is working


## Show all available metrics
- Allow the user to determine if they would like to Deploy the algo that was created, and/or restart at the beginning and build another algo

