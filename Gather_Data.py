import yfinance as yf
import pandas as pd
import numpy as np

#config
TICKER = 'SPY'
START_DATE = '2010-01-01'
VOLATILITY_WINDOW = 21 # Corresponds to one trading month
OUTPUT_CSV = f'data/raw/spy_volatility_data.csv'


#download historical data
print(f"Downloading SPY data from {START_DATE}...")
df = yf.download("SPY", start=START_DATE, auto_adjust=True, multi_level_index= False)
print("Download complete.")


# calculate log returns - they are better for time-series analysis
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

# calc realized volatility
# use the standard deviation of log returns over our window.
# multiply by sqrt(252) to annualize it (252 trading days in a year).
df['realized_volatility'] = df['log_return'].rolling(window=VOLATILITY_WINDOW).std() * np.sqrt(252)


print(f"Original data length: {len(df)}")
df.dropna(inplace=True)
print(f"Data length after dropping NaNs: {len(df)}")

#save
df.to_csv(OUTPUT_CSV)
print(f"\nData successfully cleaned and saved to '{OUTPUT_CSV}'")
print("\nFinal DataFrame head:")
print(df.head())



