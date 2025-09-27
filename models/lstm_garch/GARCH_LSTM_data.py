import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm

#1 config
RAW_DATA = "../../data/raw/spy_volatility_data.csv"
OUTPUT_CSV = 'GARCH_LSTM_feature.csv'
split_date = '2023-01-01'

#some hyper - hyper params
REFIT_FREQUENCY = 1  # How often to refit the GARCH model (in days)
WINDOW_SIZE = 252 * 5  # Use a 3-year rolling window of data for forecasts
# ----------------------

#split
df = pd.read_csv(RAW_DATA, parse_dates=True, index_col='Date')
train_df = df[df.index < split_date]
test_df = df[df.index >= split_date]

#generate predictions for the training data
print("Generating in-sample GARCH predictions for the training period...")
train_returns = train_df['log_return'] * 100
model_train = arch_model(train_returns, vol='Garch', p=1, o=1, q=1, dist='skewt')
results_train = model_train.fit(disp='off')

in_sample_variance = results_train.conditional_volatility ** 2
in_sample_predictions = np.sqrt(in_sample_variance) * np.sqrt(252) / 100
in_sample_predictions.name = 'garch_prediction'
print("In-sample prediction generation complete.")

#run a GARCH model to get data for the testing period
print("\nGenerating out-of-sample GARCH predictions for the testing period...")
test_predictions_list = []
current_fitted_model = None

for i in tqdm(range(len(test_df))):
    #rolling win
    total_obs = len(train_df) + i
    start_idx = max(0, total_obs - WINDOW_SIZE)
    history = df['log_return'].iloc[start_idx:total_obs] * 100

    #refit every time (re-fit freq is 1)
    if current_fitted_model is None or i % REFIT_FREQUENCY == 0:
        model_test = arch_model(history, vol='Garch', p=1, o=1, q=1, dist='skewt')
        current_fitted_model = model_test.fit(disp='off')

    # Generate forecast using the latest fitted model
    forecast = current_fitted_model.forecast(horizon=1, reindex=False)
    variance_forecast = forecast.variance.iloc[0, 0]
    volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100
    test_predictions_list.append(volatility_forecast)

out_of_sample_predictions = pd.Series(test_predictions_list, index=test_df.index, name='garch_prediction')
print("Out-of-sample prediction generation complete.")

#combine the training and predictions
full_garch_LSTM_predictions = pd.concat([in_sample_predictions, out_of_sample_predictions])

hybrid_df = pd.DataFrame({
    'realized_volatility':df['realized_volatility'],
    'garch_prediction':full_garch_LSTM_predictions
})

hybrid_df.dropna(inplace=True)
hybrid_df.to_csv(OUTPUT_CSV)

print(f"\nHybrid dataset successfully created and saved to '{OUTPUT_CSV}'")
print("\nFinal DataFrame head:")
print(hybrid_df.head())