import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
import matplotlib
import os
matplotlib.use("TkAgg")
warnings.filterwarnings('ignore')


INPUT_CSV = '../../data/raw/spy_volatility_data.csv'
OUTPUT_CSV = '../../data/results/garch_optimized_predictions.csv'

df = pd.read_csv(INPUT_CSV, parse_dates=True, index_col='Date')

#train/test
split_date = '2023-01-01'
train_df = df[df.index < split_date]
test_df = df[df.index >= split_date]

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

print("\nGenerating dynamic forecasts with the optimized GJR-GARCH(1,1) skew-t model...")

# hyper hyper parameters
REFIT_FREQUENCY = 4  # refit every 4 days (instead of everyday)
MIN_WINDOW_SIZE = 252 * 3  # use 5 years of data for each prediction

predictions = []
current_fitted_model = None

for i in tqdm(range(len(test_df))):
    total_history_size = len(train_df) + i
    start_idx = max(0, total_history_size - MIN_WINDOW_SIZE)
    history = df['log_return'].iloc[start_idx:total_history_size] * 100

    if current_fitted_model is None or i % REFIT_FREQUENCY == 0:
        model = arch_model(history, vol='Garch', p=1, q=1, o=1, dist='skewt')
        current_fitted_model = model.fit(disp='off', show_warning=False)

    forecast = current_fitted_model.forecast(horizon=1, reindex=False)
    variance_forecast = forecast.variance.iloc[0, 0]
    volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100
    predictions.append(volatility_forecast)

print("Forecasting complete.")

garch_prediction_series = pd.Series(predictions, index=test_df.index, name='garch_prediction')

#visualize and matplotlib stuff
predictions_df = pd.DataFrame({
    'realized_volatility':test_df['realized_volatility'],
    'garch_prediction':garch_prediction_series
})

#calc metrics
rmse = np.sqrt(np.mean((predictions_df['garch_prediction'] - predictions_df['realized_volatility']) ** 2))
mae = np.mean(np.abs(predictions_df['garch_prediction'] - predictions_df['realized_volatility']))
correlation = predictions_df['realized_volatility'].corr(predictions_df['garch_prediction'])
r_squared = r2_score(predictions_df['realized_volatility'], predictions_df['garch_prediction'])


# Plot the results
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 7))
plt.plot(predictions_df.index, predictions_df['realized_volatility'],
         label='Actual Realized Volatility', color='blue', alpha=0.7, linewidth=2)
plt.plot(predictions_df.index, predictions_df['garch_prediction'],
         label='Optimized GJR-GARCH(1,1) Skew-t Forecast', color='purple', linestyle='--', linewidth=1.5)
plt.title('Optimized GARCH Dynamic Forecast vs. Actual\n(GJR-GARCH(1,1) with Skewed-t Distribution)')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.plot([], [], ' ', label=f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, Corr: {correlation:.3f}")
plt.legend(loc = "upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

residuals = predictions_df['realized_volatility'] - predictions_df['garch_prediction']


"""
residual analysis
"""
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Hybrid LSTM Model - Residual Analysis', fontsize=16)

#time series
axes[0, 0].plot(predictions_df.index, residuals, color='red', alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Daily Residuals Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals (Actual - Predicted)')
axes[0, 0].grid(True, alpha=0.3)

#histogram
axes[0, 1].hist(residuals, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

#q-q
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
axes[1, 0].grid(True, alpha=0.3)

#residuals agains predicted
axes[1, 1].scatter(predictions_df['garch_prediction'], residuals, alpha=0.6, color='purple')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Residuals vs Predicted Values')
axes[1, 1].set_xlabel('Predicted Volatility')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#print metrics

print("\n" + "=" * 60)
print("OPTIMIZED GARCH MODEL PERFORMANCE")
print("=" * 60)
print(f"Model Specification: GJR-GARCH(1,1) with Skewed-t Distribution")
print(f"Parameters: vol='Garch', p=1, q=1, o=1, dist='skewt'")
print(f"Refit Frequency: Every {REFIT_FREQUENCY} days")
print(f"Minimum Window: {MIN_WINDOW_SIZE} observations ({MIN_WINDOW_SIZE / 252:.1f} years)")
print("-" * 60)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Correlation: {correlation:.4f}")
print(f"R-Squared: {r_squared:.4f}")
print("=" * 60)

#save
predictions_df.to_csv(OUTPUT_CSV)
print(f"\nOptimized GARCH predictions saved to '{OUTPUT_CSV}'")
