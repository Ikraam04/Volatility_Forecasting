import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#config
RESULTS = "data/results/"

GARCH_CSV = f"{RESULTS}garch_optimized_predictions.csv"
LSTM_CSV = f"{RESULTS}LSTM_predictions.csv"
GARCH_LSTM_CSV = f"{RESULTS}GARCH_LSTM_predictions.csv"

#load all our predictions
garch_df = pd.read_csv(GARCH_CSV, parse_dates=True, index_col='Date')
lstm_2f_df = pd.read_csv(LSTM_CSV, parse_dates=True, index_col='Date')
garch_inf_lstm = pd.read_csv(GARCH_LSTM_CSV, parse_dates=True, index_col='Date')
#merge em
comparison_df = garch_df.copy()
comparison_df = comparison_df.join(lstm_2f_df['lstm_prediction'])
comparison_df = comparison_df.join(garch_inf_lstm['lstm_garch_prediction'])

#for safe keeping
comparison_df.dropna(inplace=True)


#re-calc metrics
models = {
    'GARCH':'garch_prediction',
    'LSTM (2-Feature)':'lstm_prediction',
    'Hybrid LSTM':'lstm_garch_prediction'
}

metrics_results = []

for model_name, pred_col in models.items():
    realized = comparison_df['realized_volatility']
    predicted = comparison_df[pred_col]

    rmse = np.sqrt(np.mean((predicted - realized) ** 2))
    mae = np.mean(np.abs(predicted - realized))
    corr = realized.corr(predicted)
    r2 = r2_score(realized, predicted)

    metrics_results.append({
        'Model':model_name,
        'RMSE':f"{rmse:.4f}",
        'MAE':f"{mae:.4f}",
        'Correlation':f"{corr:.4f}",
        'R-Squared':f"{r2:.4f}"
    })

#print the dict
metrics_df = pd.DataFrame(metrics_results)
print("final metrics: ")
print(metrics_df.to_string(index=False))

#plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(16, 8))

#plot the actual
plt.plot(comparison_df.index, comparison_df['realized_volatility'],
         label='Actual Volatility', color='blue', alpha=0.8, linewidth=2.5)

# plot the forecast models
plt.plot(comparison_df.index, comparison_df['garch_prediction'],
         label='GARCH', color='purple', linestyle=':', linewidth=1.5)
plt.plot(comparison_df.index, comparison_df['lstm_prediction'],
         label='LSTM', color='orange', linestyle='-.', linewidth=1.5)
plt.plot(comparison_df.index, comparison_df['lstm_garch_prediction'],
         label='GARCH-inf LSTM', color='green', linestyle='--', linewidth=1.5)

plt.title('Model Comparison: Volatility Forecasting', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Annualized Volatility', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("forecast_comparison.png")
plt.show()
