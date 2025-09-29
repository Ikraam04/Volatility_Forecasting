import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#PARAMTERIC VaR CALCULATIONS!!

#config
GARCH_CSV = 'data/results/garch_optimized_predictions.csv'
LSTM_CSV = 'data/results/LSTM_predictions.csv'
GARCH_LSTM_CSV = 'data/results/GARCH_LSTM_predictions.csv'
RAW_DATA_CSV = 'data/raw/spy_volatility_data.csv'

# VaR parameters
CONFIDENCE_LEVEL = 0.95
PORTFOLIO_VALUE = 1_000_000  # Example: $1 million portfolio

#load
garch_df = pd.read_csv(GARCH_CSV, parse_dates=True, index_col='Date')
lstm_2f_df = pd.read_csv(LSTM_CSV, parse_dates=True, index_col='Date')
hybrid_lstm_df = pd.read_csv(GARCH_LSTM_CSV, parse_dates=True, index_col='Date')

#merge
comparison_df = garch_df.copy()
comparison_df = comparison_df.join(lstm_2f_df['lstm_prediction'])
comparison_df = comparison_df.join(hybrid_lstm_df['lstm_garch_prediction'])
comparison_df.dropna(inplace=True)

models = {
    'GARCH': 'garch_prediction',
    'LSTM (2-Feature)': 'lstm_prediction',
    'LSTM-inf-GARCH': 'lstm_garch_prediction'
}

#daily VaR
z_score = norm.ppf(CONFIDENCE_LEVEL)
print(f"\ncalculating VaR at {CONFIDENCE_LEVEL*100}% confidence level (Z-score: {z_score:.3f})")

for model_name, pred_col in models.items():
    daily_vol = comparison_df[pred_col] / np.sqrt(252) #de-annulaize
    var_col_name = f"{model_name.split(' ')[0].lower()}_var" #create a new column
    comparison_df[var_col_name] = PORTFOLIO_VALUE * daily_vol * z_score #cal VaR
    print(f"avg daily VaR for {model_name}: ${comparison_df[var_col_name].mean():,.2f}")


#
spy_df = pd.read_csv(RAW_DATA_CSV, parse_dates=True, index_col='Date')
spy_df['actual_loss'] = -spy_df['log_return'] * PORTFOLIO_VALUE
comparison_df = comparison_df.join(spy_df['actual_loss'])
comparison_df.dropna(inplace=True) # Drop any dates that might not align

#plot
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(16, 8))

# plot actual points
plt.plot(comparison_df.index, comparison_df['actual_loss'],
         label='Actual Daily Loss', color='red', alpha=0.4, linestyle='None', marker='o', markersize=4)

# plot VaR from all models
plt.plot(comparison_df.index, comparison_df['garch_var'],
         label='GARCH 95% VaR', color='purple', linestyle=':')
plt.plot(comparison_df.index, comparison_df['lstm_var'],
         label='LSTM (2-Feature) 95% VaR', color='orange', linestyle='-.')
plt.plot(comparison_df.index, comparison_df['lstm-inf-garch_var'],
         label='LSTM-inf GARCH 95% VaR', color='green', linestyle='--')

plt.title(f'Daily Value at Risk (VaR) vs. Actual Losses for a ${PORTFOLIO_VALUE:,.0f} Portfolio', fontsize=16)
plt.ylabel('Loss ($)')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.savefig("VaR_daily_loss_comparison.png")
plt.show()




print("\nbacktesting:")
expected_breach_rate = 1.0 - CONFIDENCE_LEVEL

for model_name, pred_col in models.items():
    var_col_name = f"{model_name.split(' ')[0].lower()}_var"
    breaches = comparison_df[comparison_df['actual_loss'] > comparison_df[var_col_name]]
    num_breaches = len(breaches)
    total_days = len(comparison_df)
    breach_rate = num_breaches / total_days

    print(f"\nmodel: {model_name}")
    print(f"breaches: {num_breaches} out of {total_days} days")
    print(f"breach rate: {breach_rate:.2%}")
    print(f"expected rate: {expected_breach_rate:.2%}")
