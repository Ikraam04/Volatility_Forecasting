import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

## 1. Configuration & Hyperparameters
# ------------------------------------
INPUT_DIM = 2
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_DIM = 1
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 45

##config data
INPUT_DIR = 'tensors'
OUTPUT_CSV = '../../data/results/LSTM_predictions.csv'
RAW_DATA = "../../data/raw/spy_volatility_data.csv"

## 2. Load Prepared 2-Feature Data
# ---------------------------------
print("Loading pre-processed 2-feature data...")
X_train = torch.load(f"{INPUT_DIR}/X_train_LSTM.pt")
y_train = torch.load(f"{INPUT_DIR}/y_train_LSTM.pt")
X_test  = torch.load(f"{INPUT_DIR}/X_test_LSTM.pt")
y_test  = torch.load(f"{INPUT_DIR}/y_test_LSTM.pt")
scaler  = joblib.load(f"{INPUT_DIR}/scaler_LSTM.pkl")
print("Data loaded successfully.")

## 3. Define the LSTM Model
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(INPUT_DIM, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

## 4. Train the Model
# --------------------
print("\nStarting 2-feature model training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')
print("Training complete.")

## 5. Evaluate the Model (with Correct Inverse Scaling)
# ----------------------------------------------------
model.eval()
with torch.no_grad():
    test_predictions_scaled = model(X_test)

dummy_array_pred = np.zeros((len(test_predictions_scaled), INPUT_DIM))
dummy_array_pred[:, 0] = test_predictions_scaled.squeeze().numpy()
test_predictions = scaler.inverse_transform(dummy_array_pred)[:, 0]

dummy_array_actual = np.zeros((len(y_test), INPUT_DIM))
dummy_array_actual[:, 0] = y_test.squeeze().numpy()
y_test_actual = scaler.inverse_transform(dummy_array_actual)[:, 0]

## 6. Create Predictions DataFrame
# ---------------------------------
vol_df = pd.read_csv(RAW_DATA, parse_dates=True, index_col='Date')
test_dates = vol_df[vol_df.index >= '2023-01-01'].index
predictions_df = pd.DataFrame({'realized_volatility': y_test_actual.flatten()}, index=test_dates[SEQUENCE_LENGTH:])
predictions_df['lstm_prediction'] = test_predictions.flatten()

## 7. Visualize & Save
# ---------------------
plt.figure(figsize=(14, 7))
plt.plot(predictions_df.index, predictions_df['realized_volatility'], label='Actual Volatility')
plt.plot(predictions_df.index, predictions_df['lstm_prediction'], label='2-Feature LSTM Forecast', linestyle='--')
plt.title('2-Feature LSTM Volatility Forecast vs. Actual')
plt.legend()
plt.show()

residuals = predictions_df['realized_volatility'] - predictions_df['lstm_prediction']

# Create a comprehensive residual analysis plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Hybrid LSTM Model - Residual Analysis', fontsize=16)

# 1. Time series plot of residuals
axes[0, 0].plot(predictions_df.index, residuals, color='red', alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Daily Residuals Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residuals (Actual - Predicted)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram of residuals
axes[0, 1].hist(residuals, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot (residuals vs normal distribution)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals vs Predicted values (heteroscedasticity check)
axes[1, 1].scatter(predictions_df['lstm_prediction'], residuals, alpha=0.6, color='purple')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Residuals vs Predicted Values')
axes[1, 1].set_xlabel('Predicted Volatility')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# (Metrics and saving code remains the same)
rmse = np.sqrt(np.mean((predictions_df['lstm_prediction'] - predictions_df['realized_volatility'])**2))
mae = np.mean(np.abs(predictions_df['lstm_prediction'] - predictions_df['realized_volatility']))
correlation = predictions_df['realized_volatility'].corr(predictions_df['lstm_prediction'])
r_squared = r2_score(predictions_df['realized_volatility'], predictions_df['lstm_prediction'])
print(f"\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, Correlation: {correlation:.4f}, R-Squared: {r_squared:.4f}")
predictions_df.to_csv(OUTPUT_CSV)
print(f"\nPredictions saved to '{OUTPUT_CSV}'")