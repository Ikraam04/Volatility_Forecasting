import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

## 1. Configuration
# ------------------
INPUT_CSV = '../../data/raw/spy_volatility_data.csv'
OUTPUT_DIR = "tensors"
FEATURE_COLUMNS = ['realized_volatility', 'Volume']
SEQUENCE_LENGTH = 45
TRAIN_TEST_SPLIT_DATE = '2023-01-01'

## 2. Load and Prepare Data
# --------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=True, index_col='Date')
feature_df = df[FEATURE_COLUMNS]
train_data = feature_df[feature_df.index < TRAIN_TEST_SPLIT_DATE]
test_data = feature_df[feature_df.index >= TRAIN_TEST_SPLIT_DATE]

## 3. Scale the Data
# -------------------
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_LSTM.pkl")
print("2-feature scaler saved.")

## 4. Create Sequences
# ---------------------
def create_sequences(data, seq_length):
    target_col_index = 0
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, target_col_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data_scaled, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(test_data_scaled, SEQUENCE_LENGTH)

## 5. Convert and Save Tensors
# -----------------------------
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float().view(-1, 1)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().view(-1, 1)


torch.save(X_train, f"{OUTPUT_DIR}/X_train_LSTM.pt")
torch.save(y_train, f"{OUTPUT_DIR}/y_train_LSTM.pt")
torch.save(X_test,  f"{OUTPUT_DIR}/X_test_LSTM.pt")
torch.save(y_test,  f"{OUTPUT_DIR}/y_test_LSTM.pt")
print("2-feature data tensors have been saved.")