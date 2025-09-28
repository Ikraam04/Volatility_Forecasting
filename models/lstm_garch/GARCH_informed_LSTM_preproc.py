import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib

#config
HYBRID_DATA_CSV = "GARCH_LSTM_feature.csv"
ORIGINAL_DATA_CSV = '../../data/raw/spy_volatility_data.csv' # We need this for the volume
TARGET_COLUMN_INDEX = 0
SEQUENCE_LENGTH = 45
TRAIN_TEST_SPLIT_DATE = '2023-01-01'
OUTPUT_DIR = "tensors"


hybrid_df = pd.read_csv(HYBRID_DATA_CSV, parse_dates=True, index_col='Date')
original_df = pd.read_csv(ORIGINAL_DATA_CSV, parse_dates=True, index_col='Date')

#add volume as an input feature
full_feature_df = hybrid_df.join(original_df['Volume'])
full_feature_df.dropna(inplace=True)

print("--- 3-Feature DataFrame Head ---")
print(full_feature_df.head())

#split

train_df = full_feature_df[full_feature_df.index < TRAIN_TEST_SPLIT_DATE]
test_df = full_feature_df[full_feature_df.index >= TRAIN_TEST_SPLIT_DATE]

#fit and save scaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_df)
test_data_scaled = scaler.transform(test_df)


joblib.dump(scaler, f"{OUTPUT_DIR}/scalar_LSTM_GARCH.pkl")
print(f"saved to {OUTPUT_DIR}/scalar_LSTM_GARCH.pkl")

#get sequences (LSTM's only understand 3d data)

def create_sequences(data, seq_length, target_col_index):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, target_col_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data_scaled, SEQUENCE_LENGTH, TARGET_COLUMN_INDEX)
X_test, y_test = create_sequences(test_data_scaled, SEQUENCE_LENGTH, TARGET_COLUMN_INDEX)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(f"training shape (X, y): {X_train.shape}, {y_train.shape}")
print(f"Testing shape (X, y): {X_test.shape}, {y_test.shape}")

#conv to tensors and save

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

torch.save(X_train, f"{OUTPUT_DIR}/X_train_GARCH_LSTM.pt")
torch.save(y_train, f"{OUTPUT_DIR}/Y_train_GARCH_LSTM.pt")
torch.save(X_test, f"{OUTPUT_DIR}/X_test_GARCH_LSTM.pt")
torch.save(y_test, f"{OUTPUT_DIR}/Y_test_GARCH_LSTM.pt")
print(f"tensors saved to {OUTPUT_DIR}")