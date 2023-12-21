# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import statsmodels.api as sm
from tqdm import tqdm
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Set variables
ticker = 'NKE'
start_date = '2018-04-02'
end_date = '2023-04-02'
interval = '1d'
data_col = 'Close'

# Download stock price from Yahoo Finance
stock_price = yf.download(ticker, start=start_date, end=end_date, interval=interval)[data_col]
stock_price = pd.DataFrame(stock_price)
stock_price.head()


# Check null
stock_price.isnull().sum()

# Create the plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(stock_price.index, stock_price.values)

# Format the x-axis ticks to show the full date
date_fmt = '%Y-%m-%d'
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_fmt))

# Set labels and title
ax.set_xlabel('Date', size=10)
ax.set_ylabel('Stock Price', size=10)
ax.set_title('Nike Stock Price Close', size=10)

# Display the plot
plt.show()


# Normalize the data
sc = MinMaxScaler(feature_range=(-1, 1))
stock_price['Close'] = sc.fit_transform(stock_price['Close'].values.reshape(-1, 1))
stock_price_scaled = stock_price
stock_price.head()

# Train/Test split
split_ratio = 0.8
split_pos = int(split_ratio*len(stock_price))

train_set = stock_price.iloc[:split_pos].values
test_set = stock_price.iloc[split_pos:].values

# Custom dataset
class Stock_Dataset(Dataset):
  def __init__(self, data, seq_len):
    self.data = data
    self.seq_len = seq_len

  def __len__(self):
    return len(self.data) - self.seq_len #-1

  def __getitem__(self, index):
    # Create sequence for LSTM model
    x_seq = self.data[index:index+self.seq_len]
    y_seq = self.data[index+self.seq_len]
    # Return tensors
    return torch.tensor(x_seq).float(), torch.tensor(y_seq).float()


seq_len = 16
batch_size = 16

# Create train/test dataset
train_dataset = Stock_Dataset(train_set, seq_len)
test_dataset = Stock_Dataset(test_set, seq_len)

# Create train/test data loader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class LSTM_Forecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=30, output_size=1, lstm_layers=1):
        super(LSTM_Forecast, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers

        # Create LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers)
        # Create fully connected layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Define states
        h_st = torch.zeros(self.lstm_layers, input.size(1), self.hidden_size)  # LSTM hidden state
        c_st = torch.zeros(self.lstm_layers, input.size(1), self.hidden_size)  # LSTM cell state

        # Outputs
        out, (hn, cn) = self.lstm(input, (h_st, c_st))
        out = self.linear(out[-1])

        return out

 # Initialize model
model = LSTM_Forecast()
model

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
learning_rate = 5e-5
epochs = 100
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = tqdm(range(epochs))

# Train/validation losses
train_total_loss = []
test_total_loss = []

# Train and evaluate model
for epoch in pbar:
    train_loss = 0
    test_loss = 0

    # Set the model to training state
    model.train()

    # Train the model on the train dataset
    for i, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.squeeze().to(device)

        # Forward pass
        tr_outputs = model(inputs).squeeze()
        tr_loss = criterion(tr_outputs, targets)
        train_loss += tr_loss.item()
        tr_loss.backward()
        optimizer.step()

    # Calculate total training loss
    tr_epoch_loss = train_loss / len(train_dataloader)
    train_total_loss.append(tr_epoch_loss)

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model on the test dataset
    for i, (inputs, targets) in enumerate(test_dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.squeeze().to(device)

            te_outputs = model(inputs).squeeze()
            te_loss = criterion(te_outputs, targets)

        test_loss += te_loss.item()

    # Calculate total test loss
    te_epoch_loss = test_loss / len(test_dataloader)
    test_total_loss.append(te_epoch_loss)

    # Print stats at each epoch
    if (epoch % 1 == 0):
        pbar.set_description(f"Epoch {epoch + 1}/{epochs} \t")
        print(f'Train Loss: {tr_loss.item():.4f}, Test Loss: {te_loss.item():.4f}')


    def predict(model, stock_dataloader):
        model.eval()

        act_vals = []
        pred_vals = []

        # Predict stock values
        for inputs, targets in stock_dataloader:
            with torch.no_grad():
                pred_outputs = model(inputs)

                # Store actual and predicted values
                act_vals.append(targets.squeeze())
                pred_vals.append(pred_outputs)

        # Add tensors
        act_vals = torch.cat(act_vals).numpy()
        pred_vals = torch.cat(pred_vals).numpy().squeeze()

        return act_vals, pred_vals


    # Predict with actual dataset
    stock_dataset = Stock_Dataset(stock_price.values, seq_len)
    stock_dataloader = DataLoader(stock_dataset, batch_size=batch_size, drop_last=True)

    # Save actual and predicted values
    acts, preds = predict(model, stock_dataloader)

    # Inverse transform
    acts_it = sc.inverse_transform(acts.reshape(-1, 1))
    preds_it = sc.inverse_transform(preds.reshape(-1, 1))

    # Root Mean Square Error
    print(f'Train RMSE: {torch.sqrt(tr_loss).item():.4f}')
    print(f'Test RMSE: {torch.sqrt(te_loss).item():.4f}')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(train_total_loss, label='Train Loss')
    ax.plot(test_total_loss, label='Test Loss')

    # Set the x-axis label and title
    ax.set_xlabel('Epochs', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.set_title('Train/Test Loss', size=15)

    # Display the plot
    plt.legend()
    plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(acts, label='Actual')
ax.plot(preds, label='Predicted')

# Set the x-axis label and title
#ax.set_xlabel('Date', size=15)
ax.set_ylabel('Stock Price', size=10)
ax.set_title('Actual vs Predicted Stock Price', size=10)

# Display the plot
plt.legend()
plt.show()


# Forecast with predicted dataset
def forecast(fo_stock_data, sequence_len, fo_days):
    stock_price_last_seq_values = np.append(fo_stock_data[-(sequence_len - 1):].values, preds[-1]).reshape(-1, 1)
    forecast_data = []

    for i in range(fo_days):
        stock_dataset_fo = Stock_Dataset(stock_price_last_seq_values, 16)
        stock_dataloader_fo = DataLoader(stock_dataset_fo, batch_size=16, drop_last=True)

        with torch.no_grad():
            acts_fo, preds_fo = predict(model, stock_dataloader_fo)

        stock_price_last_seq_values = np.append(stock_price_last_seq_values[-(sequence_len - 1):],
                                                preds_fo[-1]).reshape(-1, 1)
        forecast_data.append(preds_fo[-1])

    acts_fo_it = sc.inverse_transform(acts_fo.reshape(-1, 1))
    preds_fo_it = sc.inverse_transform(preds_fo.reshape(-1, 1))
    forecast_data_it = sc.inverse_transform(np.array(forecast_data).reshape(-1, 1))
    forecast_data_it = np.append(([np.nan] * len(fo_stock_data)), forecast_data_it)

    return forecast_data_it

forecast_days = 90
sequence_len = 32

fo_price = forecast(stock_price, sequence_len, forecast_days)

# Create the plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(acts_it, label='Actual')
ax.plot(preds_it, label='Predicted')
ax.plot(fo_price, label='Forecasts')

# Set the x-axis label and title
#ax.set_xlabel('Date', size=15)
ax.set_ylabel('Stock Price', size=10)
ax.set_title('Actual vs Predicted Stock Price', size=10)

# Display the plot
plt.legend()
plt.show()