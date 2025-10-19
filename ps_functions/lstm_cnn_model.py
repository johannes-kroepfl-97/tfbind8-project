'''
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ps_functions.train_model import prepare_data, train_model


class LSTMCNNRegressor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, lstm_dropout, cnn_channels, kernel_size, fc_dim=64):
        super().__init__()
        self.seq_len = input_dim // 4
        self.input_channels = 4  # One-hot encoded A/C/G/T

        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=2 * lstm_hidden_dim, out_channels=cnn_channels[0], kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output shape: (batch, channels, 1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels[1], fc_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, self.input_channels)  # (B, seq_len, 4)
        lstm_out, _ = self.lstm(x)  # (B, seq_len, 2*hidden)
        lstm_out = lstm_out.transpose(1, 2)  # (B, 2*hidden, seq_len) for Conv1d
        features = self.conv(lstm_out)       # (B, channels, 1)
        return self.fc(features)


def train_lstm_cnn_from_config(config, x_train, y_train, x_val, y_val):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    model = LSTMCNNRegressor(
        input_dim=input_dim,
        lstm_hidden_dim=config['model']['lstm_hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        lstm_dropout=config['model']['lstm_dropout'],
        cnn_channels=config['model']['cnn_channels'],
        kernel_size=config['model']['kernel_size'],
        fc_dim=config['model'].get('fc_dim', 64)
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'])
    return trained_model, best_mae, history
'''

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from ps_functions.train_model import prepare_data, train_model

class LSTMCNNRegressor(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, dropout_input, dropout_hidden, bidirectional, cnn_channels, kernel_size, fc_dim):
        super().__init__()
        self.seq_len = input_dim // 4
        self.input_channels = 4

        self.input_dropout = nn.Dropout(dropout_input)

        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_hidden if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out_channels = 2 * lstm_hidden_dim if bidirectional else lstm_hidden_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=lstm_out_channels, out_channels=cnn_channels[0], kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels[1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        x = self.input_dropout(x.view(x.shape[0], self.seq_len, self.input_channels))
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(1, 2)
        conv_out = self.conv(lstm_out)
        return self.fc(conv_out)

def train_lstm_cnn_from_config(config, x_train, y_train, x_val, y_val, make_prints=True):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config['training']['batch_size'], shuffle=True)

    model = LSTMCNNRegressor(
        input_dim=input_dim,
        lstm_hidden_dim=config['model']['lstm_hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden'],
        bidirectional=config['model']['bidirectional'],
        cnn_channels=config['model']['cnn_channels'],
        kernel_size=config['model']['kernel_size'],
        fc_dim=config['model']['fc_dim']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'], make_prints=make_prints)
    return trained_model, best_mae, history
