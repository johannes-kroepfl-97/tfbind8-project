import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from ps_functions.train_model import prepare_data, train_model

class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_dim, cnn_channels, kernel_size, lstm_hidden_dim, lstm_layers,
                 dropout_input, dropout_hidden, bidirectional, fc_dim):
        super().__init__()
        self.seq_len = input_dim // 4
        self.input_channels = 4

        self.input_dropout = nn.Dropout(dropout_input)

        # CNN blocks
        conv_layers = []
        for i in range(len(cnn_channels)):
            in_ch = self.input_channels if i == 0 else cnn_channels[i - 1]
            conv_layers += [
                nn.Conv1d(in_ch, cnn_channels[i], kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_hidden)
            ]
        self.conv = nn.Sequential(*conv_layers)

        # LSTM block after CNN
        self.lstm_input_dim = cnn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_hidden if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        x = self.input_dropout(x.view(x.shape[0], self.seq_len, self.input_channels))
        x = x.permute(0, 2, 1)               # (B, C, L)
        x = self.conv(x)                     # (B, C', L)
        x = x.permute(0, 2, 1)               # (B, L, C') → for LSTM
        lstm_out, _ = self.lstm(x)           # (B, L, H)
        last_out = lstm_out[:, -1, :]        # (B, H)
        return self.fc(last_out)

def train_cnn_lstm_from_config(config, x_train, y_train, x_val, y_val, make_prints=True):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config['training']['batch_size'], shuffle=True)

    model = CNNLSTMRegressor(
        input_dim=input_dim,
        cnn_channels=config['model']['cnn_channels'],
        kernel_size=config['model']['kernel_size'],
        lstm_hidden_dim=config['model']['lstm_hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden'],
        bidirectional=config['model']['bidirectional'],
        fc_dim=config['model']['fc_dim']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'], make_prints=make_prints)
    return trained_model, best_mae, history
