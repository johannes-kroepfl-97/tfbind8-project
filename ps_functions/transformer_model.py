'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ps_functions.train_model import prepare_data, train_model


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = 4  # one-hot encoding (A/C/G/T)

        self.embedding = nn.Linear(self.input_channels, d_model)

        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))  # (seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, self.input_channels)  # (B, 8, 4)
        x = self.embedding(x) + self.positional_encoding  # (B, 8, d_model)
        x_encoded = self.encoder(x)  # (B, 8, d_model)
        x_last = x_encoded[:, -1, :]  # (B, d_model)
        return self.regressor(x_last)


def train_transformer_from_config(config, x_train, y_train, x_val, y_val):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    model = TransformerRegressor(
        input_dim=input_dim,
        seq_len=config['model']['seq_len'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'])
    return trained_model, best_mae, history
'''

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from ps_functions.train_model import prepare_data, train_model

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, nhead, num_layers, dim_feedforward, dropout_input, dropout_hidden):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = 4

        self.input_dropout = nn.Dropout(dropout_input)
        self.embedding = nn.Linear(self.input_channels, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_hidden,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, self.input_channels)
        x = self.input_dropout(x)
        x = self.embedding(x) + self.positional_encoding
        x_encoded = self.encoder(x)
        x_last = x_encoded[:, -1, :]
        return self.regressor(x_last)

def train_transformer_from_config(config, x_train, y_train, x_val, y_val, make_prints=True):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config['training']['batch_size'], shuffle=True)

    model = TransformerRegressor(
        input_dim=input_dim,
        seq_len=config['model']['seq_len'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'], make_prints=make_prints)
    return trained_model, best_mae, history