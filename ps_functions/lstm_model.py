import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ps_functions.train_model import prepare_data, train_model

'''
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.seq_len = input_dim // 4  # since one-hot encoded: 4 classes
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, 4)  # reshape to (batch, seq_len, 4)
        _, (hn, _) = self.lstm(x)
        out = hn[-1] if hn.shape[0] == 1 else hn[-2:].transpose(0, 1).reshape(x.shape[0], -1)
        return self.fc(out)

def train_lstm_from_config(config, x_train, y_train, x_val, y_val):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model'].get('bidirectional', False)
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'])
    return trained_model, best_mae, history
'''

'''
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_input, dropout_hidden, bidirectional, fc_dim):
        super().__init__()
        self.seq_len = input_dim // 4
        self.input_dropout = nn.Dropout(dropout_input)

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # We define fc later using lazy init
        self.fc_dim = fc_dim
        self.dropout_hidden = dropout_hidden
        self.fc = None  # placeholder

    def forward(self, x):
        x = self.input_dropout(x.view(x.shape[0], self.seq_len, 4))
        _, (hn, _) = self.lstm(x)

        if self.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # [batch, 2*hidden_dim]
        else:
            out = hn[-1]  # [batch, hidden_dim]

        # Dynamically create fc block the first time we know the input dim
        if self.fc is None:
            input_dim = out.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(input_dim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_hidden),
                nn.Linear(self.fc_dim, 1)
            ).to(out.device)  # move to correct device

        return self.fc(out)

'''

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_input, dropout_hidden, bidirectional, fc_dim):
        super().__init__()
        self.seq_len = input_dim // 4
        self.input_dropout = nn.Dropout(dropout_input)

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)

        # Eagerly initialize fc layer
        self.fc = nn.Sequential(
            nn.Linear(output_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        x = self.input_dropout(x.view(x.shape[0], self.seq_len, 4))
        _, (hn, _) = self.lstm(x)

        if self.lstm.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # shape: [batch, 2*hidden]
        else:
            out = hn[-1]  # shape: [batch, hidden]

        return self.fc(out)

   
def train_lstm_from_config(config, x_train, y_train, x_val, y_val, make_prints=True):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config['training']['batch_size'], shuffle=True)

    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden'],
        bidirectional=config['model']['bidirectional'],
        fc_dim=config['model']['fc_dim']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'], make_prints=make_prints)
    return trained_model, best_mae, history