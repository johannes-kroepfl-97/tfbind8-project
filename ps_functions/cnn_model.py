import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from train_model import prepare_data, train_model
from ps_functions.train_model import prepare_data, train_model
import torch
import torch.nn.functional as F

'''
class CNNRegressor(nn.Module):
    def __init__(self, input_channels, channels=[32, 64, 128], kernel_size=3, dropout=0.3):
        super().__init__()
        conv_layers = []
        for i in range(len(channels)):
            in_ch = input_channels if i == 0 else channels[i-1]
            conv_layers += [
                nn.Conv1d(in_ch, channels[i], kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.pool(self.conv(x))
        x = x.squeeze(-1)
        return self.fc(x)


def train_cnn_from_config(config, x_train, y_train, x_val, y_val):
    # CNN-specific one-hot processing
    def prepare_for_cnn(x):
        return torch.tensor((x[..., None] == torch.arange(4)).float().numpy(), dtype=torch.float32)

    X_train = prepare_for_cnn(torch.tensor(x_train))
    X_val = prepare_for_cnn(torch.tensor(x_val))

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    input_channels = X_train.shape[-1]
    model = CNNRegressor(input_channels=input_channels, dropout=config['model']['dropout'])

    trained_model, best_mae, history = train_model(model, train_loader, (X_val, y_val_t), config['training'])
    return trained_model, best_mae, history
'''

class CNNRegressor(nn.Module):
    def __init__(self, input_channels, channels=[32, 64], kernel_size=3, dropout_input=0.2, dropout_hidden=0.5, fc_dim=64):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout_input)

        conv_layers = []
        for i in range(len(channels)):
            in_ch = input_channels if i == 0 else channels[i - 1]
            conv_layers += [
                nn.Conv1d(in_ch, channels[i], kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_hidden)
            ]

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_hidden),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        x = self.input_dropout(x)
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.pool(self.conv(x))
        return self.fc(x)

def train_cnn_from_config(config, x_train, y_train, x_val, y_val, make_prints=True):
    
    '''
    def prepare_for_cnn(x):
        return torch.tensor((x[..., None] == torch.arange(4)).float().numpy(), dtype=torch.float32)

    X_train = prepare_for_cnn(torch.tensor(x_train, dtype=int))
    X_val = prepare_for_cnn(torch.tensor(x_val, dtype=int))
    y_train_t = torch.tensor(y_train, dtype=float)
    y_val_t = torch.tensor(y_val, dtype=float)
    '''
    
    def prepare_for_cnn(x):
        # x: array/list of integer indices in {0,1,2,3}, shape (B, L)
        x = torch.as_tensor(x, dtype=torch.long)                 # indices!
        one_hot = F.one_hot(x, num_classes=4).to(torch.float32)  # (B, L, 4)
        return one_hot
    
    X_train = prepare_for_cnn(x_train)
    X_val   = prepare_for_cnn(x_val)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    y_val_t   = torch.as_tensor(y_val,   dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train_t), batch_size=config['training']['batch_size'], shuffle=True)
    
    model = CNNRegressor(
        input_channels=X_train.shape[-1],
        channels=config['model']['channels'],
        kernel_size=config['model']['kernel_size'],
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden'],
        fc_dim=config['model']['fc_dim']
    )
    
    trained_model, best_mae, history = train_model(model, train_loader, (X_val, y_val_t), config['training'], make_prints=make_prints)
    return trained_model, best_mae, history
