import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from train_model import prepare_data, train_model
from ps_functions.train_model import prepare_data, train_model


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_input, dropout_hidden):
        super().__init__()
        layers = [nn.Dropout(dropout_input)]
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_hidden))
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output_layer(x)

'''
def train_mlp_from_config(config, x_train, y_train, x_val, y_val):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    input_dropout_values = config['model']['dropout_input']
    if not isinstance(input_dropout_values, list):
        input_dropout_values = [input_dropout_values]

    best_mae = float("inf")
    best_model = None

    for dp_in in input_dropout_values:
        model = MLPRegressor(
            input_dim=input_dim,
            hidden_dims=config['model']['hidden_dims'],
            dropout_input=dp_in,
            dropout_hidden=config['model']['dropout_hidden'],
            use_residual=config['model'].get('use_residual', False)
        )
        trained_model, val_mae, _ = train_model(model, train_loader, (X_val_t, y_val_t), config['training'])
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = trained_model

    return best_model, best_mae


def train_mlp_from_config(config, x_train, y_train, x_val, y_val):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'])
    return trained_model, best_mae, history
'''

def train_mlp_from_config(config, x_train, y_train, x_val, y_val, make_prints=True):
    X_train_t, y_train_t, X_val_t, y_val_t, input_dim = prepare_data(x_train, y_train, x_val, y_val)
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)

    dims = [config['model']['hidden_dims']] * config['model']['num_layers']

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=dims,
        dropout_input=config['model']['dropout_input'],
        dropout_hidden=config['model']['dropout_hidden']
    )

    trained_model, best_mae, history = train_model(model, train_loader, (X_val_t, y_val_t), config['training'], make_prints=make_prints)
    return trained_model, best_mae, history