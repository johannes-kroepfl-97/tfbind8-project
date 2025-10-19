import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def one_hot_encode(sequences, num_classes=4):
    return np.eye(num_classes)[sequences].reshape(len(sequences), -1)

def prepare_data(x_train, y_train, x_val, y_val):
    X_train = one_hot_encode(x_train)
    X_val = one_hot_encode(x_val)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return (
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        X_train_scaled.shape[1]
    )

def train_model(model, train_loader, val_tensors, config):
    X_val_t, y_val_t = val_tensors
    loss_fn = nn.MSELoss() if config['loss'] == 'mse' else nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    best_mae = float('inf')
    best_state = None

    for epoch in range(config['epochs']):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % config['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_t).numpy()
                val_mae = mean_absolute_error(y_val_t.numpy(), y_pred)
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)
    return model, best_mae
