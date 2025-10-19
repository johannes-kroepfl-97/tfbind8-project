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

def train_model(model, train_loader, val_tensors, config, make_prints=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    X_val_t, y_val_t = val_tensors
    X_val_t = X_val_t.to(device)
    y_val_t = y_val_t.to(device)
    
    loss_fn = nn.MSELoss() if config['loss'] == 'mse' else nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=make_prints
    )

    best_mae = float('inf')
    best_state = None
    best_epoch = 0
    patience = config.get('early_stopping_patience', 20)
    no_improve_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_error_std": []
    }
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        
        '''
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = loss_fn(pred_val, y_val_t).item()
            val_mae = mean_absolute_error(y_val_t.numpy(), pred_val.numpy())
            val_error_std = np.std((pred_val - y_val_t).numpy())
        '''
        
        with torch.no_grad():
            # make sure tensors are on the same device as the model
            device = next(model.parameters()).device
            X_val_t = X_val_t.to(device)
            y_val_t = y_val_t.to(device)

            # forward pass on-device
            pred_val = model(X_val_t)

            # compute loss on-device (no cpu() yet!)
            val_loss = loss_fn(pred_val, y_val_t).item()

            # ==== Option A: pure-PyTorch metrics (no numpy/sklearn) ====
            diff = (pred_val - y_val_t)
            val_mae = torch.mean(torch.abs(diff)).item()
            val_error_std = torch.std(diff).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_error_std"].append(val_error_std)

        scheduler.step(val_loss)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict()
            best_epoch = epoch
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if (epoch + 1) % config['eval_every'] == 0 and make_prints==True:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        if no_improve_counter >= patience and make_prints==True:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break
        
        # print("xb:", xb.dtype, xb.device, xb.is_cuda)
        # print("model device:", next(model.parameters()).device)
        # print({p.device for p in model.parameters()})  # should be {'cuda:0'}


    if best_state:
        model.load_state_dict(best_state)

    # Return model, best score, and training history
    return model, best_mae, history

