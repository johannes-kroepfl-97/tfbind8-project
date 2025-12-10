import os
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from ps_functions.mlp_model import train_mlp_from_config, MLPRegressor
from ps_functions.cnn_model import CNNRegressor
from ps_functions.lstm_model import LSTMRegressor
from ps_functions.cnn_lstm_model import CNNLSTMRegressor
from ps_functions.lstm_cnn_model import LSTMCNNRegressor
from ps_functions.transformer_model import TransformerRegressor
from ps_functions.train_model import prepare_data

MODEL_LOADERS = {
    "mlp": MLPRegressor,
    "cnn": CNNRegressor,
    "lstm": LSTMRegressor,
    "cnn_lstm": CNNLSTMRegressor,
    "lstm_cnn": LSTMCNNRegressor,
    "transformer": TransformerRegressor
}

def load_best_model(model_name, folder, input_dim):
    config_path = Path(folder) / "models" / f"{model_name}_config.json"
    state_path = Path(folder) / "models" / f"{model_name}_best.pt"

    with open(config_path) as f:
        config = json.load(f)

    model_class = MODEL_LOADERS[model_name]

    model_kwargs = config["model"].copy()

    if model_name == "mlp":
        num_layers = model_kwargs.pop("num_layers")
        hidden_dim = model_kwargs.pop("hidden_dims")
        model_kwargs["hidden_dims"] = [hidden_dim] * num_layers
        model = model_class(input_dim=input_dim, **model_kwargs)

    elif model_name == "cnn":
        model = model_class(input_channels=4, **model_kwargs)

    elif model_name == "transformer":
        model = model_class(input_dim=input_dim, **model_kwargs)

    else:
        model = model_class(input_dim=input_dim, **model_kwargs)

    model.load_state_dict(torch.load(state_path, map_location=torch.device("cpu")))
    model.eval()
    return model, config

def load_all_models(model_name, folder, input_dim):
    
    models_list = list()
    config_list = list()
    i = 0
    
    while True:
        
        try:
            config_path = Path(folder) / "models" / f"{model_name}_config_{i}.json"
            state_path = Path(folder) / "models" / f"{model_name}_{i}.pt"

            with open(config_path) as f:
                config = json.load(f)

            model_class = MODEL_LOADERS[model_name]

            model_kwargs = config["model"].copy()

            if model_name == "mlp":
                num_layers = model_kwargs.pop("num_layers")
                hidden_dim = model_kwargs.pop("hidden_dims")
                model_kwargs["hidden_dims"] = [hidden_dim] * num_layers
                model = model_class(input_dim=input_dim, **model_kwargs)

            elif model_name == "cnn":
                model = model_class(input_channels=4, **model_kwargs)

            elif model_name == "transformer":
                model = model_class(input_dim=input_dim, **model_kwargs)

            else:
                model = model_class(input_dim=input_dim, **model_kwargs)
            
            model.load_state_dict(torch.load(state_path, map_location=torch.device("cpu")))
            model.eval()
            
            models_list.append(model)
            config_list.append(config)
            
            i += 1
            
        except Exception as e:
            break
    
    return models_list, config

'''
def evaluate_model(model, x_data, y_true, batch_size=128):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X = torch.tensor(x_data, dtype=torch.float32).to(device)
    y = torch.tensor(y_true, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X).squeeze()

    mae = mean_absolute_error(y.cpu().numpy(), y_pred.cpu().numpy())
    return mae
'''

def evaluate_model(model, x_data, y_true, batch_size=128):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X = torch.tensor(x_data, dtype=torch.float32).to(device)
    y = torch.tensor(y_true, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X).squeeze()

    y_true_np = np.array(y.cpu().cpu().detach().tolist()) # .numpy()
    y_pred_np = np.array(y_pred.cpu().detach().tolist()) # .numpy()

    mae = mean_absolute_error(y_true_np, y_pred_np)
    std = np.std(np.abs(y_pred_np - y_true_np))

    return mae, std

'''
def run_evaluation(folder, model_names, x_test, y_test, x_ood, y_ood):
    # convert test data using prepare_data just to get correct input_dim
    _, _, _, _, input_dim = prepare_data(x_test, y_test, x_test, y_test)

    results = []

    for model_name in model_names:
        print(f"Evaluating: {model_name}")
        model, config = load_best_model(model_name, folder, input_dim)

        # Prepare input depending on model type
        if model_name == "cnn":
            one_hot = lambda x: (x[..., None] == np.arange(4)).astype(np.float32)
        else:
            one_hot = lambda x: prepare_data(x, y_test, x, y_test)[0].numpy()

        X_test_encoded = one_hot(x_test)
        X_ood_encoded = one_hot(x_ood)

        # mae_test = evaluate_model(model, X_test_encoded, y_test)
        # mae_ood = evaluate_model(model, X_ood_encoded, y_ood)

        mae_test, std_test = evaluate_model(model, X_test_encoded, y_test)
        mae_ood, std_ood = evaluate_model(model, X_ood_encoded, y_ood)

        # print(f" {model_name}: Test MAE = {mae_test:.4f}, OOD MAE = {mae_ood:.4f}")

        print(f" {model_name}: Test MAE = {mae_test:.4f}, std: {std_test:.4f}, OOD MAE = {mae_ood:.4f},  std: {std_ood:.4f}")
        
        results.append({
            "model": model_name,
            "test_mae": mae_test,
            "test_std": std_test,
            "ood_mae": mae_ood,
            "ood_std": std_ood
        })


    return results
'''

def run_evaluation(
    folder,
    model_names,
    x_test,
    y_test,
    x_ood_7,
    y_ood_7,
    x_ood_8,
    y_ood_8,
    final_evaluation=False
):
    # Prepare data for determining input_dim
    _, _, _, _, input_dim = prepare_data(x_test, y_test, x_test, y_test)

    results = []

    for model_name in model_names:
        print(f"Evaluating {model_name}: {model_name}")
        
        if not final_evaluation:
            best_model, best_config = load_best_model(model_name, folder, input_dim)
            models_list = [best_model]
        else:
            models_list, _ = load_all_models(model_name, folder, input_dim)
        
        # Model-specific input encoding
        if model_name == "cnn":
            one_hot = lambda x: (x[..., None] == np.arange(4)).astype(np.float32)
        else:
            one_hot = lambda x: prepare_data(x, y_test, x, y_test)[0] # .numpy()
        
        X_test_encoded = one_hot(x_test)
        X_ood_7_encoded = one_hot(x_ood_7)
        X_ood_8_encoded = one_hot(x_ood_8)
        
        for model_num, model in enumerate(models_list):
            
            # Evaluate all sets
            mae_test, std_test = evaluate_model(model, X_test_encoded, y_test)
            mae_ood_7, std_ood_7 = evaluate_model(model, X_ood_7_encoded, y_ood_7)
            mae_ood_8, std_ood_8 = evaluate_model(model, X_ood_8_encoded, y_ood_8)
            
            # compute standard errors
            se_test = std_test / torch.sqrt(torch.tensor(len(y_test), dtype=torch.float))
            se_ood_7 = std_ood_7 / torch.sqrt(torch.tensor(len(y_ood_7), dtype=torch.float))
            se_ood_8 = std_ood_8 / torch.sqrt(torch.tensor(len(y_ood_8), dtype=torch.float))
            
            #print(f" {model_name}: Test MAE = {mae_test:.4f}, std = {std_test:.4f}, "
            #      f"OOD 7 MAE = {mae_ood_7:.4f}, std = {std_ood_7:.4f}, "
            #      f"OOD 8 MAE = {mae_ood_8:.4f}, std = {std_ood_8:.4f}")
            
            print(
                f"{model_name} #{model_num}: "
                f"Test MAE = {mae_test:.4f} (SD {std_test:.4f}, SE {se_test:.4f}); "
                f"OOD7 MAE = {mae_ood_7:.4f} (SD {std_ood_7:.4f}, SE {se_ood_7:.4f}); "
                f"OOD8 MAE = {mae_ood_8:.4f} (SD {std_ood_8:.4f}, SE {se_ood_8:.4f})"
            )
            
            results.append({
                
                "model": model_name,
                "model_num": str(model_num),
                
                "test_mae": mae_test,
                "test_std": std_test,
                "test_se": se_test,
                
                "ood_7_mut_mae": mae_ood_7,
                "ood_7_mut_std": std_ood_7,
                "ood_7_mut_se": se_ood_7,
                
                "ood_8_mut_mae": mae_ood_8,
                "ood_8_mut_std": std_ood_8,
                "ood_8_mut_se": se_ood_8
                
            })
    
    return results


def predict_y_values(folder, model_names, x_test):
    """
    Loads models from folder and predicts y-values for x_test only.
    Returns a dictionary {model_name: y_pred_array}.
    """

    results = {}

    # prepare data once to determine input_dim
    _, _, _, _, input_dim = prepare_data(x_test, np.zeros(len(x_test)), x_test, np.zeros(len(x_test)))

    for model_name in model_names:
        print(f"Predicting with model: {model_name}")

        # Load model + config
        model, config = load_best_model(model_name, folder, input_dim)
        model.eval()

        # Prepare one-hot encoding depending on model type
        if model_name == "cnn":
            one_hot = lambda x: (x[..., None] == np.arange(4)).astype(np.float32)
        else:
            one_hot = lambda x: prepare_data(x, np.zeros(len(x)), x, np.zeros(len(x)))[0]

        # Encode x_test
        X_test_encoded = one_hot(x_test)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        X_test_encoded_torch = torch.tensor(X_test_encoded, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_pred = model(X_test_encoded_torch).squeeze()
        
        y_pred_np = np.array(y_pred.cpu().detach().tolist())

        results[model_name] = y_pred_np
        print(f" → Done: {model_name}, predicted {len(y_pred_np)} samples.")

    return results

