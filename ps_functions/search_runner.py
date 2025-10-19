'''
import os
import itertools
import torch
import json

def grid_search(train_fn, base_config, param_grid, x_train, y_train, x_val, y_val, save_path, model_name):
    """
    Runs a grid search and saves the best model.
    """
    keys, values = zip(*param_grid.items())
    best_mae = float("inf")
    best_model = None
    best_config = None

    for i, combo in enumerate(itertools.product(*values)):
        config = json.loads(json.dumps(base_config))  # deep copy
        for k, v in zip(keys, combo):
            section, param = k.split(".")
            config[section][param] = v

        print(f"\n=== [{model_name} Try {i+1}] Config: {config}")
        model, mae = train_fn(config, x_train, y_train, x_val, y_val)
        print(f">>> Val MAE: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_config = config

            # Save model
            torch.save(model.state_dict(), os.path.join(save_path, f"best_{model_name.lower()}.pt"))
            with open(os.path.join(save_path, f"best_{model_name.lower()}_config.json"), "w") as f:
                json.dump(config, f, indent=2)

    return best_model, best_config, best_mae
'''

import os
import itertools
import torch
import json
import pandas as pd

def grid_search(train_fn, base_config, param_grid, x_train, y_train, x_val, y_val, save_path, model_name, run_timestamp):
    keys, values = zip(*param_grid.items())
    best_mae = float("inf")
    best_model = None
    best_config = None
    results = []

    # Use the shared timestamp for this run
    run_dir = os.path.join(save_path, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    for i, combo in enumerate(itertools.product(*values)):
        config = json.loads(json.dumps(base_config))  # deep copy
        for k, v in zip(keys, combo):
            section, param = k.split(".")
            config[section][param] = v

        print(f"\n=== [{model_name} Try {i+1}] Config: {config}")
        model, mae, history = train_fn(config, x_train, y_train, x_val, y_val)
        print(f">>> Val MAE: {mae:.4f}")

        results.append({
            "run": i + 1,
            "val_mae": mae,
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            **{f"{k}.{p}": config[k][p] for k in config for p in config[k]}
        })

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_config = config
            torch.save(model.state_dict(), os.path.join(run_dir, f"best_{model_name.lower()}.pt"))
            with open(os.path.join(run_dir, f"best_{model_name.lower()}_config.json"), "w") as f:
                json.dump(config, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, f"{model_name.lower()}_results.csv"), index=False)

    return best_model, best_config, best_mae
