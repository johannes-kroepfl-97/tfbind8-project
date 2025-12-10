'''
import os
import json
import random
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

from ps_functions.mlp_model import train_mlp_from_config
from ps_functions.cnn_model import train_cnn_from_config
from ps_functions.lstm_model import train_lstm_from_config
from ps_functions.lstm_cnn_model import train_lstm_cnn_from_config
from ps_functions.cnn_lstm_model import train_cnn_lstm_from_config
from ps_functions.transformer_model import train_transformer_from_config

def load_search_space(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def sample_config(search_space):
    config = {}
    for section, params in search_space.items():
        config[section] = {k: random.choice(v) for k, v in params.items()}
    return config

def random_search(train_functions, search_space_paths, x_train, y_train, x_val, y_val, n_trials=10):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    base_dir = Path("results") / "training" / f"random_search_{timestamp}"
    model_dir = base_dir / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    config_rows = []

    for model_name, train_fn in train_functions.items():
        print(f"\n🚀 Starting random search for {model_name}")
        search_space = load_search_space(search_space_paths[model_name])

        best_model = None
        best_mae = float("inf")
        best_config = None
        best_std = None
        best_history = None

        for trial in range(n_trials):
            config = sample_config(search_space)
            model, mae, history = train_fn(config, x_train, y_train, x_val, y_val)
            std = history["val_error_std"][-1]

            summary_rows.append({
                "model_class": model_name,
                "trial": trial,
                "best_mae": mae,
                "val_std": std,
                "history": history
            })

            config_rows.append({
                "model_class": model_name,
                "trial": trial,
                **{f"{section}.{param}": value for section in config for param, value in config[section].items()}
            })

            if mae < best_mae:
                best_mae = mae
                best_config = config
                best_model = model
                best_std = std
                best_history = history

        # Save best model + config
        model_path = model_dir / f"{model_name}_best.pt"
        torch.save(best_model.state_dict(), model_path)

        with open(model_dir / f"{model_name}_config.json", 'w') as f:
            json.dump(best_config, f, indent=2)

    # Save run summaries
    pd.DataFrame(summary_rows).to_csv(base_dir / "summary.csv", index=False)
    pd.DataFrame(config_rows).to_csv(base_dir / "configurations.csv", index=False)

    print(f"\n✅ Random search finished. Results saved in: {base_dir}")
'''

import os
import json
import random
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

from ps_functions.mlp_model import train_mlp_from_config
from ps_functions.cnn_model import train_cnn_from_config
from ps_functions.lstm_model import train_lstm_from_config
from ps_functions.lstm_cnn_model import train_lstm_cnn_from_config
from ps_functions.cnn_lstm_model import train_cnn_lstm_from_config
from ps_functions.transformer_model import train_transformer_from_config

# Load shared training defaults
with open("search_spaces/shared_training_config.json", "r") as f:
    shared_training_config = json.load(f)

def load_search_space(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def sample_config(search_space, shared_training):
    config = {}
    for section, params in search_space.items():
        config[section] = {k: random.choice(v) for k, v in params.items()}
    
    # Inject shared training values
    config.setdefault("training", {}).update(shared_training)
    return config

def random_search(train_functions, search_space_paths, x_train, y_train, x_val, y_val, n_trials=10, evaluation=False):
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    base_dir = Path("results") / "training" / f"random_search_{timestamp}" if not evaluation else Path("results") / "evaluation" / f"evaluation_top_config_{n_trials}_times_{timestamp}"
    
    print(f"\n Creating folder: {base_dir}")
    
    model_dir = base_dir / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    config_rows = []

    for model_name, train_fn in train_functions.items():
        print(f"\n --> Starting random search for {model_name}")
        search_space = load_search_space(search_space_paths[model_name])

        best_model = None
        best_mae = float("inf")
        best_config = None
        best_std = None
        best_history = None

        for trial in range(n_trials):
            
            print(f"Run [{trial}]", end=' ')
            
            config = sample_config(search_space, shared_training_config)
            model, mae, history = train_fn(config, x_train, y_train, x_val, y_val, make_prints=False)
            std = history["val_error_std"][-1]

            summary_rows.append({
                "model_class": model_name,
                "trial": trial,
                "best_mae": mae,
                "val_error_std": std,
                "history": history
            })

            config_rows.append({
                "model_class": model_name,
                "trial": trial,
                **{f"{section}.{param}": value for section in config for param, value in config[section].items()}
            })
            
            if not evaluation:
                if mae < best_mae:
                    best_mae = mae
                    best_config = config
                    best_model = model
                    best_std = std
                    best_history = history
            else:
                
                model_path = model_dir / f"{model_name}_{trial}.pt"
                torch.save(model.state_dict(), model_path)
                
                with open(model_dir / f"{model_name}_config_{trial}.json", 'w') as f:
                    json.dump(config, f, indent=2)
            
            print(f"mae: {round(mae, 4)}")
        
        if not evaluation:
            # Save best model + config
            model_path = model_dir / f"{model_name}_best.pt"
            torch.save(best_model.state_dict(), model_path)

            with open(model_dir / f"{model_name}_config.json", 'w') as f:
                json.dump(best_config, f, indent=2)

            print(f"\n Best config for {model_name}:\n{best_config}\n")

    # Save run summaries
    pd.DataFrame(summary_rows).to_csv(base_dir / "summary.csv", index=False)
    pd.DataFrame(config_rows).to_csv(base_dir / "configurations.csv", index=False)

    print(f"\n Random search finished. Results saved to {base_dir}")
    
    return base_dir