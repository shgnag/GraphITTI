import torch.nn as nn
import torch.optim as optim


ELEA_inputs = {
    'models_out': '/models/ELEA',
    'features_path': '/data_processed/ELEA_OpenFace',
    # for ablation
    'train_id': [20,23, 32, 12,15, 16, 17, 22, 25, 26, 27, 28, 31,  33, 34, 35, 36, 38, 40, 14, 18],
    'val_id':[21, 30, 37, 29, 39, 24]
}

ELEA_optimization_params = {
    # Optimizer config
    'optimizer': optim.SGD, 
    'learning_rate': 0.01, #3e-2,#3e-4
    'epochs': 50,
    'step_size': 7,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 16,
    'threads': 1
}
